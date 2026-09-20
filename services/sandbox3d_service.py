"""3D Sandbox service — GLB, GLTF, OBJ import and scene management."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from ai_companion.infrastructure.base_service import BaseService
from ai_companion.infrastructure.json_store import JsonStore
from ai_companion.infrastructure.path_validator import PathValidator, PathValidationError


class SceneObject(BaseModel):
    """A 3D object in the scene."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    file_path: str
    format: str  # glb, gltf, obj
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    rotation: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0])
    scale: list[float] = Field(default_factory=lambda: [1.0, 1.0, 1.0])
    visible: bool = True
    imported: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )


class Sandbox3DService(BaseService):
    """Manages a 3D scene with import support for GLB, GLTF, and OBJ files.

    Provides scene management, object placement, and basic transformations.
    Rendering is handled by the UI layer.
    """

    service_name = "Sandbox3D"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._models_dir: Optional[Path] = None
        self._scene_store: Optional[JsonStore] = None
        self._objects: dict[str, SceneObject] = {}
        self._path_validator: Optional[PathValidator] = None

    def start(self) -> None:
        super().start()
        self._models_dir = Path("sandbox3d/models").resolve()
        self._models_dir.mkdir(parents=True, exist_ok=True)

        self._path_validator = PathValidator(
            approved_roots=[str(self._models_dir)],
            allowed_extensions=self._config.sandbox3d.supported_formats,
            max_file_size_bytes=(
                self._config.sandbox3d.max_import_size_mb * 1024 * 1024
            ),
        )

        self._scene_store = JsonStore("sandbox3d/scene.json")
        self._scene_store.load()

        # Load existing objects
        for key, data in self._scene_store.items():
            try:
                self._objects[key] = SceneObject(**data)
            except Exception:
                continue

        self.emit_status(f"3D Sandbox ready: {len(self._objects)} objects")

    def stop(self) -> None:
        self._save_scene()
        super().stop()

    def import_model(
        self,
        source_path: str,
        name: str = "",
        position: Optional[list[float]] = None,
    ) -> Optional[SceneObject]:
        """Import a 3D model file into the sandbox."""
        source = Path(source_path).resolve()

        if not source.exists():
            self.emit_error(f"File not found: {source_path}")
            self._signal_bus.sandbox3d.import_error.emit(
                source_path, "File not found"
            )
            return None

        ext = source.suffix.lower()
        if ext not in self._config.sandbox3d.supported_formats:
            self.emit_error(f"Unsupported format: {ext}")
            self._signal_bus.sandbox3d.import_error.emit(
                source_path, f"Unsupported format: {ext}"
            )
            return None

        # Check size
        size = source.stat().st_size
        max_size = self._config.sandbox3d.max_import_size_mb * 1024 * 1024
        if size > max_size:
            self.emit_error(
                f"File too large: {size} bytes (max {max_size})"
            )
            self._signal_bus.sandbox3d.import_error.emit(
                source_path, "File too large"
            )
            return None

        # Copy to models directory
        import shutil
        dest = self._models_dir / source.name
        counter = 1
        while dest.exists():
            dest = self._models_dir / f"{source.stem}_{counter}{ext}"
            counter += 1

        try:
            shutil.copy2(source, dest)
        except OSError as e:
            self.emit_error(f"Import copy failed: {e}")
            return None

        # Create scene object
        obj = SceneObject(
            name=name or source.stem,
            file_path=str(dest.relative_to(Path("sandbox3d").resolve())),
            format=ext.lstrip("."),
            position=position or [0.0, 0.0, 0.0],
        )

        self._objects[obj.id] = obj
        self._save_scene()
        self._signal_bus.sandbox3d.model_imported.emit(obj.id)
        return obj

    def remove_object(self, object_id: str) -> bool:
        """Remove an object from the scene."""
        if object_id not in self._objects:
            return False
        del self._objects[object_id]
        self._scene_store.delete(object_id)
        self._scene_store.save()
        self._signal_bus.sandbox3d.model_removed.emit(object_id)
        return True

    def update_transform(
        self,
        object_id: str,
        position: Optional[list[float]] = None,
        rotation: Optional[list[float]] = None,
        scale: Optional[list[float]] = None,
    ) -> Optional[SceneObject]:
        """Update an object's transform."""
        obj = self._objects.get(object_id)
        if not obj:
            return None

        if position is not None and len(position) == 3:
            obj.position = position
        if rotation is not None and len(rotation) == 3:
            obj.rotation = rotation
        if scale is not None and len(scale) == 3:
            obj.scale = scale

        self._save_scene()
        return obj

    def set_visibility(self, object_id: str, visible: bool) -> bool:
        obj = self._objects.get(object_id)
        if not obj:
            return False
        obj.visible = visible
        self._save_scene()
        return True

    def get_object(self, object_id: str) -> Optional[SceneObject]:
        return self._objects.get(object_id)

    def list_objects(self) -> list[SceneObject]:
        return list(self._objects.values())

    def clear_scene(self) -> None:
        """Remove all objects from the scene."""
        self._objects.clear()
        self._scene_store.clear()
        self._scene_store.save()
        self._signal_bus.sandbox3d.scene_cleared.emit()

    def export_scene(self) -> dict:
        """Export the scene state as a dict."""
        return {
            "objects": [obj.model_dump() for obj in self._objects.values()],
            "exported": datetime.now(timezone.utc).isoformat(),
        }

    def _save_scene(self) -> None:
        """Persist scene state."""
        for obj_id, obj in self._objects.items():
            self._scene_store.set(obj_id, obj.model_dump())
        self._scene_store.save()
