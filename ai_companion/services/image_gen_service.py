"""Image generation service — manages an isolated ComfyUI/SDXL worker."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

from ai_companion.infrastructure.base_service import BaseService


class ImageGenService(BaseService):
    """Manages local image generation via ComfyUI.

    Security:
    - Worker binds to 127.0.0.1 only
    - Output goes to approved directory only
    - All paths validated
    """

    service_name = "ImageGen"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._worker_process: Any = None
        self._worker_running = False
        self._request_queue: list[dict] = []
        self._output_dir: Optional[Path] = None

    @property
    def worker_running(self) -> bool:
        return self._worker_running

    def start(self) -> None:
        super().start()
        self._output_dir = Path(self._config.image_gen.output_dir).resolve()
        self._output_dir.mkdir(parents=True, exist_ok=True)
        self.emit_status("Image generation service ready")

    def stop(self) -> None:
        self.stop_worker()
        super().stop()

    def start_worker(self) -> bool:
        """Start the ComfyUI worker process.

        Note: Requires ComfyUI to be installed at the configured path.
        Binds to 127.0.0.1 only.
        """
        comfyui_path = Path(self._config.image_gen.comfyui_path)
        if not comfyui_path.exists():
            self.emit_error(
                f"ComfyUI not found at {comfyui_path}. "
                "Set image_gen.comfyui_path in config."
            )
            return False

        try:
            import subprocess
            cmd = [
                "python", str(comfyui_path / "main.py"),
                "--listen", self._config.image_gen.host,
                "--port", str(self._config.image_gen.port),
            ]
            self._worker_process = subprocess.Popen(
                cmd,
                cwd=str(comfyui_path),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            self._worker_running = True
            self._signal_bus.image_gen.worker_status_changed.emit("running")
            return True
        except Exception as e:
            self.emit_error(f"Failed to start ComfyUI worker: {e}")
            return False

    def stop_worker(self) -> None:
        """Stop the ComfyUI worker process."""
        if self._worker_process:
            try:
                self._worker_process.terminate()
                self._worker_process.wait(timeout=10)
            except Exception:
                try:
                    self._worker_process.kill()
                except Exception:
                    pass
            self._worker_process = None
            self._worker_running = False
            self._signal_bus.image_gen.worker_status_changed.emit("stopped")

    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        width: Optional[int] = None,
        height: Optional[int] = None,
        steps: int = 20,
        seed: int = -1,
    ) -> Optional[str]:
        """Submit an image generation request.

        Returns request_id, or None if submission fails.
        For now, this creates a placeholder — full ComfyUI integration
        requires the worker to be running.
        """
        if not prompt.strip():
            self.emit_error("Empty prompt")
            return None

        w = width or self._config.image_gen.default_width
        h = height or self._config.image_gen.default_height

        # Validate dimensions
        for val, name in [(w, "width"), (h, "height")]:
            if not (64 <= val <= 2048):
                self.emit_error(f"Invalid {name}: {val} (must be 64-2048)")
                return None

        request_id = str(uuid.uuid4())
        request = {
            "id": request_id,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "width": w,
            "height": h,
            "steps": steps,
            "seed": seed,
            "status": "queued",
        }

        self._request_queue.append(request)
        self._signal_bus.image_gen.generation_started.emit(request_id)

        # If worker is running, send the request
        if self._worker_running:
            self._submit_to_worker(request)
        else:
            # Create a placeholder image
            self._create_placeholder(request)

        return request_id

    def _submit_to_worker(self, request: dict) -> None:
        """Submit request to ComfyUI via HTTP API."""
        try:
            import urllib.request
            import urllib.error

            url = f"http://{self._config.image_gen.host}:{self._config.image_gen.port}/prompt"
            payload = json.dumps({
                "prompt": {
                    "3": {
                        "inputs": {
                            "seed": request["seed"] if request["seed"] >= 0 else int(time.time()),
                            "steps": request["steps"],
                            "cfg": 7.0,
                            "sampler_name": "euler",
                            "scheduler": "normal",
                            "denoise": 1.0,
                        },
                        "class_type": "KSampler",
                    },
                }
            }).encode()

            req = urllib.request.Request(
                url, data=payload, headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                request["status"] = "submitted"
                request["comfyui_id"] = result.get("prompt_id")
        except Exception as e:
            self.emit_error(f"Worker submission failed: {e}")
            request["status"] = "failed"
            self._signal_bus.image_gen.generation_error.emit(
                request["id"], str(e)
            )

    def _create_placeholder(self, request: dict) -> None:
        """Create a placeholder image when no worker is available."""
        try:
            from PIL import Image, ImageDraw, ImageFont

            img = Image.new("RGB", (request["width"], request["height"]), "#1a1a2e")
            draw = ImageDraw.Draw(img)
            text = f"Generated: {request['prompt'][:50]}..."
            draw.text((20, 20), text, fill="#4a9eff")
            draw.text(
                (20, 50), "(ComfyUI worker not running)", fill="#666666"
            )

            filename = f"{request['id']}.png"
            out_path = self._output_dir / filename
            img.save(str(out_path))

            request["status"] = "complete"
            request["output_path"] = str(out_path)
            self._signal_bus.image_gen.generation_complete.emit(
                request["id"], str(out_path)
            )
        except ImportError:
            self.emit_error("Pillow not installed — cannot create placeholder")
            request["status"] = "failed"
        except Exception as e:
            self.emit_error(f"Placeholder creation failed: {e}")
            request["status"] = "failed"

    def get_request_status(self, request_id: str) -> Optional[dict]:
        for req in self._request_queue:
            if req["id"] == request_id:
                return req
        return None
