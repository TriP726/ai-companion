"""Mind Map panel: graph view plus a detail pane for the selected concept.

The detail pane is where the Mind Map earns its keep. Selecting a bubble shows
the concept's written overview (or offers to write one), the memories that
formed it, and the concepts it connects to.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ai_companion.ui.mind_map_view import MindMapView
from ai_companion.ui.theme import theme

if TYPE_CHECKING:
    from ai_companion.infrastructure.service_manager import ServiceManager


class MindMapPanel(QWidget):
    """Concept graph with an inspector."""

    def __init__(self, service_manager: "ServiceManager") -> None:
        super().__init__()
        self._sm = service_manager
        self._setup_ui()
        self.rebuild()

    # -- construction -------------------------------------------------

    def _setup_ui(self) -> None:
        m = theme.m
        root = QVBoxLayout(self)
        root.setContentsMargins(m.space_4, m.space_4, m.space_4, m.space_4)
        root.setSpacing(m.space_3)

        toolbar = QHBoxLayout()
        toolbar.setSpacing(m.space_2)

        self._rebuild_btn = QPushButton("Rebuild")
        self._rebuild_btn.setObjectName("PrimaryButton")
        self._rebuild_btn.setToolTip(
            "Derive concepts from your approved memories. Instant - no model "
            "needed."
        )
        self._rebuild_btn.clicked.connect(self.rebuild)
        toolbar.addWidget(self._rebuild_btn)

        self._fit_btn = QPushButton("Fit")
        self._fit_btn.clicked.connect(lambda: self._view.fit())
        toolbar.addWidget(self._fit_btn)

        self._summary = QLabel("")
        self._summary.setObjectName("MetaText")
        toolbar.addWidget(self._summary, 1)
        root.addLayout(toolbar)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._view = MindMapView()
        self._view.concept_selected.connect(self._on_selected)
        splitter.addWidget(self._view)

        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.setContentsMargins(m.space_3, 0, 0, 0)
        detail_layout.setSpacing(m.space_2)

        self._title = QLabel("Select a concept")
        self._title.setObjectName("TitleText")
        self._title.setWordWrap(True)
        detail_layout.addWidget(self._title)

        self._meta = QLabel("")
        self._meta.setObjectName("MetaText")
        self._meta.setWordWrap(True)
        detail_layout.addWidget(self._meta)

        self._overview = QTextEdit()
        self._overview.setObjectName("CodeSurface")
        self._overview.setReadOnly(True)
        self._overview.setMaximumHeight(150)
        detail_layout.addWidget(self._overview)

        self._synth_btn = QPushButton("Write Overview")
        self._synth_btn.setToolTip(
            "Ask the local model to connect this concept's memories into a "
            "short overview. Takes 20-30s on CPU."
        )
        self._synth_btn.clicked.connect(self._synthesise)
        self._synth_btn.setEnabled(False)
        detail_layout.addWidget(self._synth_btn)

        memories_label = QLabel("MEMORIES")
        memories_label.setObjectName("SectionLabel")
        detail_layout.addWidget(memories_label)

        self._memories = QTextEdit()
        self._memories.setReadOnly(True)
        detail_layout.addWidget(self._memories, 1)

        splitter.addWidget(detail)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        splitter.setSizes([760, 420])
        root.addWidget(splitter, 1)

    # -- behaviour ----------------------------------------------------

    def _service(self):
        return self._sm.get("MindMap")

    def rebuild(self) -> None:
        service = self._service()
        if service is None:
            self._summary.setText("Mind Map service unavailable.")
            return
        mind_map = service.build()
        self._view.set_map(mind_map, service.layout())
        self._summary.setText(mind_map.summary())
        self._on_selected(self._view.selected)

    def _on_selected(self, key: str) -> None:
        service = self._service()
        if service is None or not key:
            self._title.setText("Select a concept")
            self._meta.setText("")
            self._overview.setPlainText("")
            self._memories.setPlainText("")
            self._synth_btn.setEnabled(False)
            return

        concept = service.map.concepts.get(key)
        if concept is None:
            return

        self._title.setText(concept.label)
        links = service.map.links_for(key)
        neighbours = []
        for link in sorted(links, key=lambda l: -l.shared)[:6]:
            other = link.target if link.source == key else link.source
            other_concept = service.map.concepts.get(other)
            if other_concept is not None:
                neighbours.append(f"{other_concept.label} ({link.shared})")
        self._meta.setText(
            f"{concept.category} · {concept.weight} memories · "
            f"{len(links)} connections\n"
            + ("Connected to: " + ", ".join(neighbours) if neighbours
               else "No connections yet.")
        )

        if concept.overview and not concept.overview_stale:
            self._overview.setPlainText(concept.overview)
        elif concept.overview:
            self._overview.setPlainText(
                "[OUT OF DATE - memories changed since this was written]\n\n"
                + concept.overview
            )
        else:
            self._overview.setPlainText(
                "No overview yet. Press 'Write Overview' to have the local "
                "model connect these memories into a short summary."
            )

        memories = service.memories_for(key)
        self._memories.setPlainText(
            "\n\n".join(f"- {m.content}" for m in memories)
            or "No memories."
        )
        self._synth_btn.setEnabled(bool(memories))

    def _synthesise(self) -> None:
        service = self._service()
        key = self._view.selected
        if service is None or not key:
            return

        self._synth_btn.setEnabled(False)
        self._synth_btn.setText("Writing... (20-30s)")
        self._overview.setPlainText("Generating overview, please wait...")
        QApplication.processEvents()

        try:
            result = service.synthesise(key)
        finally:
            self._synth_btn.setText("Write Overview")
            self._synth_btn.setEnabled(True)

        if result.ok:
            self._on_selected(key)
        else:
            self._overview.setPlainText(f"Could not write overview.\n\n{result.error}")

    def apply_theme(self) -> None:
        self._view.update()
