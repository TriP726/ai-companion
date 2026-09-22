"""True-3D knowledge-graph view on a hand-rolled QOpenGLWidget.

Why not Qt3D or an embedded browser: for a scene of spheres and lines,
Qt3D's framegraph ceremony buys nothing and QtWebEngine would bundle a
whole Chromium for dots. Two tiny shader programs (nodes as point-sprite
spheres, edges as lines) do the entire job with depth buffering, and all
the math that decides what appears where lives in ai_companion.graph3d —
Qt-free and unit-tested.

GL failure policy: if context/shader setup raises (ancient driver,
remote desktop, ANGLE oddity), the widget emits gl_unavailable with the
reason and stops touching GL. GraphPanel listens and falls back to the
2D view. Nothing crashes, nothing renders garbage.

Everything that can be exercised without a GL context — set_graph,
layout, picking, selection, camera moves — is a plain method usable in
tests on the offscreen platform.
"""
from __future__ import annotations

from typing import Iterable, Optional

from PySide6.QtCore import QPoint, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QMatrix4x4, QMouseEvent, QPainter, QWheelEvent
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtWidgets import QWidget

from ai_companion.graph3d.camera import OrbitCamera, ray_sphere_hit
from ai_companion.graph3d.layout3d import TARGET_SPREAD, force_layout_3d

# QOpenGLFunctions wraps OpenGL calls but PySide6 does not expose GL_* enum
# constants on that object (unlike PyOpenGL). Keep the small set this renderer
# needs here using the values fixed by the OpenGL specification.
_GL_TRIANGLES = 0x0004
_GL_SRC_ALPHA = 0x0302
_GL_ONE_MINUS_SRC_ALPHA = 0x0303
_GL_DEPTH_BUFFER_BIT = 0x0100
_GL_BLEND = 0x0BE2
_GL_DEPTH_TEST = 0x0B71
_GL_FLOAT = 0x1406
_GL_COLOR_BUFFER_BIT = 0x4000

# --------------------------------------------------------------------------
# Shaders (GLSL 330 core — every GPU/driver from the last 15 years)
# --------------------------------------------------------------------------

_EDGE_VERT = """#version 330 core
layout(location = 0) in vec3 endpointA;
layout(location = 1) in vec3 controlPoint;
layout(location = 2) in vec3 endpointB;
layout(location = 3) in vec3 color;
layout(location = 4) in float particleT;
layout(location = 5) in float phase;
layout(location = 6) in vec2 corner;
layout(location = 7) in float particleSize;
uniform mat4 view;
uniform mat4 proj;
uniform float time;
uniform float intensity;
out vec3 vColor;
out vec2 vUV;
out float vSpark;
void main() {
    // Every six vertices form one camera-facing particle. Its centre flows
    // along a quadratic Bezier curve, so connections remain obvious in 3D.
    // intensity scales flow speed and pulse depth; clamped so a future
    // "turn it up more" request can't flip particleSize negative and
    // invert the billboard.
    float speed = clamp(0.34 * (0.7 + 0.3 * intensity), 0.1, 1.2);
    float t = fract(particleT + time * speed + phase);
    float oneMinus = 1.0 - t;
    vec3 centre = oneMinus * oneMinus * endpointA
                + 2.0 * oneMinus * t * controlPoint
                + t * t * endpointB;
    float cometPulse = clamp(
        0.78 + 0.22 * intensity * sin(t * 31.0 - time * 5.5 + phase * 6.28318),
        0.35, 1.6
    );
    vec4 eyePos = view * vec4(centre, 1.0);
    eyePos.xy += corner * particleSize * cometPulse;
    gl_Position = proj * eyePos;
    vColor = color;
    vUV = corner;
    vSpark = cometPulse;
}
"""

_EDGE_FRAG = """#version 330 core
in vec3 vColor;
in vec2 vUV;
in float vSpark;
out vec4 frag;
void main() {
    float radius = length(vUV);
    if (radius > 1.65) discard;

    // Three distinct layers for real visual depth:
    // 1. Brilliant white-hot kernel (intense point-source core)
    float hotCore = 1.0 - smoothstep(0.0, 0.22, radius);
    // 2. Saturated colored mid-glow body
    float midGlow = 1.0 - smoothstep(0.12, 0.65, radius);
    // 3. Soft translucent outer aura
    float aura = 1.0 - smoothstep(0.35, 1.60, radius);

    float alpha = clamp(hotCore * 0.96 + midGlow * 0.80 + aura * aura * (0.32 + vSpark * 0.22), 0.0, 1.0);
    vec3 midCol = vColor * (1.15 + 0.35 * vSpark);
    vec3 energy = mix(midCol, vec3(1.0), hotCore * 0.94);
    frag = vec4(energy, alpha);
}
"""

_BG_VERT = """#version 330 core
out vec2 vScreen;
void main() {
    // Fullscreen triangle generated from gl_VertexID - no vertex buffer.
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vScreen = p * 2.0 - 1.0;
    gl_Position = vec4(vScreen, 0.9999, 1.0);
}
"""

_BG_FRAG = """#version 330 core
in vec2 vScreen;
uniform vec3 baseColor;
uniform vec3 glowColor;
uniform float aspect;
out vec4 frag;
void main() {
    vec2 p = vScreen;
    p.x *= aspect;
    vec2 centre = vec2(0.0, 0.12);
    centre.x *= aspect;
    float d = length(p - centre);
    // One broad soft light behind the graph; the corners fall back to the
    // dark base. No bands, no stars - a quiet illuminated backdrop.
    float glow = exp(-d * d * 1.9);
    float vignette = 1.0 - smoothstep(1.25, 2.10, length(p));
    vec3 col = baseColor + glowColor * (glow * 0.60);
    col *= 0.72 + 0.28 * vignette;
    frag = vec4(col, 1.0);
}
"""

_NODE_VERT = """#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in float worldRadius;
layout(location = 3) in float stateFlag;  // 0 normal, 1 hovered, 2 selected
layout(location = 4) in vec2 corner;      // billboard corner, -1.55..1.55
layout(location = 5) in float phase;
uniform mat4 view;
uniform mat4 proj;
uniform float time;
uniform float intensity;
out vec3 vColor;
out float vState;
out vec2 vUV;
out float vGlow;
out float vRotation;
void main() {
    // intensity scales pulse depth and spin speed; clamped so it can be
    // turned up later without ever inverting the billboard (breathe <= 0).
    float breathe = clamp(
        1.0 + 0.075 * intensity * sin(time * 2.35 + phase * 6.28318),
        0.5, 1.8
    );
    vec4 eyePos = view * vec4(position, 1.0);
    eyePos.xy += corner * worldRadius * breathe;
    gl_Position = proj * eyePos;
    vColor = color;
    vState = stateFlag;
    vUV = corner;
    vGlow = clamp(0.72 + 0.28 * intensity * sin(time * 2.8 + phase * 6.28318), 0.2, 1.6);
    vRotation = time * (0.78 + 0.35 * (intensity - 1.0)) + phase * 6.28318;
}
"""

_NODE_FRAG = """#version 330 core
in vec3 vColor;
in float vState;
in vec2 vUV;
in float vGlow;
in float vRotation;
out vec4 frag;
void main() {
    float radius = length(vUV);
    if (radius > 1.55) discard;

    // Rotate an angular energy glyph inside its billboard.
    float cs = cos(vRotation);
    float sn = sin(vRotation);
    vec2 glyphUV = mat2(cs, -sn, sn, cs) * vUV;
    vec2 absoluteUV = abs(glyphUV);
    float hexDistance = max(absoluteUV.y, absoluteUV.x * 0.866025 + absoluteUV.y * 0.5);

    // 1. Hot white kernel at the center (reads as a lit energy source)
    float kernel = exp(-radius * radius * 28.0);
    float innerGlow = exp(-radius * radius * 5.0) * 0.60;

    // 2. Six faceted spoke lines at 60 deg like cut lines on a gem
    // Hexagon vertices are at pi/6 + k*pi/3 (30, 90, 150, 210, 270, 330 deg)
    float theta = atan(glyphUV.y, glyphUV.x);
    float spokeAngle = abs(sin((theta - 0.52359877) * 3.0));
    float spokeDist = radius * spokeAngle;
    float spokeMask = (1.0 - smoothstep(0.68, 0.78, hexDistance)) * smoothstep(0.08, 0.24, radius);
    float spokes = exp(-spokeDist * 32.0) * spokeMask;
    float facetLight = 0.85 + 0.15 * cos((theta - 0.52359877) * 6.0);

    // 3. Hexagon body and lit rim
    float coreHex = 1.0 - smoothstep(0.66, 0.78, hexDistance);
    float rim = smoothstep(0.70, 0.76, hexDistance) - smoothstep(0.80, 0.86, hexDistance);
    float innerRing = exp(-abs(radius - 0.44) * 28.0) * 0.28;
    float halo = (1.0 - smoothstep(0.74, 1.50, radius)) * 0.35 * vGlow;

    // Selection/hover boost
    float stateBoost = (vState > 0.5) ? 1.25 : 1.0;

    float alpha = clamp(max(coreHex + rim * 0.6 + halo, kernel), 0.0, 1.0) * stateBoost;
    alpha = clamp(alpha, 0.0, 1.0);
    if (alpha < 0.012) discard;

    // Facet lighting on the saturated base
    vec3 baseCol = vColor * (0.65 + 0.35 * facetLight) * (0.75 + coreHex * 0.25);
    vec3 coloredBody = baseCol + vColor * (spokes * 0.75 + innerRing * 0.50 + innerGlow * 0.40);
    vec3 rimCol = mix(vColor, vec3(1.0), 0.88);
    vec3 nodeCol = mix(coloredBody, rimCol, rim);

    // Blend hot white kernel at center and spoke crease highlights
    nodeCol = mix(nodeCol, vec3(1.0), kernel);
    nodeCol += vec3(1.0, 0.95, 1.0) * (spokes * 0.50);

    frag = vec4(nodeCol, alpha);
}
"""


def _qcolor_rgbf(color: str = "") -> tuple[float, float, float]:
    """Hex string -> RGB floats. Colours fall back to the theme accent:
    a hardcoded hex would stay right in dark mode and wrong in light."""
    from ai_companion.ui.theme import theme

    c = QColor(color) if color else QColor(theme.p.accent)
    if not c.isValid():
        c = QColor(theme.p.accent)
    return c.redF(), c.greenF(), c.blueF()


class Graph3DView(QOpenGLWidget):
    """Drag-to-orbit 3D graph. Mirrors GraphView's panel-facing API."""

    node_selected = Signal(str)
    gl_unavailable = Signal(str)

    NODE_RADIUS = 22.0
    TAG_RADIUS_SCALE = 0.72
    PICK_RADIUS_SCALE = 1.15  # forgiveness so nodes are easy to hit
    MAX_LABELS = 40
    MAX_LABEL_CHARS = 26
    # Single knob for motion/pulse strength: >1 = more energetic, 1 = the
    # original tuning, <1 = calmer. One number to change instead of
    # re-tuning every sin() amplitude across both shaders.
    ANIMATION_INTENSITY = 1.35

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        fmt = QSurfaceFormat()
        fmt.setVersion(3, 3)
        fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
        fmt.setDepthBufferSize(24)
        fmt.setSamples(4)
        self.setFormat(fmt)

        self._nodes: dict[str, dict] = {}
        self._edges: list[tuple[str, str]] = []
        self._positions: dict[str, tuple[float, float, float]] = {}
        self._radii: dict[str, float] = {}
        self._signature: tuple = (frozenset(), frozenset())

        self._camera = OrbitCamera(distance=TARGET_SPREAD * 3.2)
        self._selected_id: Optional[str] = None
        self._hover_id: Optional[str] = None

        self._gl_ready = False
        self._gl_failed = False
        self._buffers_dirty = True
        self._prog_nodes: Optional[QOpenGLShaderProgram] = None
        self._prog_edges: Optional[QOpenGLShaderProgram] = None
        self._prog_bg: Optional[QOpenGLShaderProgram] = None
        self._vao_nodes: Optional[QOpenGLVertexArrayObject] = None
        self._vao_edges: Optional[QOpenGLVertexArrayObject] = None
        self._vao_bg: Optional[QOpenGLVertexArrayObject] = None
        self._vbo_nodes: Optional[QOpenGLBuffer] = None
        self._vbo_edges: Optional[QOpenGLBuffer] = None
        self._edge_vertex_count = 0
        self._node_vertex_count = 0
        self._gl = None

        self._press: Optional[tuple[Qt.MouseButton, QPoint]] = None
        self._animation_time = 0.0
        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(33)  # smooth without wasting CPU
        self._animation_timer.timeout.connect(self._advance_animation)
        self._animation_timer.start()

        self.setMouseTracking(True)

    def _advance_animation(self) -> None:
        """Advance a deterministic shader clock and request the next frame."""
        self._animation_time = (self._animation_time + 0.033) % 10000.0
        self.update()

    # ------------------------------------------------------------------
    # Data (no GL needed)
    # ------------------------------------------------------------------

    def set_graph(
        self,
        nodes: Iterable[dict],
        edges: Iterable[tuple[str, str]],
    ) -> None:
        """Replace the graph. `nodes` items: {id, label, color, is_tag}.
        Layout is only recomputed when the node/edge set actually changes,
        seeded from the previous positions so the view doesn't jump."""
        from ai_companion.ui.theme import theme

        self._nodes = {
            n["id"]: {
                "label": n.get("label", ""),
                "color": n.get("color") or theme.p.accent,
                "is_tag": bool(n.get("is_tag", False)),
            }
            for n in nodes
        }
        self._edges = [(a, b) for a, b in edges if a in self._nodes and b in self._nodes]

        signature = (frozenset(self._nodes), frozenset(frozenset(e) for e in self._edges))
        if signature != self._signature:
            self._signature = signature
            self._positions = force_layout_3d(
                list(self._nodes), self._edges, seed_positions=self._positions
            )
            self._radii = {
                nid: self.NODE_RADIUS
                * (self.TAG_RADIUS_SCALE if self._nodes[nid]["is_tag"] else 1.0)
                for nid in self._nodes
            }
            self._buffers_dirty = True
        self.update()

    @property
    def camera(self) -> OrbitCamera:
        return self._camera

    def world_positions(self) -> dict[str, tuple[float, float, float]]:
        return dict(self._positions)

    def clear_graph(self) -> None:
        self.set_graph([], [])
        self._selected_id = None
        self._hover_id = None

    # ------------------------------------------------------------------
    # Selection / focus (no GL needed)
    # ------------------------------------------------------------------

    def selected_node(self) -> Optional[str]:
        return self._selected_id

    def select(self, node_id: Optional[str], *, emit: bool = True) -> None:
        if node_id is not None and node_id not in self._nodes:
            return
        if node_id != self._selected_id:
            self._selected_id = node_id
            if emit and node_id is not None:
                self.node_selected.emit(node_id)
        self._buffers_dirty = True
        self.update()

    def focus_node(self, node_id: str) -> None:
        """Point the camera at a node and select it. The 3D analogue of
        GraphView.highlight_node (which centers the 2D view on the item)."""
        pos = self._positions.get(node_id)
        if pos is None:
            return
        import numpy as np

        self._camera.target = np.asarray(pos, dtype=float)
        self.select(node_id)

    def highlight_node(self, node_id: str) -> None:  # GraphView-compatible
        self.focus_node(node_id)

    # ------------------------------------------------------------------
    # Picking (no GL needed)
    # ------------------------------------------------------------------

    def pick_ray(self, ray_origin, ray_dir) -> Optional[str]:
        """Nearest node hit by a world-space ray, honoring per-node radii."""
        best_id: Optional[str] = None
        best_t = float("inf")
        for nid, center in self._positions.items():
            t = ray_sphere_hit(
                ray_origin,
                ray_dir,
                center,
                self._radii.get(nid, self.NODE_RADIUS) * self.PICK_RADIUS_SCALE,
            )
            if t is not None and t < best_t:
                best_t, best_id = t, nid
        return best_id

    def node_at(self, x_px: float, y_px: float) -> Optional[str]:
        """Pick the node under widget pixel (x, y)."""
        ray = self._camera.unproject_ray(x_px, y_px, self.width(), self.height())
        return self.pick_ray(*ray)

    # ------------------------------------------------------------------
    # Camera moves (no GL needed)
    # ------------------------------------------------------------------

    def reset_camera(self) -> None:
        """Frame the whole graph: target at the bounding-sphere centre,
        distance so the sphere fits the frustum."""
        if not self._positions:
            self._camera.target *= 0.0
            self._camera.distance = TARGET_SPREAD * 3.2
            self.update()
            return
        import numpy as np

        pts = np.array(list(self._positions.values()))
        center = pts.mean(axis=0)
        radius = float(np.sqrt(((pts - center) ** 2).sum(axis=1)).max())
        self._camera.target = center
        import math

        self._camera.distance = max(
            (radius + self.NODE_RADIUS * 2) / math.tan(math.radians(self._camera.fov) / 2) * 1.15,
            OrbitCamera.MIN_DISTANCE,
        )
        self.update()

    # ------------------------------------------------------------------
    # Mouse interaction (no GL needed to construct; events need a widget)
    # ------------------------------------------------------------------

    def mousePressEvent(self, event: QMouseEvent) -> None:
        self._press = (event.button(), event.position().toPoint())

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        if self._press is not None:
            button, last = self._press
            dx, dy = pos.x() - last.x(), pos.y() - last.y()
            if button == Qt.MouseButton.LeftButton:
                self._camera.orbit(dx * 0.35, -dy * 0.35)
            elif button in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
                self._camera.pan(dx, dy, self.height())
            self._press = (button, pos)
            self.update()
            return
        # Hover feedback: cursor + tooltip + slight brighten in shader.
        hover = self.node_at(pos.x(), pos.y())
        if hover != self._hover_id:
            self._hover_id = hover
            if hover is None:
                self.unsetCursor()
                self.setToolTip("")
            else:
                self.setCursor(Qt.CursorShape.PointingHandCursor)
                self.setToolTip(self._nodes[hover]["label"])
            self._buffers_dirty = True
            self.update()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        if self._press is not None:
            button, press_pos = self._press
            self._press = None
            pos = event.position().toPoint()
            moved = (pos - press_pos).manhattanLength()
            if button == Qt.MouseButton.LeftButton and moved < 6:
                self.select(self.node_at(pos.x(), pos.y()))

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:
        pos = event.position().toPoint()
        hit = self.node_at(pos.x(), pos.y())
        if hit is None:
            self.reset_camera()
        else:
            self.focus_node(hit)

    def wheelEvent(self, event: QWheelEvent) -> None:
        steps = event.angleDelta().y() / 120.0
        if steps:
            self._camera.zoom(0.88 ** steps)
            self.update()

    # ------------------------------------------------------------------
    # GL lifecycle (only safe to touch with a current context)
    # ------------------------------------------------------------------

    def initializeGL(self) -> None:  # noqa: N802 (Qt override)
        try:
            self._gl = self.context().functions()
            self._prog_bg = self._build_program(_BG_VERT, _BG_FRAG)
            self._prog_edges = self._build_program(_EDGE_VERT, _EDGE_FRAG)
            self._prog_nodes = self._build_program(_NODE_VERT, _NODE_FRAG)
            self._vao_bg = QOpenGLVertexArrayObject(self)
            self._vao_bg.create()
            self._vao_nodes = QOpenGLVertexArrayObject(self)
            self._vao_nodes.create()
            self._vao_edges = QOpenGLVertexArrayObject(self)
            self._vao_edges.create()
            self._vbo_nodes = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._vbo_nodes.create()
            self._vbo_edges = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            self._vbo_edges.create()
            self._gl_ready = True
        except Exception as exc:  # noqa: BLE001 - any GL/driver failure
            self._gl_failed = True
            # Emit after initializeGL returns; emitting mid-init on some
            # drivers is what turns an error into a crash.
            QTimer.singleShot(
                0, lambda m=str(exc): self.gl_unavailable.emit(m)
            )

    def _build_program(self, vert: str, frag: str) -> QOpenGLShaderProgram:
        prog = QOpenGLShaderProgram(self)
        if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Vertex, vert):
            raise RuntimeError(f"vertex shader: {prog.log()}")
        if not prog.addShaderFromSourceCode(QOpenGLShader.ShaderTypeBit.Fragment, frag):
            raise RuntimeError(f"fragment shader: {prog.log()}")
        if not prog.link():
            raise RuntimeError(f"link: {prog.log()}")
        return prog

    def resizeGL(self, w: int, h: int) -> None:  # noqa: N802
        if self._gl is not None:
            self._gl.glViewport(0, 0, w, h)

    def paintGL(self) -> None:  # noqa: N802
        if not self._gl_ready or self._gl_failed:
            return
        import numpy as np

        from ai_companion.ui.theme import theme

        gl = self._gl
        # The backdrop is drawn by the _BG_* pass (dark base + one soft glow),
        # so the clear colour only needs to match the pass's dark base.
        base = QColor(theme.p.bg_surface)
        hue, sat, _val, _alpha = base.getHsvF()
        bg = QColor.fromHsvF(hue, max(0.0, min(sat * 0.50, 1.0)), 0.12)
        gl.glClearColor(bg.redF(), bg.greenF(), bg.blueF(), 1.0)
        gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
        gl.glEnable(_GL_DEPTH_TEST)
        gl.glEnable(_GL_BLEND)
        gl.glBlendFunc(_GL_SRC_ALPHA, _GL_ONE_MINUS_SRC_ALPHA)

        if self._buffers_dirty:
            self._upload_buffers()

        w, h = max(self.width(), 1), max(self.height(), 1)
        aspect = w / h
        proj = self._camera.projection_matrix(aspect)
        view = self._camera.view_matrix()
        elapsed = float(self._animation_time)

        # Illuminated backdrop first. The quad sits at NDC z = 0.9999 so it
        # passes the depth test against the cleared 1.0 without needing the
        # depth test toggled off; depthMask(False) keeps it from writing.
        if self._prog_bg is not None:
            gl.glDepthMask(False)
            self._draw_background(aspect)
            gl.glDepthMask(True)

        if self._edge_vertex_count:
            # Particles are pure glow, and many of them can overlap on
            # screen. Writing depth for each one means whichever particle
            # happens to be drawn first "wins" and the rest pop/hard-cut
            # against it instead of blending. Test depth (so particles still
            # sit behind a node genuinely in front of them) but don't write
            # it, so overlapping particles blend smoothly with each other.
            gl.glDepthMask(False)
            self._prog_edges.bind()
            self._prog_edges.setUniformValue("view", _to_qmat(view))
            self._prog_edges.setUniformValue("proj", _to_qmat(proj))
            _set_float_uniform(gl, self._prog_edges, "time", elapsed)
            _set_float_uniform(gl, self._prog_edges, "intensity", self.ANIMATION_INTENSITY)
            self._vao_edges.bind()
            gl.glDrawArrays(_GL_TRIANGLES, 0, self._edge_vertex_count)
            self._vao_edges.release()
            self._prog_edges.release()
            gl.glDepthMask(True)

        if self._node_vertex_count:
            self._prog_nodes.bind()
            self._prog_nodes.setUniformValue("view", _to_qmat(view))
            self._prog_nodes.setUniformValue("proj", _to_qmat(proj))
            _set_float_uniform(gl, self._prog_nodes, "time", elapsed)
            _set_float_uniform(gl, self._prog_nodes, "intensity", self.ANIMATION_INTENSITY)
            self._vao_nodes.bind()
            gl.glDrawArrays(_GL_TRIANGLES, 0, self._node_vertex_count)
            self._vao_nodes.release()
            self._prog_nodes.release()

        self._paint_labels(proj=proj, view=view, width=w, height=h)

    def _draw_background(self, aspect: float) -> None:
        """Screen-space illuminated backdrop behind the whole scene."""
        if self._prog_bg is None or self._gl is None:
            return
        from ai_companion.ui.theme import theme

        base = QColor(theme.p.bg_surface)
        hue, sat, _v, _a = base.getHsvF()
        dark = QColor.fromHsvF(hue, max(0.0, min(sat * 0.50, 1.0)), 0.12)
        glow = QColor.fromHsvF(hue, max(0.0, min(sat * 0.80, 1.0)), 0.42)
        prog = self._prog_bg
        prog.bind()
        _set_float_uniform(self._gl, prog, "aspect", float(aspect))
        _set_vec3_uniform(self._gl, prog, "baseColor", dark.redF(), dark.greenF(), dark.blueF())
        _set_vec3_uniform(self._gl, prog, "glowColor", glow.redF(), glow.greenF(), glow.blueF())
        self._vao_bg.bind()
        self._gl.glDrawArrays(_GL_TRIANGLES, 0, 3)
        self._vao_bg.release()
        prog.release()

    def _paint_labels(self, *, proj, view, width: int, height: int) -> None:
        """Text can't come from the shaders — draw labels with QPainter
        on top of the GL pass (the supported QOpenGLWidget overlay)."""
        from ai_companion.ui.theme import theme

        labeled: list[tuple[float, float, str, bool, float]] = []
        for nid in self._label_candidates():
            xy = self._camera.project(self._positions[nid], width, height)
            if xy is None:
                continue
            label = self._nodes[nid]["label"]
            if len(label) > self.MAX_LABEL_CHARS:
                label = label[: self.MAX_LABEL_CHARS - 1].rstrip() + "\u2026"
            labeled.append(
                (xy[0], xy[1], label, nid in (self._selected_id, self._hover_id), xy[2])
            )

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        font = QFont("Arial", 8)
        painter.setFont(font)
        # Selection/hover wins first; below MAX_LABELS, _label_candidates()
        # returns every node in arbitrary dict order rather than
        # distance-sorted, so without a second key here a far node's label
        # could beat a closer, more relevant one to a crowded spot purely
        # by insertion order. depth (xy[2], already computed by project())
        # breaks that tie the way it visually should: nearest wins.
        labeled.sort(key=lambda item: (not item[3], item[4]))
        occupied = []
        for x, y, label, emphasized, _depth in labeled:
            rect = painter.fontMetrics().boundingRect(label)
            rect.moveCenter(QPoint(int(x), int(y) - 28))
            rect.adjust(-4, -2, 4, 3)
            collision_box = rect.adjusted(-6, -4, 6, 4)
            if not emphasized and any(
                collision_box.intersects(other) for other in occupied
            ):
                continue
            occupied.append(collision_box)
            # Plain dim text, no chip: on the illuminated dark backdrop a
            # solid rounded chip reads as a heavy dark box (the old look).
            pen = QColor(theme.p.fg_primary if emphasized else theme.p.fg_muted)
            pen.setAlpha(235 if emphasized else 165)
            painter.setPen(pen)
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, label)

        if not self._nodes:
            painter.setPen(QColor(theme.p.fg_muted))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No graph data"
            )
        painter.end()

    def _label_candidates(self) -> list[str]:
        """Nearest N nodes, plus hovered/selected regardless of depth —
        otherwise the one label you actually care about gets culled."""
        ids = list(self._positions)
        if len(ids) <= self.MAX_LABELS:
            return ids
        import numpy as np

        eye = self._camera.eye()
        by_depth = sorted(
            ids, key=lambda nid: float(np.sum((np.asarray(self._positions[nid]) - eye) ** 2))
        )
        chosen = by_depth[: self.MAX_LABELS]
        for nid in (self._selected_id, self._hover_id):
            if nid is not None and nid not in chosen and nid in self._positions:
                chosen.append(nid)
        return chosen

    def _upload_buffers(self) -> None:
        """(Re)build node/edge VBOs. Runs inside paintGL (context current)."""
        import hashlib

        import numpy as np

        self._buffers_dirty = False

        # Particle streams: each particle is a six-vertex billboard whose
        # centre is animated along a deterministic quadratic Bezier path.
        if self._edges and self._positions:
            from ai_companion.ui.theme import theme

            ec = _qcolor_rgbf(theme.p.accent)
            rows = []
            particle_count = 32
            extent = 1.65
            particle_corners = (
                (-extent, -extent), (extent, -extent), (extent, extent),
                (-extent, -extent), (extent, extent), (-extent, extent),
            )
            for a, b in self._edges:
                pa_raw, pb_raw = self._positions.get(a), self._positions.get(b)
                if pa_raw is None or pb_raw is None:
                    continue
                pa = np.asarray(pa_raw, dtype=float)
                pb = np.asarray(pb_raw, dtype=float)
                delta = pb - pa
                length = float(np.linalg.norm(delta))
                digest = hashlib.sha256(f"{a}\0{b}".encode("utf-8")).digest()
                edge_phase = int.from_bytes(digest[:2], "big") / 65535.0
                seed_axis = np.array(
                    [digest[2] / 127.5 - 1.0, digest[3] / 127.5 - 1.0, 0.65]
                )
                perpendicular = np.cross(delta, seed_axis)
                norm = float(np.linalg.norm(perpendicular))
                if norm < 1e-6:
                    perpendicular = np.array([0.0, 1.0, 0.0])
                else:
                    perpendicular /= norm
                binormal = np.cross(delta, perpendicular)
                b_norm = float(np.linalg.norm(binormal))
                if b_norm < 1e-6:
                    binormal = np.array([1.0, 0.0, 0.0])
                else:
                    binormal /= b_norm

                bend = min(max(length * 0.20, 35.0), 125.0)
                base_control = (pa + pb) * 0.5 + perpendicular * bend
                base_size = min(max(length * 0.018, 4.8), 8.5)

                for index in range(particle_count):
                    p_digest = hashlib.sha256(
                        f"{a}\0{b}\0{index}".encode("utf-8")
                    ).digest()
                    # 1. Desynchronize spacing: natural clustering surges instead of rigid equidistant steps
                    t_jitter = (p_digest[0] / 255.0 - 0.5) * (0.80 / particle_count)
                    particle_t = (index / particle_count + t_jitter) % 1.0

                    # 2. Individual phase offset: desyncs flicker & pulse timing per particle
                    phase = (
                        edge_phase
                        + int.from_bytes(p_digest[1:3], "big") / 65535.0
                    ) % 1.0

                    # 3. Size jitter: small sparks to prominent energy droplets
                    size_mult = 0.60 + 0.75 * (p_digest[3] / 255.0)
                    size = base_size * size_mult

                    # 4. Subtle lateral conduit dispersion: particles wander slightly around
                    # the curve spine rather than marching single-file on a 1D hairline
                    disp_perp = (p_digest[4] / 127.5 - 1.0) * (base_size * 1.6)
                    disp_binorm = (p_digest[5] / 127.5 - 1.0) * (base_size * 1.6)
                    control = (
                        base_control
                        + perpendicular * disp_perp
                        + binormal * disp_binorm
                    )

                    base = [*pa, *control, *pb, *ec, particle_t, phase]
                    rows.extend(
                        [*base, *corner, size] for corner in particle_corners
                    )
            arr = np.asarray(rows, dtype=np.float32)
        else:
            arr = np.zeros(0, dtype=np.float32)

        self._edge_vertex_count = arr.size // 17
        self._vao_edges.bind()
        self._vbo_edges.bind()
        self._vbo_edges.allocate(arr.tobytes(), int(arr.nbytes))
        if self._edge_vertex_count:
            prog = self._prog_edges
            prog.bind()
            prog.enableAttributeArray(0)
            prog.setAttributeBuffer(0, _GL_FLOAT, 0, 3, 68)
            prog.enableAttributeArray(1)
            prog.setAttributeBuffer(1, _GL_FLOAT, 12, 3, 68)
            prog.enableAttributeArray(2)
            prog.setAttributeBuffer(2, _GL_FLOAT, 24, 3, 68)
            prog.enableAttributeArray(3)
            prog.setAttributeBuffer(3, _GL_FLOAT, 36, 3, 68)
            prog.enableAttributeArray(4)
            prog.setAttributeBuffer(4, _GL_FLOAT, 48, 1, 68)
            prog.enableAttributeArray(5)
            prog.setAttributeBuffer(5, _GL_FLOAT, 52, 1, 68)
            prog.enableAttributeArray(6)
            prog.setAttributeBuffer(6, _GL_FLOAT, 56, 2, 68)
            prog.enableAttributeArray(7)
            prog.setAttributeBuffer(7, _GL_FLOAT, 64, 1, 68)
            prog.release()
        self._vbo_edges.release()
        self._vao_edges.release()

        # Nodes are camera-facing quads (two triangles each), not GL_POINTS.
        # Record: position, color, radius, state, corner, animation phase.
        # Corners extend beyond the solid sphere to make room for its aura.
        rows = []
        extent = 1.55
        corners = (
            (-extent, -extent), (extent, -extent), (extent, extent),
            (-extent, -extent), (extent, extent), (-extent, extent),
        )
        for nid, pos in self._positions.items():
            rgb = _qcolor_rgbf(self._nodes[nid]["color"])
            state = (
                2.0
                if nid == self._selected_id
                else 1.0
                if nid == self._hover_id
                else 0.0
            )
            digest = hashlib.sha256(nid.encode("utf-8")).digest()
            phase = int.from_bytes(digest[:2], "big") / 65535.0
            base = [*pos, *rgb, self._radii.get(nid, self.NODE_RADIUS), state]
            rows.extend([*base, *corner, phase] for corner in corners)
        narr = (
            np.asarray(rows, dtype=np.float32)
            if rows
            else np.zeros(0, dtype=np.float32)
        )
        self._node_vertex_count = narr.size // 11
        self._vao_nodes.bind()
        self._vbo_nodes.bind()
        self._vbo_nodes.allocate(narr.tobytes(), int(narr.nbytes))
        if self._node_vertex_count:
            prog = self._prog_nodes
            prog.bind()
            prog.enableAttributeArray(0)
            prog.setAttributeBuffer(0, _GL_FLOAT, 0, 3, 44)
            prog.enableAttributeArray(1)
            prog.setAttributeBuffer(1, _GL_FLOAT, 12, 3, 44)
            prog.enableAttributeArray(2)
            prog.setAttributeBuffer(2, _GL_FLOAT, 24, 1, 44)
            prog.enableAttributeArray(3)
            prog.setAttributeBuffer(3, _GL_FLOAT, 28, 1, 44)
            prog.enableAttributeArray(4)
            prog.setAttributeBuffer(4, _GL_FLOAT, 32, 2, 44)
            prog.enableAttributeArray(5)
            prog.setAttributeBuffer(5, _GL_FLOAT, 40, 1, 44)
            prog.release()
        self._vbo_nodes.release()
        self._vao_nodes.release()


def _set_float_uniform(
    gl, program: QOpenGLShaderProgram, name: str, value: float
) -> None:
    """Set a scalar via the raw GL call, not Qt's overloaded setUniformValue().

    This used to resolve the location and then call
    ``program.setUniformValue(location, float(value))`` - a workaround for
    PySide6 not exposing a (str, float) overload for scalars. That
    resolved the location correctly, but the *value* still wasn't
    reliably reaching the shader: setUniformValue is itself an overloaded
    Qt wrapper (int/float/QVector*/QMatrix* all share one call), and
    which overload Shiboken picks for a bare Python float has been
    inconsistent enough across PySide6 builds to be worth not depending
    on at all. glUniform1f(GLint, GLfloat) is a flat, single-signature C
    call - there is no overload for Shiboken to pick wrong. Every other
    GL call in this file already goes through this same `gl` functions
    object (glClearColor, glDrawArrays, glDepthMask, ...), so this is
    consistent with the rest of the file rather than a new pattern.
    """
    # uniformLocation() changed binding expectations across PySide6 releases:
    # current Windows builds prefer str, while some older builds accept bytes.
    try:
        location = program.uniformLocation(name)
    except TypeError:
        location = -1
    if location < 0:
        try:
            location = program.uniformLocation(name.encode("ascii"))
        except TypeError:
            location = -1
    if location >= 0:
        if gl is not None and hasattr(gl, "glUniform1f"):
            gl.glUniform1f(location, float(value))
        else:
            try:
                program.setUniformValue(location, float(value))
            except Exception:
                pass


def _set_vec3_uniform(
    gl, program: QOpenGLShaderProgram, name: str, r: float, g: float, b: float
) -> None:
    """Set a vec3 uniform via glUniform3f or setUniformValue by integer location."""
    try:
        location = program.uniformLocation(name)
    except TypeError:
        location = -1
    if location < 0:
        try:
            location = program.uniformLocation(name.encode("ascii"))
        except TypeError:
            location = -1
    if location >= 0:
        if gl is not None and hasattr(gl, "glUniform3f"):
            gl.glUniform3f(location, float(r), float(g), float(b))
        else:
            try:
                from PySide6.QtGui import QVector3D

                program.setUniformValue(location, QVector3D(float(r), float(g), float(b)))
            except Exception:
                try:
                    program.setUniformValue(location, float(r), float(g), float(b))
                except Exception:
                    pass


def _to_qmat(m) -> QMatrix4x4:
    """numpy 4x4 (row-major) -> QMatrix4x4 (its ctor takes row-major)."""
    return QMatrix4x4(*[float(v) for v in m.flatten()])
