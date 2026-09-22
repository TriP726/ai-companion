"""Tests for the 3D graph view.

Everything here runs on the offscreen platform: the math core
(ai_companion.graph3d) is Qt-free by design, and Graph3DView exposes all
non-GL behavior — layout, picking, selection, camera moves — as plain
methods. Only initializeGL/paintGL need a real context, and those are
deliberately thin so the drawing either works or falls back to 2D.
"""
from __future__ import annotations

import math
import os

import numpy as np
import pytest

from ai_companion.graph3d.camera import (
    OrbitCamera,
    pick_nearest,
    ray_sphere_hit,
)
from ai_companion.graph3d.layout3d import TARGET_SPREAD, force_layout_3d


@pytest.fixture
def tmp_env(tmp_path):
    """Isolated config with every store path under tmp_path (same contract
    as the fixture in test_services.py, kept local so this module is
    self-contained)."""
    from ai_companion.models.config_models import AppConfig

    config = AppConfig().relocate(tmp_path)
    os.makedirs(config.data_dir, exist_ok=True)
    return config, tmp_path


@pytest.fixture
def signal_bus():
    from ai_companion.infrastructure.signal_bus import SignalBus

    return SignalBus()


# ----------------------------------------------------------------------
# OrbitCamera
# ----------------------------------------------------------------------


class TestOrbitCamera:
    def test_eye_orbits_target_at_distance(self):
        cam = OrbitCamera(target=(1.0, 2.0, 3.0), distance=500.0)
        eye = cam.eye()
        assert float(np.linalg.norm(eye - np.array([1.0, 2.0, 3.0]))) == pytest.approx(
            500.0, rel=1e-9
        )

    def test_orbit_changes_yaw_not_distance(self):
        cam = OrbitCamera(distance=800.0)
        before = cam.eye()
        cam.orbit(90.0, 0.0)
        after = cam.eye()
        assert not np.allclose(before, after)
        assert cam.distance == pytest.approx(800.0)

    def test_pitch_is_clamped_away_from_the_poles(self):
        cam = OrbitCamera()
        cam.orbit(0.0, 10_000.0)
        assert cam.pitch < math.radians(90.0)
        cam.orbit(0.0, -20_000.0)
        assert cam.pitch > math.radians(-90.0)

    def test_zoom_is_clamped(self):
        cam = OrbitCamera(distance=800.0)
        cam.zoom(1e-9)
        assert cam.distance == OrbitCamera.MIN_DISTANCE
        cam.zoom(1e12)
        assert cam.distance == OrbitCamera.MAX_DISTANCE

    def test_zoom_rejects_nonpositive_factors(self):
        cam = OrbitCamera(distance=800.0)
        cam.zoom(0.0)
        cam.zoom(-2.0)
        assert cam.distance == pytest.approx(800.0)

    def test_target_projects_to_screen_center(self):
        cam = OrbitCamera(target=(10.0, -5.0, 3.0), distance=1000.0)
        xy = cam.project((10.0, -5.0, 3.0), 800, 600)
        assert xy is not None
        assert xy[0] == pytest.approx(400.0, abs=1e-6)
        assert xy[1] == pytest.approx(300.0, abs=1e-6)

    def test_behind_camera_projects_to_none(self):
        cam = OrbitCamera(target=(0.0, 0.0, 0.0), distance=500.0)
        eye = cam.eye()
        fwd = cam.forward()
        behind = eye - fwd * 100.0  # 100 units behind the eye
        assert cam.project(tuple(behind), 800, 600) is None

    def test_unproject_ray_through_center_passes_through_target(self):
        cam = OrbitCamera(target=(4.0, 1.0, -2.0), distance=1200.0)
        origin, direction = cam.unproject_ray(400.0, 300.0, 800, 600)
        # Distance from the ray to the target should be ~0.
        to_target = np.array([4.0, 1.0, -2.0]) - origin
        perp = to_target - (to_target @ direction) * direction
        assert float(np.linalg.norm(perp)) == pytest.approx(0.0, abs=1e-6)
        # And the ray originates at the eye.
        assert np.allclose(origin, cam.eye())

    def test_project_unproject_roundtrip(self):
        cam = OrbitCamera(target=(0.0, 0.0, 0.0), distance=900.0)
        point = cam.target + np.array([50.0, 80.0, -30.0])
        xy = cam.project(tuple(point), 1024, 768)
        assert xy is not None
        origin, direction = cam.unproject_ray(xy[0], xy[1], 1024, 768)
        # The ray must pass through the original point.
        to_point = point - origin
        perp = to_point - (to_point @ direction) * direction
        assert float(np.linalg.norm(perp)) == pytest.approx(0.0, abs=1e-5)

    def test_pan_moves_target_in_camera_plane(self):
        cam = OrbitCamera(distance=1000.0)
        before = cam.target.copy()
        right = cam.right()
        cam.pan(100.0, 0.0, 600)  # drag right -> target moves left (content follows cursor)
        delta = cam.target - before
        # Motion must be perpendicular to the view direction (in the plane).
        assert abs(float(delta @ cam.forward())) < 1e-6
        assert float(delta @ right) < 0.0
        assert float(np.linalg.norm(delta)) > 0.0


# ----------------------------------------------------------------------
# Ray picking
# ----------------------------------------------------------------------


class TestRayPicking:
    def test_head_on_hit(self):
        t = ray_sphere_hit(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 100.0]),
            10.0,
        )
        assert t == pytest.approx(90.0)

    def test_miss_returns_none(self):
        t = ray_sphere_hit(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([50.0, 0.0, 100.0]),
            10.0,
        )
        assert t is None

    def test_behind_the_eye_is_not_a_hit(self):
        t = ray_sphere_hit(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, -100.0]),
            10.0,
        )
        assert t is None

    def test_eye_inside_sphere_still_hits(self):
        t = ray_sphere_hit(
            np.array([0.0, 0.0, 100.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 100.0]),
            10.0,
        )
        assert t is not None

    def test_pick_nearest_wins(self):
        positions = {
            "far": (0.0, 0.0, 200.0),
            "near": (0.0, 0.0, 60.0),
        }
        hit = pick_nearest(
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            positions,
            10.0,
        )
        assert hit == "near"

    def test_grazing_hit(self):
        # Ray passes exactly at the sphere's radius: still a hit.
        t = ray_sphere_hit(
            np.array([0.0, 10.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            np.array([0.0, 0.0, 100.0]),
            10.0,
        )
        assert t == pytest.approx(100.0)


# ----------------------------------------------------------------------
# 3D force layout
# ----------------------------------------------------------------------


class TestForceLayout3D:
    def test_empty_and_single(self):
        assert force_layout_3d([], []) == {}
        single = force_layout_3d(["only"], [])
        assert single == {"only": (0.0, 0.0, 0.0)}

    def test_deterministic_across_calls(self):
        ids = [f"node-{i}" for i in range(12)]
        edges = [(ids[i], ids[i + 1]) for i in range(11)]
        a = force_layout_3d(ids, edges)
        b = force_layout_3d(ids, edges)
        assert set(a) == set(b)
        for nid in a:
            assert a[nid] == pytest.approx(b[nid], abs=1e-12)

    def test_edge_endpoints_closer_than_independent_pairs(self):
        ids = [f"n{i}" for i in range(8)]
        # Two tight clusters far apart: {0-1-2} and {3-4-5}; 6,7 isolated.
        edges = [
            ("n0", "n1"), ("n1", "n2"), ("n0", "n2"),
            ("n3", "n4"), ("n4", "n5"), ("n3", "n5"),
        ]
        pos = force_layout_3d(ids, edges)

        def dist(a, b):
            pa, pb = np.array(pos[a]), np.array(pos[b])
            return float(np.linalg.norm(pa - pb))

        intra = [dist(*e) for e in edges]
        assert max(intra) < dist("n0", "n5")
        assert max(intra) < dist("n2", "n3")

    def test_two_nodes_separate(self):
        pos = force_layout_3d(["a", "b"], [])
        pa, pb = np.array(pos["a"]), np.array(pos["b"])
        assert float(np.linalg.norm(pa - pb)) > 1.0

    def test_positions_are_finite_and_centered(self):
        ids = [f"x{i}" for i in range(20)]
        edges = [(ids[i], ids[(i + 3) % 20]) for i in range(20)]
        pos = force_layout_3d(ids, edges)
        pts = np.array(list(pos.values()))
        assert np.isfinite(pts).all()
        assert np.allclose(pts.mean(axis=0), 0.0, atol=1e-6)

    def test_rms_spread_is_normalized(self):
        ids = [f"y{i}" for i in range(15)]
        pos = force_layout_3d(ids, [(ids[0], ids[1])])
        pts = np.array(list(pos.values()))
        rms = float(np.sqrt(np.mean(np.sum(pts**2, axis=1))))
        assert rms == pytest.approx(TARGET_SPREAD, rel=1e-6)

    def test_seeded_positions_stay_near_the_seed(self):
        ids = [f"z{i}" for i in range(10)]
        edges = [(ids[i], ids[i + 1]) for i in range(9)]
        first = force_layout_3d(ids, edges)
        # Add one node, seed from previous result: the old nodes should
        # barely move compared to a cold re-layout.
        moved = force_layout_3d(
            ids + ["z-new"], edges + [("z9", "z-new")], seed_positions=first
        )
        cold = force_layout_3d(ids + ["z-new"], edges + [("z9", "z-new")])
        warm_drift = sum(
            float(np.linalg.norm(np.array(moved[i]) - np.array(first[i])))
            for i in ids
        )
        cold_drift = sum(
            float(np.linalg.norm(np.array(cold[i]) - np.array(first[i])))
            for i in ids
        )
        assert warm_drift < cold_drift


# ----------------------------------------------------------------------
# Graph3DView — everything except actual GL drawing
# ----------------------------------------------------------------------


@pytest.fixture
def app():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    return QApplication.instance() or QApplication([])


def _make_view():
    from ai_companion.ui.graph3d_view import Graph3DView

    view = Graph3DView()
    nodes = [
        {"id": "alpha", "label": "Alpha node", "color": "#ff0000"},
        {"id": "beta", "label": "Beta node", "color": "#00ff00"},
        {"id": "taggy", "label": "tag: work", "color": "#888888", "is_tag": True},
    ]
    edges = [("alpha", "beta"), ("alpha", "taggy")]
    view.set_graph(nodes, edges)
    return view


class TestGraph3DViewData:
    def test_scalar_uniforms_are_set_by_integer_location(self, app):
        from ai_companion.ui.graph3d_view import _set_float_uniform

        calls = []

        class FakeProgram:
            def uniformLocation(self, name):
                calls.append(("lookup", name))
                return 7

        class FakeGl:
            def glUniform1f(self, location, value):
                calls.append(("set", location, value))

        _set_float_uniform(FakeGl(), FakeProgram(), "time", 1.25)
        assert calls == [("lookup", "time"), ("set", 7, 1.25)]

    def test_vec3_uniforms_are_set_by_integer_location(self, app):
        from ai_companion.ui.graph3d_view import _set_vec3_uniform

        calls = []

        class FakeProgram:
            def uniformLocation(self, name):
                calls.append(("lookup", name))
                return 9

        class FakeGl:
            def glUniform3f(self, location, r, g, b):
                calls.append(("set3", location, r, g, b))

        _set_vec3_uniform(FakeGl(), FakeProgram(), "baseColor", 0.1, 0.2, 0.3)
        assert calls == [("lookup", "baseColor"), ("set3", 9, 0.1, 0.2, 0.3)]

    def test_draw_background_sets_uniforms_and_draws(self, app):
        from ai_companion.ui.graph3d_view import _GL_TRIANGLES

        view = _make_view()
        calls = []

        class FakeProgram:
            def bind(self):
                calls.append("bind_prog")

            def release(self):
                calls.append("release_prog")

            def uniformLocation(self, name):
                calls.append(("lookup", name))
                return 5

        class FakeVAO:
            def bind(self):
                calls.append("bind_vao")

            def release(self):
                calls.append("release_vao")

        class FakeGl:
            def glUniform1f(self, loc, val):
                calls.append(("1f", loc, val))

            def glUniform3f(self, loc, r, g, b):
                calls.append(("3f", loc, r, g, b))

            def glDrawArrays(self, mode, first, count):
                calls.append(("draw", mode, first, count))

        view._gl = FakeGl()
        view._prog_bg = FakeProgram()
        view._vao_bg = FakeVAO()
        view._draw_background(1.5)

        assert "bind_prog" in calls
        assert ("lookup", "aspect") in calls
        assert ("1f", 5, 1.5) in calls
        assert ("lookup", "baseColor") in calls
        assert ("lookup", "glowColor") in calls
        assert ("draw", _GL_TRIANGLES, 0, 3) in calls
        assert "release_prog" in calls

    def test_animation_clock_advances_and_requests_repaint(self, app):
        view = _make_view()
        before = view._animation_time
        view._advance_animation()
        assert view._animation_time > before
        assert view._animation_timer.isActive()

    def test_energy_glyph_shader_uses_declared_uv_varying(self, app):
        from ai_companion.ui.graph3d_view import _NODE_FRAG

        assert "length(vUV)" in _NODE_FRAG
        assert "hexDistance" in _NODE_FRAG
        assert "vec3(uv, z)" not in _NODE_FRAG

    def test_connections_are_particle_billboards_not_driver_lines(self, app):
        from ai_companion.ui.graph3d_view import _EDGE_FRAG, _EDGE_VERT

        assert "particleT" in _EDGE_VERT
        assert "quadratic Bezier" in _EDGE_VERT
        assert "hotCore" in _EDGE_FRAG

    def test_paint_uses_module_constants_not_qt_function_attributes(self, app):
        """PySide6 QOpenGLFunctions has calls but no GL_* enum attributes."""
        view = _make_view()
        calls = []

        class FakeFunctions:
            def glClearColor(self, *args):
                calls.append(("clear_color", args))

            def glClear(self, mask):
                calls.append(("clear", mask))

            def glEnable(self, capability):
                calls.append(("enable", capability))

            def glBlendFunc(self, source, destination):
                calls.append(("blend", source, destination))

        view._gl = FakeFunctions()
        view._gl_ready = True
        view._gl_failed = False
        view._buffers_dirty = False
        view._edge_vertex_count = 0
        view._node_vertex_count = 0
        view._paint_labels = lambda **_kwargs: None

        view.paintGL()

        assert ("clear", 0x4000 | 0x0100) in calls
        assert ("enable", 0x0B71) in calls
        assert ("enable", 0x0BE2) in calls
        # Nodes use triangle billboards, not driver-sensitive GL point size.
        assert all(call != ("enable", 0x8642) for call in calls)

    def test_layout_computed_for_all_nodes(self, app):
        view = _make_view()
        pos = view.world_positions()
        assert set(pos) == {"alpha", "beta", "taggy"}

    def test_same_graph_does_not_relayout(self, app):
        view = _make_view()
        before = view.world_positions()
        view.set_graph(
            [
                {"id": "alpha", "label": "Alpha node", "color": "#ff0000"},
                {"id": "beta", "label": "Beta node", "color": "#00ff00"},
                {"id": "taggy", "label": "tag: work", "color": "#888888", "is_tag": True},
            ],
            [("alpha", "beta"), ("alpha", "taggy")],
        )
        assert view.world_positions() == before

    def test_edges_referencing_missing_nodes_are_dropped(self, app):
        view = _make_view()
        view.set_graph(
            [{"id": "solo", "label": "Solo"}],
            [("solo", "ghost")],
        )
        assert view.world_positions()["solo"] == (0.0, 0.0, 0.0)

    def test_pick_hits_the_right_node(self, app):
        view = _make_view()
        pos = view.world_positions()
        eye = np.array([0.0, 0.0, 2000.0])
        target = np.array(pos["beta"])
        direction = target - eye
        direction = direction / np.linalg.norm(direction)
        assert view.pick_ray(eye, direction) == "beta"
        # Alpha's direction must not hit beta.
        other = np.array(pos["alpha"])
        d2 = other - eye
        d2 = d2 / np.linalg.norm(d2)
        assert view.pick_ray(eye, d2) == "alpha"

    def test_pick_miss_returns_none(self, app):
        view = _make_view()
        eye = np.array([0.0, 0.0, 2000.0])
        direction = np.array([0.0, 1.0, 0.0])
        assert view.pick_ray(eye, direction) is None

    def test_selection_lifecycle(self, app):
        view = _make_view()
        fired = []
        view.node_selected.connect(fired.append)
        view.select("alpha")
        assert view.selected_node() == "alpha"
        assert fired == ["alpha"]
        view.select(None)
        assert view.selected_node() is None

    def test_select_unknown_id_is_ignored(self, app):
        view = _make_view()
        view.select("nope")
        assert view.selected_node() is None

    def test_focus_node_moves_camera_and_selects(self, app):
        view = _make_view()
        view.focus_node("beta")
        assert view.selected_node() == "beta"
        target = view.camera.target
        assert np.allclose(target, np.array(view.world_positions()["beta"]))

    def test_focus_unknown_id_is_a_noop(self, app):
        view = _make_view()
        before = view.camera.target.copy()
        view.focus_node("nope")
        assert np.allclose(view.camera.target, before)

    def test_node_at_uses_camera_consistently(self, app):
        """node_at() must agree with project(): a node's screen position
        picks that node back."""
        view = _make_view()
        view.reset_camera()
        view.resize(800, 600)
        for nid, p in view.world_positions().items():
            xy = view.camera.project(p, view.width(), view.height())
            if xy is None:
                continue
            assert view.node_at(xy[0], xy[1]) == nid

    def test_reset_camera_frames_everything(self, app):
        view = _make_view()
        view.resize(800, 600)
        view.reset_camera()
        projected = [
            view.camera.project(p, view.width(), view.height())
            for p in view.world_positions().values()
        ]
        assert all(p is not None for p in projected)
        margin = 40
        for x, y, _ in projected:
            assert -margin <= x <= view.width() + margin
            assert -margin <= y <= view.height() + margin

    def test_clear_graph(self, app):
        view = _make_view()
        view.select("alpha")
        view.clear_graph()
        assert view.world_positions() == {}
        assert view.selected_node() is None


class TestGraphPanel3DIntegration:
    """The panel owns the 2D/3D toggle and the fallback logic."""

    def _panel(self, tmp_env, signal_bus):
        from ai_companion.config import ConfigManager
        from ai_companion.infrastructure.service_manager import ServiceManager
        from ai_companion.services.graph_service import GraphService
        from ai_companion.ui.graph_panel import GraphPanel

        config, _ = tmp_env
        cm = ConfigManager()
        cm._config = config
        sm = ServiceManager(cm)
        sm._signal_bus = signal_bus
        graph = GraphService(config, signal_bus)
        sm.register(graph)
        graph.start()
        panel = GraphPanel(sm)
        return panel, graph

    def test_toggle_switches_the_stack(self, tmp_env, signal_bus, app):
        panel, graph = self._panel(tmp_env, signal_bus)
        assert panel._toggle3d_btn.isEnabled()
        assert panel._view_stack.currentIndex() == 0
        panel._on_toggle_3d(True)
        assert panel._view_stack.currentIndex() == 1
        panel._on_toggle_3d(False)
        assert panel._view_stack.currentIndex() == 0

    def test_refresh_feeds_the_3d_view(self, tmp_env, signal_bus, app):
        panel, graph = self._panel(tmp_env, signal_bus)
        from ai_companion.models.graph import NodeType

        graph.add_node(NodeType.IDEA, "Test idea 3d")
        panel._refresh_graph()
        assert panel._graph3d_view is not None
        assert len(panel._graph3d_view.world_positions()) == 1

    def test_gl_failure_falls_back_to_2d(self, tmp_env, signal_bus, app):
        panel, _ = self._panel(tmp_env, signal_bus)
        panel._on_toggle_3d(True)
        assert panel._view_stack.currentIndex() == 1
        panel._on_3d_unavailable("mock driver failure")
        assert panel._view_stack.currentIndex() == 0
        assert not panel._toggle3d_btn.isChecked()
        assert not panel._toggle3d_btn.isEnabled()
        assert "OpenGL" in panel._stats_label.text()

    def test_clear_empties_both_views(self, tmp_env, signal_bus, app):
        panel, graph = self._panel(tmp_env, signal_bus)
        from ai_companion.models.graph import NodeType

        graph.add_node(NodeType.IDEA, "Doomed idea")
        panel._refresh_graph()
        graph.clear()
        panel._on_graph_cleared()
        assert panel._graph3d_view.world_positions() == {}
