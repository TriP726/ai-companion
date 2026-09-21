"""Orbit camera, perspective projection and ray picking — no Qt anywhere.

Conventions: right-handed world, +Y up, column vectors, row-major matrices
applied as ``M @ v``. Angles in radians at the math level, degrees at the
API edges that humans/tests touch.
"""
from __future__ import annotations

import math
from typing import Optional

import numpy as np

Vec3 = np.ndarray  # shape (3,)

_UP: Vec3 = np.array([0.0, 1.0, 0.0])


def _normalize(v: Vec3) -> Vec3:
    n = float(np.linalg.norm(v))
    if n == 0.0:
        raise ValueError("cannot normalize a zero vector")
    return v / n


class OrbitCamera:
    """A camera that orbits a target point: yaw, pitch, distance.

    Left-drag rotates (yaw/pitch), the wheel dollies (distance), and
    right-drag pans (moves the target in the camera plane). Pitch is
    clamped just short of the poles so `up` never degenerates.
    """

    MIN_PITCH = math.radians(-89.0)
    MAX_PITCH = math.radians(89.0)
    MIN_DISTANCE = 40.0
    MAX_DISTANCE = 20000.0

    def __init__(
        self,
        target: Vec3 | tuple[float, float, float] = (0.0, 0.0, 0.0),
        distance: float = 1200.0,
        yaw: float = math.radians(30.0),
        pitch: float = math.radians(-20.0),
        fov_degrees: float = 45.0,
    ) -> None:
        self.target: Vec3 = np.asarray(tuple(target), dtype=float)
        self.distance = float(distance)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.fov = float(fov_degrees)
        self._clamp()

    # -- interaction -----------------------------------------------------

    def _clamp(self) -> None:
        self.pitch = min(self.MAX_PITCH, max(self.MIN_PITCH, self.pitch))
        self.distance = min(
            self.MAX_DISTANCE, max(self.MIN_DISTANCE, self.distance)
        )

    def orbit(self, d_yaw_degrees: float, d_pitch_degrees: float) -> None:
        self.yaw += math.radians(d_yaw_degrees)
        self.pitch += math.radians(d_pitch_degrees)
        self._clamp()

    def zoom(self, factor: float) -> None:
        """Multiply distance by `factor` (<1 moves closer)."""
        if factor <= 0:
            return
        self.distance *= factor
        self._clamp()

    def pan(self, dx_pixels: float, dy_pixels: float, viewport_height: int) -> None:
        """Move the target in the camera plane; pixels map to world units
        at the target's depth so panning feels 1:1 at any zoom."""
        world_per_pixel = self._world_per_pixel(self.distance, viewport_height)
        right = self.right()
        up = self.up()
        self.target = (
            self.target - right * dx_pixels * world_per_pixel
            + up * dy_pixels * world_per_pixel
        )

    # -- geometry ---------------------------------------------------------

    def eye(self) -> Vec3:
        """Camera position in world space."""
        cp = math.cos(self.pitch)
        offset = np.array(
            [
                math.sin(self.yaw) * cp,
                -math.sin(self.pitch),
                math.cos(self.yaw) * cp,
            ]
        )
        return self.target + offset * self.distance

    def forward(self) -> Vec3:
        return _normalize(self.target - self.eye())

    def right(self) -> Vec3:
        r = np.cross(self.forward(), _UP)
        n = float(np.linalg.norm(r))
        if n < 1e-9:  # looking straight down/up (pitch clamp prevents this)
            return np.array([1.0, 0.0, 0.0])
        return r / n

    def up(self) -> Vec3:
        return np.cross(self.right(), self.forward())

    def view_matrix(self) -> np.ndarray:
        eye = self.eye()
        f = self.forward()
        r = self.right()
        u = np.cross(r, f)
        m = np.identity(4)
        m[0, :3], m[1, :3], m[2, :3] = r, u, -f
        m[0, 3], m[1, 3], m[2, 3] = -r @ eye, -u @ eye, f @ eye
        return m

    def projection_matrix(self, aspect: float) -> np.ndarray:
        aspect = max(aspect, 1e-6)
        near, far = 1.0, 100000.0
        t = 1.0 / math.tan(math.radians(self.fov) / 2.0)
        m = np.zeros((4, 4))
        m[0, 0] = t / aspect
        m[1, 1] = t
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    def _world_per_pixel(self, depth: float, viewport_height: int) -> float:
        return (
            2.0
            * depth
            * math.tan(math.radians(self.fov) / 2.0)
            / max(int(viewport_height), 1)
        )

    # -- projection / unprojection ----------------------------------------

    def project(
        self, point: Vec3 | tuple[float, float, float], width: int, height: int
    ) -> Optional[tuple[float, float, float]]:
        """World -> screen. Returns (x_px, y_px, depth01) or None when the
        point is behind the camera."""
        p = np.asarray(tuple(point), dtype=float)
        vp = self.projection_matrix(width / max(height, 1)) @ self.view_matrix()
        clip = vp @ np.append(p, 1.0)
        w = float(clip[3])
        if w <= 1e-9:
            return None
        ndc = clip[:3] / w
        x = (float(ndc[0]) + 1.0) * 0.5 * width
        y = (1.0 - float(ndc[1])) * 0.5 * height
        return x, y, float(ndc[2])

    def unproject_ray(
        self, x_px: float, y_px: float, width: int, height: int
    ) -> tuple[Vec3, Vec3]:
        """Pixel -> world-space picking ray as (origin, unit direction).
        Origin is the eye; the ray passes exactly through the pixel."""
        ndc_x = (2.0 * x_px) / max(width, 1) - 1.0
        ndc_y = 1.0 - (2.0 * y_px) / max(height, 1)
        vp = self.projection_matrix(width / max(height, 1)) @ self.view_matrix()
        inv = np.linalg.inv(vp)
        near = inv @ np.array([ndc_x, ndc_y, -1.0, 1.0])
        far = inv @ np.array([ndc_x, ndc_y, 1.0, 1.0])
        near_p, far_p = near[:3] / near[3], far[:3] / far[3]
        eye = self.eye()
        return eye, _normalize(far_p - near_p)


def ray_sphere_hit(
    ray_origin: Vec3, ray_dir: Vec3, center: Vec3, radius: float
) -> Optional[float]:
    """Distance t along the ray to the sphere's surface, or None on a miss.
    Returns the near intersection; a hit behind the eye is not a hit."""
    oc = center - ray_origin
    t_ca = float(oc @ ray_dir)
    if t_ca < 0.0:
        return None
    d2 = float(oc @ oc) - t_ca * t_ca
    r2 = radius * radius
    if d2 > r2:
        return None
    t = t_ca - math.sqrt(r2 - d2)
    if t <= 0.0:  # eye is inside the sphere; still counts as a hit
        return t_ca
    return t


def pick_nearest(
    ray_origin: Vec3,
    ray_dir: Vec3,
    positions: dict[str, Vec3 | tuple[float, float, float]],
    radius: float,
) -> Optional[str]:
    """Id of the nearest sphere the ray hits, or None."""
    best_id: Optional[str] = None
    best_t = math.inf
    for node_id, center in positions.items():
        t = ray_sphere_hit(
            ray_origin, ray_dir, np.asarray(tuple(center), dtype=float), radius
        )
        if t is not None and t < best_t:
            best_t, best_id = t, node_id
    return best_id
