"""Qt-free math for the 3D knowledge-graph view.

Everything in here is pure Python + numpy — no Qt imports — so the camera,
projection, picking and 3D layout logic can be unit-tested headlessly and
reused behind a different renderer later. The OpenGL widget lives in
ai_companion/ui/graph3d_view.py and is only ever a thin skin over this.
"""
