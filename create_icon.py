"""Generate a multi-resolution application icon."""
from __future__ import annotations

import struct
import zlib
from pathlib import Path


def create_png(width: int, height: int) -> bytes:
    """Create a minimal PNG icon with a brain/circuit design."""
    pixels = []
    cx, cy = width // 2, height // 2
    radius = min(width, height) // 2 - 2

    for y in range(height):
        row = []
        for x in range(width):
            dx = x - cx
            dy = y - cy
            dist = (dx * dx + dy * dy) ** 0.5

            if dist > radius:
                row.extend([0, 0, 0, 0])  # transparent
            elif dist > radius - 2:
                row.extend([74, 158, 255, 255])  # border
            else:
                # Background gradient
                t = dist / radius
                r = int(13 + t * 10)
                g = int(17 + t * 15)
                b = int(30 + t * 20)

                # Neural network pattern
                # Central node
                if dist < radius * 0.15:
                    r, g, b = 74, 158, 255

                # Ring pattern
                ring_dist = abs(dist - radius * 0.4)
                if ring_dist < 2:
                    r, g, b = 74, 158, 255

                ring_dist2 = abs(dist - radius * 0.7)
                if ring_dist2 < 1.5:
                    r, g, b = 50, 120, 200

                # Connection lines (radial)
                import math
                angle = math.atan2(dy, dx)
                for a in [0, 60, 120, 180, 240, 300]:
                    a_rad = math.radians(a)
                    angle_diff = abs(angle - a_rad)
                    if angle_diff > math.pi:
                        angle_diff = 2 * math.pi - angle_diff
                    if angle_diff < 0.08 and dist > radius * 0.15:
                        r, g, b = 74, 158, 255

                # Dots at intersections
                for a in [0, 60, 120, 180, 240, 300]:
                    a_rad = math.radians(a)
                    nx = cx + radius * 0.4 * math.cos(a_rad)
                    ny = cy + radius * 0.4 * math.sin(a_rad)
                    if ((x - nx) ** 2 + (y - ny) ** 2) < (radius * 0.05) ** 2:
                        r, g, b = 0, 200, 83

                    nx2 = cx + radius * 0.7 * math.cos(a_rad)
                    ny2 = cy + radius * 0.7 * math.sin(a_rad)
                    if ((x - nx2) ** 2 + (y - ny2) ** 2) < (radius * 0.04) ** 2:
                        r, g, b = 124, 77, 255

                row.extend([r, g, b, 255])
        pixels.append(bytes(row))

    return _encode_png(width, height, pixels)


def _encode_png(width: int, height: int, rows: list[bytes]) -> bytes:
    """Encode raw pixel data as PNG."""
    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    header = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)

    raw = b""
    for row in rows:
        raw += b"\x00" + row  # filter byte + pixel data

    compressed = zlib.compress(raw, 9)

    return (
        header
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def create_ico(png_data_256: bytes, png_data_48: bytes, png_data_32: bytes, png_data_16: bytes) -> bytes:
    """Create a .ico file from multiple PNG images."""
    images = [png_data_256, png_data_48, png_data_32, png_data_16]
    sizes = [256, 48, 32, 16]

    # ICO header
    header = struct.pack("<HHH", 0, 1, len(images))

    # Calculate offsets
    offset = 6 + 16 * len(images)  # header + directory entries
    entries = b""
    all_data = b""

    for i, (png, size) in enumerate(zip(images, sizes)):
        w = 0 if size >= 256 else size  # 0 means 256
        h = 0 if size >= 256 else size
        entry = struct.pack("<BBBBHHII", w, h, 0, 0, 1, 32, len(png), offset)
        entries += entry
        all_data += png
        offset += len(png)

    return header + entries + all_data


def main() -> None:
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)

    # Generate PNGs at different sizes
    png_256 = create_png(256, 256)
    png_48 = create_png(48, 48)
    png_32 = create_png(32, 32)
    png_16 = create_png(16, 16)

    # Save individual PNGs
    (output_dir / "icon_256.png").write_bytes(png_256)
    (output_dir / "icon_48.png").write_bytes(png_48)
    (output_dir / "icon_32.png").write_bytes(png_32)
    (output_dir / "icon_16.png").write_bytes(png_16)

    # Create ICO
    ico_data = create_ico(png_256, png_48, png_32, png_16)
    (output_dir / "app.ico").write_bytes(ico_data)

    print(f"Icons generated in {output_dir}/")
    print(f"  app.ico: {len(ico_data)} bytes")


if __name__ == "__main__":
    main()
