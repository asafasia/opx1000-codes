from __future__ import annotations

import base64
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "last_figs_preview" / "root_lorentzian_simple_small.jpg"
OUTPUT_IMAGE = Path(__file__).resolve().parent / "root_lorentzian_simple_mobile.jpg"
OUTPUT_HTML = Path(__file__).resolve().parent / "root_lorentzian_simple_mobile.html"


def main() -> None:
    image = Image.open(SOURCE).convert("RGB")
    image.thumbnail((1100, 850))
    image.save(OUTPUT_IMAGE, quality=82, optimize=True)

    encoded = base64.b64encode(OUTPUT_IMAGE.read_bytes()).decode("ascii")
    document = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Root Lorentzian Echo True</title>
  <style>
    body {{ margin: 0; background: #111; color: #eee; font-family: Arial, sans-serif; }}
    main {{ padding: 10px; }}
    img {{ width: 100%; height: auto; background: white; display: block; }}
    a {{ color: #9ddcff; font-size: 18px; }}
  </style>
</head>
<body>
  <main>
    <h3>Root-Lorentzian length sequence, echo=True</h3>
    <img alt="Root Lorentzian figure" src="data:image/jpeg;base64,{encoded}">
    <p><a download="root_lorentzian_simple_mobile.jpg" href="data:image/jpeg;base64,{encoded}">Download JPG</a></p>
  </main>
</body>
</html>
"""
    OUTPUT_HTML.write_text(document, encoding="utf-8")
    print(OUTPUT_HTML)
    print(OUTPUT_IMAGE)
    print(f"html_bytes={OUTPUT_HTML.stat().st_size}")
    print(f"jpg_bytes={OUTPUT_IMAGE.stat().st_size}")


if __name__ == "__main__":
    main()
