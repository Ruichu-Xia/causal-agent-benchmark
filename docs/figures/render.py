"""Render the README figures from figures.html to light and dark PNGs with headless Chrome.

    uv run --with pillow python docs/figures/render.py

Each figure is screenshotted at 2x on a tall page, then trimmed to its content.
"""

import subprocess
import sys
from pathlib import Path

from PIL import Image

HERE = Path(__file__).resolve().parent
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
FIGURES = (
    "hero",
    "workflow",
    "steps",
    "episode",
    "actions",
    "example",
    "generation",
    "families",
    "transfer",
    "study",
    "results",
)
WIDTH = 960
SCALE = 2


def render(figure: str, theme: str) -> Path:
    """Screenshot one figure in one theme and trim the empty space below it."""
    out = HERE / f"{figure}-{theme}.png"
    url = f"{(HERE / 'figures.html').as_uri()}?fig={figure}&theme={theme}"
    subprocess.run(
        [
            CHROME,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--force-device-scale-factor={SCALE}",
            f"--window-size={WIDTH},1400",
            "--virtual-time-budget=5000",
            f"--screenshot={out}",
            url,
        ],
        check=True,
        capture_output=True,
    )
    image = Image.open(out).convert("RGB")
    background = image.getpixel((image.width - 2, image.height - 2))
    pixels = image.load()
    bottom = image.height
    while bottom > 1 and all(pixels[x, bottom - 1] == background for x in range(0, image.width, 4)):
        bottom -= 1
    image.crop((0, 0, image.width, min(image.height, bottom + 26 * SCALE))).save(out, optimize=True)
    return out


def main() -> None:
    """Render every figure, or only those named on the command line."""
    for figure in sys.argv[1:] or FIGURES:
        for theme in ("light", "dark"):
            print(render(figure, theme))


if __name__ == "__main__":
    main()
