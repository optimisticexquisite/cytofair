"""Combine rendered PDF pages into 2x2 QA contact sheets."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "tmp" / "monograph_render"
OUTPUT = ROOT / "tmp" / "monograph_contacts"


def main() -> None:
    pages = sorted(SOURCE.glob("page-*.png"))
    if not pages:
        raise RuntimeError(f"No rendered pages found under {SOURCE}")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for old in OUTPUT.glob("sheet-*.png"):
        old.unlink()

    margin, label_height = 24, 34
    font = ImageFont.load_default(size=20)
    for offset in range(0, len(pages), 4):
        batch = pages[offset : offset + 4]
        opened = [Image.open(path).convert("RGB") for path in batch]
        width = max(image.width for image in opened)
        height = max(image.height for image in opened)
        sheet = Image.new(
            "RGB",
            (2 * width + 3 * margin, 2 * (height + label_height) + 3 * margin),
            "#DCE5EA",
        )
        draw = ImageDraw.Draw(sheet)
        for index, (path, page) in enumerate(zip(batch, opened, strict=True)):
            row, column = divmod(index, 2)
            x = margin + column * (width + margin)
            y = margin + row * (height + label_height + margin)
            draw.text((x, y), path.stem.replace("page-", "PDF page "), fill="#24323D", font=font)
            sheet.paste(page, (x, y + label_height))
        first = offset + 1
        last = offset + len(batch)
        sheet.save(OUTPUT / f"sheet-{first:02d}-{last:02d}.png", optimize=True)

    print(f"rendered {len(pages)} pages into {(len(pages) + 3) // 4} contact sheets")


if __name__ == "__main__":
    main()
