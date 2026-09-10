"""Bind the two unchanged factory raster sheets into one local PDF."""
from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parent / "originals"
pages = [Image.open(root / name).convert("RGB") for name in ("jcm800pr.gif", "jcm800pw.gif")]
pages[0].save(root / "marshall_2203_factory_1981.pdf", save_all=True, append_images=pages[1:], resolution=150.)
print("Created reference/originals/marshall_2203_factory_1981.pdf")
