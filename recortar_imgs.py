import sys
import os
from dataclasses import dataclass
from pathlib import Path

import cv2 as cv

EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")


@dataclass(frozen=True)
class CropSpec:
    name: str
    x: int
    y: int
    w: int
    h: int


CROP_SPECS: list[CropSpec] = [
    CropSpec("PM5", 3958, 3015, 435, 749),
    CropSpec("PM4", 3584, 4018, 495, 508),
    CropSpec("PM3", 2569, 4513, 689, 338),
    CropSpec("PM2", 1759, 4066, 508, 544),
    CropSpec("PM1", 1276, 3124, 350, 640),
    CropSpec("CAN", 3475, 2036, 387, 375),
    CropSpec("ALLOY_SCREW4", 3185, 2266, 278, 242),
    CropSpec("ALLOY_SCREW3", 3971, 4018, 242, 363),
    CropSpec("ALLOY_SCREW2", 1941, 4501, 266, 242),
    CropSpec("ALLOY_SCREW1", 1361, 2797, 254, 266),
    CropSpec("ALLOY_EXC", 2931, 2411, 338, 387),
    CropSpec("ALLOY_CAN", 3088, 1988, 1148, 544),
    CropSpec("LASER_SOLDERING1", 2206, 1927, 242, 423),
    CropSpec("LASER_SOLDERING2", 1602, 2060, 278, 363),
]


def _safe_imread(path: Path):
    img = cv.imread(str(path))
    return img


def _clip_rect(x: int, y: int, w: int, h: int, img_w: int, img_h: int):
    x2 = min(img_w, x + w)
    y2 = min(img_h, y + h)
    x = max(0, x)
    y = max(0, y)
    w2 = max(0, x2 - x)
    h2 = max(0, y2 - y)
    return x, y, w2, h2


def list_images(in_dir: Path) -> list[Path]:
    return sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])


def crop_folder(in_dir: str | Path, out_dir: str | Path) -> int:
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)

    if not in_dir.exists() or not in_dir.is_dir():
        raise ValueError("Carpeta de entrada inválida.")
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list_images(in_dir)
    if not images:
        raise ValueError("La carpeta de entrada no contiene imágenes soportadas.")

    # Crear carpetas de recortes una vez
    for spec in CROP_SPECS:
        (out_dir / spec.name).mkdir(parents=True, exist_ok=True)

    written = 0
    for img_path in images:
        img = _safe_imread(img_path)
        if img is None:
            continue

        img_h, img_w = img.shape[:2]
        stem = img_path.stem

        for spec in CROP_SPECS:
            x, y, w, h = _clip_rect(spec.x, spec.y, spec.w, spec.h, img_w, img_h)
            if w <= 0 or h <= 0:
                continue
            crop = img[y : y + h, x : x + w]
            out_path = (out_dir / spec.name) / f"{stem}.png"
            if cv.imwrite(str(out_path), crop):
                written += 1

    return written


def main(*_args, **_kwargs):
    # Si se empaqueta como .exe (PyInstaller/cx_Freeze), evitamos cualquier salida a consola.
    if getattr(sys, "frozen", False):
        try:
            devnull = open(os.devnull, "w")  # noqa: SIM115
            sys.stdout = devnull
            sys.stderr = devnull
        except Exception:
            pass

    # Compatibilidad con launchers antiguos: si pasan muchos args,
    # se usan los dos primeros como in_dir/out_dir.
    in_dir = _kwargs.get("in_dir") if isinstance(_kwargs, dict) else None
    out_dir = _kwargs.get("out_dir") if isinstance(_kwargs, dict) else None

    if in_dir is None and len(_args) >= 1:
        in_dir = _args[0]
    if out_dir is None and len(_args) >= 2:
        out_dir = _args[1]

    if not in_dir or not out_dir:
        # Evitar traceback al ejecutar sin argumentos.
        if not getattr(sys, "frozen", False):
            print("Uso:", file=sys.stderr)
            print('  python recortar_imgs.py "/ruta/entrada" "/ruta/salida"', file=sys.stderr)
        return 2

    crop_folder(in_dir, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
