# refbbox_batch_crop.py (versión con escala de vista)
import cv2 as cv
from pathlib import Path
from natsort import natsorted
import argparse

EXTS = (".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp")

def list_images(folder):
    return natsorted([p for p in Path(folder).iterdir() if p.suffix.lower() in EXTS])

def to_rel(rois, w, h):
    return [(x/w, y/h, ww/w, hh/h) for x,y,ww,hh in rois]

def to_abs(rel_rois, w, h, pad=0):
    absb = []
    for rx,ry,rw,rh in rel_rois:
        x = int(round(rx*w)) - pad
        y = int(round(ry*h)) - pad
        ww = int(round(rw*w)) + 2*pad
        hh = int(round(rh*h)) + 2*pad
        x = max(0,x); y = max(0,y)
        absb.append((x,y, min(ww, w-x), min(hh, h-y)))
    return absb

def select_rois_on(image, scale_view=1.0, win="Define BBoxes"):
    h, w = image.shape[:2]
    disp = image
    if scale_view != 1.0:
        disp = cv.resize(image, (int(w*scale_view), int(h*scale_view)))
    cv.namedWindow(win, cv.WINDOW_NORMAL)
    rois = cv.selectROIs(win, disp, showCrosshair=True, fromCenter=False)
    cv.destroyWindow(win)
    # reescalar coords a tamaño original
    if scale_view != 1.0:
        rois = [(int(x/scale_view), int(y/scale_view),
                 int(wb/scale_view), int(hb/scale_view))
                for x,y,wb,hb in rois]
    return rois

def main(in_dir, out_dir, ref_path=None, pad=0, preview=False, scale_view=0.5):
    imgs = list_images(in_dir)
    if not imgs:
        print("No hay imágenes en la carpeta.")
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    ref_img_path = Path(ref_path) if ref_path else imgs[0]
    ref = cv.imread(str(ref_img_path))
    if ref is None: raise FileNotFoundError(ref_img_path)
    rh, rw = ref.shape[:2]

    rois = select_rois_on(ref, scale_view)
    if not rois:
        print("No definiste BBoxes. Abortado.")
        return
    rel_rois = to_rel(rois, rw, rh)

    for i, img_path in enumerate(imgs, 1):
        img = cv.imread(str(img_path))
        if img is None:
            print(f"[WARN] No se pudo leer: {img_path}")
            continue
        h, w = img.shape[:2]
        abs_rois = to_abs(rel_rois, w, h, pad)

        saved = 0
        for j, (x,y,ww,hh) in enumerate(abs_rois, 1):
            if ww<=0 or hh<=0: continue
            crop = img[y:y+hh, x:x+ww]
            out_name = f"{img_path.stem}_bb{j:02d}.png"
            cv.imwrite(str(Path(out_dir)/out_name), crop)
            saved += 1

        print(f"[{i}/{len(imgs)}] {img_path.name}: {saved} recortes")

        if preview:
            vis = img.copy()
            for (x,y,ww,hh) in abs_rois:
                cv.rectangle(vis, (x,y), (x+ww,y+hh), (0,0,255), 2)
            cv.imshow("Preview (q para cerrar)", vis)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                preview = False
    cv.destroyAllWindows()
    print("Terminado.")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Definir BBoxes en una imagen y aplicarlos al resto.")
    ap.add_argument("--in",  dest="in_dir",  required=True, help="Carpeta de entrada")
    ap.add_argument("--out", dest="out_dir", required=True, help="Carpeta de salida")
    ap.add_argument("--ref", dest="ref_path", default=None, help="Imagen de referencia (opcional)")
    ap.add_argument("--pad", type=int, default=0, help="Padding en píxeles alrededor de cada bbox")
    ap.add_argument("--preview", action="store_true", help="Mostrar vista previa")
    ap.add_argument("--scale_view", type=float, default=0.5, help="Escala para mostrar la imagen de referencia")
    args = ap.parse_args()
    main(args.in_dir, args.out_dir, args.ref_path, args.pad, args.preview, args.scale_view)
