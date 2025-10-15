# refbbox_batch_crop.py (versión con escala de vista)
import cv2 as cv
from pathlib import Path
from natsort import natsorted
import argparse
import os

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

def load_yolo_model(weights_path):
    """Carga el modelo YOLO desde la ruta de pesos"""
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        return model
    except ImportError:
        print("Error: ultralytics no está instalado. Instálalo con: pip install ultralytics")
        return None
    except Exception as e:
        print(f"Error cargando modelo YOLO: {e}")
        return None

def detect_objects_yolo(model, image_path):
    """Realiza detección de objetos con YOLO en una imagen"""
    try:
        results = model(image_path)
        detections = []
        # Verificar si hay detecciones
        for result in results:
            if len(result.boxes) > 0:
                for box in result.boxes:
                    # Obtener coordenadas del bbox y confianza
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    class_id = int(box.cls[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': (int(x1), int(y1), int(x2), int(y2)),
                        'confidence': float(confidence),
                        'class_id': class_id
                    })
        return detections
    except Exception as e:
        print(f"Error en detección YOLO: {e}")
        return []

def create_detection_folders(out_dir):
    """Crea las carpetas con_deteccion y sin_deteccion"""
    con_detection_dir = Path(out_dir) / "con_deteccion"
    sin_detection_dir = Path(out_dir) / "sin_deteccion"
    
    con_detection_dir.mkdir(parents=True, exist_ok=True)
    sin_detection_dir.mkdir(parents=True, exist_ok=True)
    
    return con_detection_dir, sin_detection_dir

def save_image_list(folder_path, image_names, filename="lista_imagenes.txt"):
    """Guarda una lista de nombres de imágenes en un archivo .txt"""
    txt_path = Path(folder_path) / filename
    with open(txt_path, 'w', encoding='utf-8') as f:
        for name in sorted(image_names):
            f.write(f"{name}\n")

def save_detection_summary(con_detection_dir, detection_data, model=None):
    """Guarda un resumen detallado de las detecciones"""
    summary_path = con_detection_dir / "resumen_detecciones.txt"
    
    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("RESUMEN DE DETECCIONES\n")
        f.write("=" * 50 + "\n\n")
        
        total_detections = 0
        class_counts = {}
        
        for img_name, detections in detection_data.items():
            if detections:
                f.write(f"Imagen: {img_name}\n")
                f.write("-" * 30 + "\n")
                
                for i, det in enumerate(detections, 1):
                    x1, y1, x2, y2 = det['bbox']
                    conf = det['confidence']
                    class_id = det['class_id']
                    
                    class_name = f"Class {class_id}"
                    if model and hasattr(model, 'names'):
                        class_name = model.names.get(class_id, f"Class {class_id}")
                    
                    f.write(f"  Detección {i}:\n")
                    f.write(f"    Clase: {class_name} (ID: {class_id})\n")
                    f.write(f"    Confianza: {conf:.3f}\n")
                    f.write(f"    BBox: ({x1}, {y1}, {x2}, {y2})\n")
                    f.write(f"    Tamaño: {x2-x1}x{y2-y1} píxeles\n\n")
                    
                    total_detections += 1
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1
                
                f.write("\n")
        
        f.write("ESTADÍSTICAS GENERALES\n")
        f.write("=" * 50 + "\n")
        f.write(f"Total de detecciones: {total_detections}\n")
        f.write(f"Imágenes con detección: {len(detection_data)}\n\n")
        
        f.write("Conteo por clase:\n")
        for class_name, count in sorted(class_counts.items()):
            f.write(f"  {class_name}: {count}\n")

def draw_detections(image, detections, model=None):
    """Dibuja bounding boxes y confianza en la imagen"""
    if not detections:
        return image
    
    # Crear una copia de la imagen para no modificar la original
    img_with_boxes = image.copy()
    
    # Colores para diferentes clases (puedes personalizar)
    colors = [
        (0, 255, 0),    # Verde
        (255, 0, 0),    # Azul
        (0, 0, 255),    # Rojo
        (255, 255, 0),  # Cian
        (255, 0, 255),  # Magenta
        (0, 255, 255),  # Amarillo
    ]
    
    for i, detection in enumerate(detections):
        x1, y1, x2, y2 = detection['bbox']
        confidence = detection['confidence']
        class_id = detection['class_id']
        
        # Seleccionar color basado en la clase
        color = colors[class_id % len(colors)]
        
        # Dibujar bounding box
        cv.rectangle(img_with_boxes, (x1, y1), (x2, y2), color, 2)
        
        # Preparar texto con confianza
        label = f"Conf: {confidence:.2f}"
        
        # Obtener el nombre de la clase si el modelo lo tiene
        if model and hasattr(model, 'names'):
            class_name = model.names.get(class_id, f"Class {class_id}")
            label = f"{class_name}: {confidence:.2f}"
        
        # Calcular tamaño del texto
        font = cv.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv.getTextSize(label, font, font_scale, thickness)
        
        # Dibujar fondo para el texto
        cv.rectangle(img_with_boxes, (x1, y1 - text_height - 10), 
                    (x1 + text_width, y1), color, -1)
        
        # Dibujar texto
        cv.putText(img_with_boxes, label, (x1, y1 - 5), 
                  font, font_scale, (255, 255, 255), thickness)
    
    return img_with_boxes

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

def main(in_dir, out_dir, ref_path=None, pad=0, preview=False, scale_view=0.5, yolo_weights=None):
    imgs = list_images(in_dir)
    if not imgs:
        print("No hay imágenes en la carpeta.")
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    # Cargar modelos YOLO si se proporcionan
    models = []
    if yolo_weights:
        print(f"Cargando {len([w for w in yolo_weights if w])} modelos YOLO...")
        for i, weight_path in enumerate(yolo_weights):
            if weight_path:
                print(f"Cargando modelo {i+1} desde: {weight_path}")
                model = load_yolo_model(weight_path)
                if model:
                    models.append(model)
                    print(f"Modelo {i+1} cargado correctamente")
                else:
                    models.append(None)
                    print(f"Error al cargar modelo {i+1}")
            else:
                models.append(None)
        
        if any(models):
            print("Al menos un modelo YOLO cargado correctamente")
            # Crear carpetas de detección
            create_detection_folders(out_dir)
        else:
            print("Ningún modelo YOLO se pudo cargar. Continuando sin detección.")
            models = []

    ref_img_path = Path(ref_path) if ref_path else imgs[0]
    ref = cv.imread(str(ref_img_path))
    if ref is None: raise FileNotFoundError(ref_img_path)
    rh, rw = ref.shape[:2]

    rois = select_rois_on(ref, scale_view)
    if not rois:
        print("No definiste BBoxes. Abortado.")
        return
    rel_rois = to_rel(rois, rw, rh)

    # Si hay modelos YOLO, crear carpetas de detección
    con_detection_dir = None
    sin_detection_dir = None
    con_detection_images = []
    sin_detection_images = []
    detection_data = {}  # Para almacenar datos detallados de detecciones
    
    if models and any(models):
        con_detection_dir, sin_detection_dir = create_detection_folders(out_dir)
        print("Carpetas de detección creadas: con_deteccion y sin_deteccion")

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
            out_path = Path(out_dir) / out_name
            
            # Si hay modelos YOLO, usar el modelo correspondiente al recorte
            if models and j <= len(models) and models[j-1] is not None:
                model = models[j-1]
                crop_path = Path(out_dir) / "temp_crop.png"
                cv.imwrite(str(crop_path), crop)
                
                detections = detect_objects_yolo(model, str(crop_path))
                if detections:
                    # Dibujar detecciones en la imagen
                    crop_with_boxes = draw_detections(crop, detections, model)
                    cv.imwrite(str(out_path), crop_with_boxes)
                    con_detection_images.append(out_name)
                    detection_data[out_name] = detections
                    print(f"  -> Recorte {j} con modelo {j}: {len(detections)} detecciones")
                else:
                    cv.imwrite(str(out_path), crop)
                    sin_detection_images.append(out_name)
                    print(f"  -> Recorte {j} con modelo {j}: Sin detecciones")
                
                # Limpiar archivo temporal
                crop_path.unlink()
            else:
                # Sin modelo YOLO para este recorte, guardar directamente
                cv.imwrite(str(out_path), crop)
                if models:
                    sin_detection_images.append(out_name)
                    print(f"  -> Recorte {j}: Sin modelo asignado")
            
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
    
    # Si se usaron modelos YOLO, mover imágenes a carpetas correspondientes y generar archivos
    if models and any(models):
        # Mover imágenes a carpetas correspondientes
        for img_name in con_detection_images:
            src = Path(out_dir) / img_name
            dst = con_detection_dir / img_name
            if src.exists():
                src.rename(dst)
        
        for img_name in sin_detection_images:
            src = Path(out_dir) / img_name
            dst = sin_detection_dir / img_name
            if src.exists():
                src.rename(dst)
        
        # Generar archivos .txt con listas de imágenes
        save_image_list(con_detection_dir, con_detection_images)
        save_image_list(sin_detection_dir, sin_detection_images)
        
        # Generar resumen de detecciones
        save_detection_summary(con_detection_dir, detection_data, models[0] if models else None)
        
        print(f"Imágenes con detección: {len(con_detection_images)}")
        print(f"Imágenes sin detección: {len(sin_detection_images)}")
        print(f"Resumen detallado guardado en: {con_detection_dir}/resumen_detecciones.txt")
    
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
    ap.add_argument("--weights", dest="weights_path", default=None, help="Ruta de pesos YOLO (opcional)")
    args = ap.parse_args()
    main(args.in_dir, args.out_dir, args.ref_path, args.pad, args.preview, args.scale_view, args.weights_path)
