# Recortar Imágenes con Detección YOLO

Herramienta para recortar imágenes basándose en áreas definidas manualmente, con funcionalidad opcional de detección de objetos usando YOLO.

## Características

- **Recorte de imágenes**: Define áreas de recorte en una imagen de referencia y aplica los mismos recortes a todas las imágenes de una carpeta
- **Detección YOLO opcional**: Si se proporciona un modelo YOLO, las imágenes recortadas se clasifican automáticamente según si contienen objetos detectados
- **Visualización de detecciones**: Las imágenes con detecciones muestran bounding boxes y niveles de confianza
- **Interfaz gráfica**: Menú fácil de usar con tkinter
- **Organización automática**: Cuando se usa YOLO, las imágenes se organizan en carpetas `con_deteccion` y `sin_deteccion`
- **Listas de archivos**: Se generan archivos `.txt` con los nombres de las imágenes en cada carpeta
- **Resumen detallado**: Archivo con información completa de todas las detecciones encontradas

## Instalación

1. Instala las dependencias:
```bash
pip install -r requirements.txt
```

## Uso

### Interfaz Gráfica (Recomendado)

Ejecuta el menú gráfico:
```bash
python recortar_imgs_menu.py
```

**Campos del menú:**
- **Carpeta de entrada**: Carpeta con las imágenes a procesar
- **Carpeta de salida**: Carpeta donde se guardarán los recortes
- **Imagen de referencia** (opcional): Imagen para definir las áreas de recorte
- **Ruta de pesos YOLO** (opcional): Archivo `.pt` o `.onnx` del modelo YOLO
- **Escala de vista**: Factor de escala para la vista previa (0.1-1.0)
- **Padding**: Píxeles adicionales alrededor de cada recorte
- **Preview**: Mostrar vista previa de los recortes

### Línea de Comandos

```bash
python recortar_imgs.py --in carpeta_entrada --out carpeta_salida [opciones]
```

**Opciones:**
- `--ref`: Imagen de referencia (opcional)
- `--weights`: Ruta de pesos YOLO (opcional)
- `--pad`: Padding en píxeles (default: 0)
- `--preview`: Mostrar vista previa
- `--scale_view`: Escala de vista (default: 0.5)

## Flujo de Trabajo

### Sin YOLO (Modo Normal)
1. Selecciona carpeta de entrada y salida
2. Define áreas de recorte en la imagen de referencia
3. Las imágenes se recortan y guardan en la carpeta de salida

### Con YOLO (Modo Detección)
1. Selecciona carpeta de entrada y salida
2. Proporciona ruta de pesos YOLO
3. Define áreas de recorte en la imagen de referencia
4. Las imágenes se recortan y se analizan con YOLO
5. Los recortes se organizan en:
   - `con_deteccion/`: Recortes que contienen objetos detectados (con bboxes dibujados)
   - `sin_deteccion/`: Recortes sin objetos detectados
6. Se generan archivos `lista_imagenes.txt` en cada carpeta
7. Se crea un archivo `resumen_detecciones.txt` con información detallada de todas las detecciones

## Requisitos

- Python 3.7+
- OpenCV
- Ultralytics (para YOLO)
- natsort

## Visualización de Detecciones

Cuando se usa YOLO, las imágenes en la carpeta `con_deteccion` incluyen:

- **Bounding boxes**: Rectángulos de colores que marcan los objetos detectados
- **Etiquetas de confianza**: Muestran el nombre de la clase y el nivel de confianza (ej: "person: 0.85")
- **Colores por clase**: Cada clase de objeto tiene un color diferente para facilitar la identificación
- **Fondo para texto**: Las etiquetas tienen un fondo del mismo color que el bbox para mejor legibilidad

## Archivos Generados

### Con YOLO habilitado:
- `con_deteccion/`: Imágenes con objetos detectados (con bboxes dibujados)
- `sin_deteccion/`: Imágenes sin objetos detectados
- `con_deteccion/lista_imagenes.txt`: Lista de imágenes con detecciones
- `sin_deteccion/lista_imagenes.txt`: Lista de imágenes sin detecciones
- `con_deteccion/resumen_detecciones.txt`: Resumen detallado con estadísticas y coordenadas

### Sin YOLO:
- Imágenes recortadas guardadas directamente en la carpeta de salida

## Notas

- Si no se proporciona imagen de referencia, se usa la primera imagen de la carpeta de entrada
- El modelo YOLO debe ser compatible con Ultralytics (formatos `.pt` o `.onnx`)
- Si hay errores al cargar YOLO, el programa continúa en modo normal sin detección
- Las detecciones se muestran con diferentes colores para facilitar la identificación visual