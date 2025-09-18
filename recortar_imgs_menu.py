# recortar_imgs_menu.py
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

# Importa tu script existente
import recortar_imgs as core  # requiere que definas core.main(in_dir,out_dir,ref_path,pad,preview,scale_view,weights_path)

def run_main(in_dir, out_dir, ref_path, pad, preview, scale_view, weights_path, btn):
    try:
        btn.config(state="disabled")
        core.main(in_dir, out_dir, ref_path if ref_path else None, pad, preview, scale_view, weights_path if weights_path else None)
        messagebox.showinfo("Listo", "Proceso terminado.")
    except Exception as e:
        messagebox.showerror("Error", str(e))
    finally:
        btn.config(state="normal")

def select_in():
    p = filedialog.askdirectory(title="Carpeta de entrada")
    if p: in_var.set(p)

def select_out():
    p = filedialog.askdirectory(title="Carpeta de salida")
    if p: out_var.set(p)

def select_ref():
    p = filedialog.askopenfilename(title="Imagen de referencia (opcional)",
                                   filetypes=[("Imágenes","*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp"), ("Todos","*.*")])
    if p: ref_var.set(p)

def select_weights():
    p = filedialog.askopenfilename(title="Archivo de pesos YOLO (opcional)",
                                   filetypes=[("Pesos YOLO","*.pt *.onnx"), ("Todos","*.*")])
    if p: weights_var.set(p)

def start():
    in_dir  = in_var.get().strip()
    out_dir = out_var.get().strip()
    ref_path = ref_var.get().strip()
    weights_path = weights_var.get().strip()
    try:
        scale = float(scale_var.get())
        pad   = int(pad_var.get())
    except:
        messagebox.showerror("Error", "Escala debe ser float (ej. 0.5) y padding un entero.")
        return
    if not in_dir or not out_dir:
        messagebox.showerror("Faltan datos", "Entrada y salida son obligatorias.")
        return
    # Valida existencia
    if not Path(in_dir).exists():
        messagebox.showerror("Error", f"No existe: {in_dir}")
        return
    if weights_path and not Path(weights_path).exists():
        messagebox.showerror("Error", f"No existe el archivo de pesos: {weights_path}")
        return
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    btn_run = btn_start
    t = threading.Thread(target=run_main, args=(in_dir, out_dir, ref_path, pad, preview_var.get()==1, scale, weights_path, btn_run), daemon=True)
    t.start()

root = tk.Tk()
root.title("Recortar Imágenes - Menú")
root.geometry("950x380")

in_var = tk.StringVar()
out_var = tk.StringVar()
ref_var = tk.StringVar()
weights_var = tk.StringVar()
scale_var = tk.StringVar(value="0.5")
pad_var = tk.StringVar(value="0")
preview_var = tk.IntVar(value=0)

row = 0
tk.Label(root, text="Carpeta de entrada:").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(root, textvariable=in_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(root, text="Seleccionar...", command=select_in).grid(row=row, column=2, padx=4); row+=1

tk.Label(root, text="Carpeta de salida:").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(root, textvariable=out_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(root, text="Seleccionar...", command=select_out).grid(row=row, column=2, padx=4); row+=1

tk.Label(root, text="Imagen de referencia (opcional):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(root, textvariable=ref_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(root, text="Elegir...", command=select_ref).grid(row=row, column=2, padx=4); row+=1

tk.Label(root, text="Ruta de pesos YOLO (opcional):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(root, textvariable=weights_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(root, text="Elegir...", command=select_weights).grid(row=row, column=2, padx=4); row+=1

tk.Label(root, text="Escala de vista (0.1–1.0):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(root, textvariable=scale_var, width=12).grid(row=row, column=1, sticky="w", padx=4); row+=1

tk.Label(root, text="Padding (px):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(root, textvariable=pad_var, width=12).grid(row=row, column=1, sticky="w", padx=4); row+=1

tk.Checkbutton(root, text="Preview", variable=preview_var).grid(row=row, column=1, sticky="w", padx=4); row+=1

btn_start = tk.Button(root, text="Iniciar", command=start, width=18)
btn_start.grid(row=row, column=1, pady=16)

root.mainloop()
