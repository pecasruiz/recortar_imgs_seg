# recortar_imgs_menu.py
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
import threading

# Importa tu script existente
import recortar_imgs as core

def run_main(in_dir, out_dir, ref_path, pad, preview, scale_view, yolo_weights, btn):
    try:
        btn.config(state="disabled")
        core.main(in_dir, out_dir, ref_path if ref_path else None, pad, preview, scale_view, yolo_weights)
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

def select_yolo_weight(weight_num):
    p = filedialog.askopenfilename(title=f"Archivo de pesos YOLO {weight_num} (opcional)",
                                   filetypes=[("Pesos YOLO","*.pt *.onnx"), ("Todos","*.*")])
    if p: 
        yolo_vars[weight_num-1].set(p)

def start():
    in_dir  = in_var.get().strip()
    out_dir = out_var.get().strip()
    ref_path = ref_var.get().strip()
    
    # Recopilar todas las rutas de pesos YOLO
    yolo_weights = []
    for i, var in enumerate(yolo_vars):
        weight_path = var.get().strip()
        if weight_path:
            if not Path(weight_path).exists():
                messagebox.showerror("Error", f"No existe el archivo de pesos {i+1}: {weight_path}")
                return
            yolo_weights.append(weight_path)
        else:
            yolo_weights.append(None)
    
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
    
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    btn_run = btn_start
    t = threading.Thread(target=run_main, args=(in_dir, out_dir, ref_path, pad, preview_var.get()==1, scale, yolo_weights, btn_run), daemon=True)
    t.start()

root = tk.Tk()
root.title("Recortar Imágenes - Menú")
root.geometry("1200x800")

# Variables principales
in_var = tk.StringVar()
out_var = tk.StringVar()
ref_var = tk.StringVar()
scale_var = tk.StringVar(value="0.5")
pad_var = tk.StringVar(value="0")
preview_var = tk.IntVar(value=0)

# Variables para los 10 pesos YOLO
yolo_vars = [tk.StringVar() for _ in range(10)]

# Frame principal con scroll
canvas = tk.Canvas(root)
scrollbar = tk.Scrollbar(root, orient="vertical", command=canvas.yview)
scrollable_frame = tk.Frame(canvas)

scrollable_frame.bind(
    "<Configure>",
    lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
)

canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
canvas.configure(yscrollcommand=scrollbar.set)

# Configuración básica
row = 0
tk.Label(scrollable_frame, text="Carpeta de entrada:").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(scrollable_frame, textvariable=in_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(scrollable_frame, text="Seleccionar...", command=select_in).grid(row=row, column=2, padx=4); row+=1

tk.Label(scrollable_frame, text="Carpeta de salida:").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(scrollable_frame, textvariable=out_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(scrollable_frame, text="Seleccionar...", command=select_out).grid(row=row, column=2, padx=4); row+=1

tk.Label(scrollable_frame, text="Imagen de referencia (opcional):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(scrollable_frame, textvariable=ref_var, width=60).grid(row=row, column=1, padx=4)
tk.Button(scrollable_frame, text="Elegir...", command=select_ref).grid(row=row, column=2, padx=4); row+=1

# Separador
tk.Label(scrollable_frame, text="─" * 80, fg="gray").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row+=1

# Título para pesos YOLO
tk.Label(scrollable_frame, text="Rutas de Pesos YOLO (opcional)", font=("Arial", 12, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", padx=8, pady=6); row+=1
tk.Label(scrollable_frame, text="Cada peso se aplicará al recorte correspondiente (1º peso → 1º recorte, 2º peso → 2º recorte, etc.)", 
         fg="blue").grid(row=row, column=0, columnspan=3, sticky="w", padx=8, pady=2); row+=1

# 10 campos para pesos YOLO
for i in range(10):
    tk.Label(scrollable_frame, text=f"Peso YOLO {i+1}:").grid(row=row, column=0, sticky="w", padx=8, pady=4)
    tk.Entry(scrollable_frame, textvariable=yolo_vars[i], width=60).grid(row=row, column=1, padx=4)
    tk.Button(scrollable_frame, text="Elegir...", command=lambda num=i+1: select_yolo_weight(num)).grid(row=row, column=2, padx=4); row+=1

# Separador
tk.Label(scrollable_frame, text="─" * 80, fg="gray").grid(row=row, column=0, columnspan=3, sticky="ew", pady=10); row+=1

# Configuración adicional
tk.Label(scrollable_frame, text="Escala de vista (0.1–1.0):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(scrollable_frame, textvariable=scale_var, width=12).grid(row=row, column=1, sticky="w", padx=4); row+=1

tk.Label(scrollable_frame, text="Padding (px):").grid(row=row, column=0, sticky="w", padx=8, pady=6)
tk.Entry(scrollable_frame, textvariable=pad_var, width=12).grid(row=row, column=1, sticky="w", padx=4); row+=1

tk.Checkbutton(scrollable_frame, text="Preview", variable=preview_var).grid(row=row, column=1, sticky="w", padx=4); row+=1

# Botón de inicio
btn_start = tk.Button(scrollable_frame, text="Iniciar", command=start, width=18, font=("Arial", 12, "bold"), bg="#4CAF50", fg="white")
btn_start.grid(row=row, column=1, pady=20)

# Configurar scroll
canvas.pack(side="left", fill="both", expand=True)
scrollbar.pack(side="right", fill="y")

root.mainloop()