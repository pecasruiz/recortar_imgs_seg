import threading
from pathlib import Path

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import recortar_imgs as core


EXTS = core.EXTS


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Recortar imágenes")
        self.geometry("760x360")

        self.in_dir_var = tk.StringVar()
        self.out_dir_var = tk.StringVar()

        self._build_ui()

    def _build_ui(self):
        root = ttk.Frame(self, padding=12)
        root.pack(fill="both", expand=True)

        box = ttk.LabelFrame(root, text="Rutas", padding=10)
        box.pack(fill="x")

        ttk.Label(box, text="Carpeta de fotos (entrada)").grid(row=0, column=0, sticky="w")
        ttk.Entry(box, textvariable=self.in_dir_var).grid(row=0, column=1, sticky="we", padx=8)
        ttk.Button(box, text="Elegir...", command=self._pick_in).grid(row=0, column=2, sticky="e")

        ttk.Label(box, text="Carpeta de recortes (salida)").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(box, textvariable=self.out_dir_var).grid(row=1, column=1, sticky="we", padx=8, pady=(8, 0))
        ttk.Button(box, text="Elegir...", command=self._pick_out).grid(row=1, column=2, sticky="e", pady=(8, 0))

        box.columnconfigure(1, weight=1)

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(12, 0))
        self.btn = ttk.Button(actions, text="Procesar", command=self._run)
        self.btn.pack(side="left")

        self.status = tk.Text(root, height=8, wrap="word")
        self.status.pack(fill="both", expand=True, pady=(12, 0))
        self.status.configure(state="disabled")
        self._log("Selecciona carpeta de entrada y salida, y pulsa Procesar.")

    def _log(self, msg: str):
        self.status.configure(state="normal")
        self.status.insert("end", msg + "\n")
        self.status.see("end")
        self.status.configure(state="disabled")

    def _pick_in(self):
        p = filedialog.askdirectory(title="Carpeta de entrada")
        if p:
            self.in_dir_var.set(p)

    def _pick_out(self):
        p = filedialog.askdirectory(title="Carpeta de salida")
        if p:
            self.out_dir_var.set(p)

    def _validate(self):
        in_dir = Path(self.in_dir_var.get().strip())
        out_dir = Path(self.out_dir_var.get().strip())
        if not in_dir.exists() or not in_dir.is_dir():
            raise ValueError("Selecciona una carpeta de entrada válida.")
        if not out_dir:
            raise ValueError("Selecciona una carpeta de salida válida.")
        imgs = sorted([p for p in in_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTS])
        if not imgs:
            raise ValueError("La carpeta de entrada no contiene imágenes soportadas.")
        return in_dir, out_dir, len(imgs)

    def _run(self):
        try:
            in_dir, out_dir, count = self._validate()
        except Exception as e:
            messagebox.showerror("Error", str(e))
            return

        self.btn.configure(state="disabled")
        self._log("")
        self._log(f"Entrada: {in_dir} ({count} imágenes)")
        self._log(f"Salida: {out_dir}")

        def worker():
            try:
                written = core.crop_folder(in_dir, out_dir)
                self.after(0, lambda: self._log(f"Terminado. Recortes guardados: {written}"))
                self.after(0, lambda: messagebox.showinfo("OK", "Procesamiento terminado."))
            except Exception as e:
                self.after(0, lambda: messagebox.showerror("Error", str(e)))
            finally:
                self.after(0, lambda: self.btn.configure(state="normal"))

        threading.Thread(target=worker, daemon=True).start()


def main(*_args, **_kwargs):
    App().mainloop()


if __name__ == "__main__":
    main()