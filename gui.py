#!/usr/bin/env python3
"""Tkinter GUI for the brake-disc hole analyzer.

Run with::

    python gui.py
"""

from __future__ import annotations

import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from brake_disc import BrakeDiscDetector, DetectorConfig, draw_result, result_summary_text

current_dir = os.path.dirname(os.path.abspath(__file__))
results_dir = os.path.join(current_dir, "results")
os.makedirs(results_dir, exist_ok=True)


class BrakeDiscApp:
    """Graphical interface wrapping :class:`BrakeDiscDetector`."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Brake Disc Hole Analyzer")
        self.root.geometry("1100x750")
        self.root.configure(bg="#f0f0f0")

        self.detector = BrakeDiscDetector()
        self.image_path: str | None = None
        self.original_image = None
        self.result_image = None
        self.result = None

        self._build_ui()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        # Title bar
        title = tk.Frame(self.root, bg="#2c3e50", height=55)
        title.pack(fill=tk.X)
        tk.Label(
            title, text="Brake Disc Hole Analyzer",
            font=("Arial", 18, "bold"), fg="white", bg="#2c3e50",
        ).pack(pady=12)

        # Buttons
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=12)
        self._make_button(btn_frame, "Select Image", self.select_image, "#3498db").pack(side=tk.LEFT, padx=8)
        self.analyze_btn = self._make_button(btn_frame, "Analyze", self.analyze_image, "#27ae60")
        self.analyze_btn.pack(side=tk.LEFT, padx=8)
        self.save_btn = self._make_button(btn_frame, "Save Result", self.save_result, "#e67e22")
        self.save_btn.pack(side=tk.LEFT, padx=8)

        # Images
        img_frame = tk.Frame(self.root, bg="#f0f0f0")
        img_frame.pack(fill=tk.BOTH, expand=True, padx=10)

        left = tk.Frame(img_frame, bg="white", relief=tk.RAISED, bd=2)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(left, text="Original", font=("Arial", 11, "bold"), bg="white").pack(pady=4)
        self.original_canvas = tk.Canvas(left, bg="#e8e8e8")
        self.original_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        right = tk.Frame(img_frame, bg="white", relief=tk.RAISED, bd=2)
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        tk.Label(right, text="Analysis Result", font=("Arial", 11, "bold"), bg="white").pack(pady=4)
        self.result_canvas = tk.Canvas(right, bg="#e8e8e8")
        self.result_canvas.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Result text
        text_frame = tk.Frame(self.root, bg="#f0f0f0")
        text_frame.pack(fill=tk.X, padx=15, pady=8)
        tk.Label(text_frame, text="Measurements", font=("Arial", 11, "bold"), bg="#f0f0f0").pack(anchor=tk.W)
        self.result_text = tk.Text(text_frame, height=9, font=("Consolas", 10), bg="#f8f9fa", relief=tk.SUNKEN, bd=1)
        self.result_text.pack(fill=tk.X, pady=4)

        # Status bar
        status = tk.Frame(self.root, bg="#34495e", height=28)
        status.pack(fill=tk.X, side=tk.BOTTOM)
        self.status_label = tk.Label(status, text="Ready", font=("Arial", 9), fg="white", bg="#34495e")
        self.status_label.pack(anchor=tk.W, padx=10, pady=5)

    @staticmethod
    def _make_button(parent, text, cmd, color):
        return tk.Button(
            parent, text=text, command=cmd, width=14, height=2,
            bg=color, fg="white", font=("Arial", 10, "bold"), relief=tk.FLAT,
        )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def select_image(self) -> None:
        path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        self.image_path = path
        self.original_image = cv2.imread(path)
        if self.original_image is None:
            messagebox.showerror("Error", "Cannot read image.")
            return
        self._show_image(self.original_canvas, self.original_image)
        self.result_canvas.delete("all")
        self.result_text.delete(1.0, tk.END)
        self.status_label.config(text=f"Loaded: {os.path.basename(path)}")

    def analyze_image(self) -> None:
        if self.original_image is None:
            messagebox.showinfo("Info", "Please select an image first.")
            return
        self.status_label.config(text="Analyzing...")
        self.root.update()
        try:
            self.result = self.detector.analyze(self.original_image)
            self.result_image = draw_result(self.original_image, self.result)
            self._show_image(self.result_canvas, self.result_image)
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, result_summary_text(self.result))
            self.status_label.config(text="Analysis complete")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Analysis failed: {exc}")
            self.status_label.config(text="Analysis failed")

    def save_result(self) -> None:
        if self.result_image is None:
            messagebox.showinfo("Info", "Please analyze an image first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All", "*.*")],
        )
        if not path:
            return
        try:
            cv2.imwrite(path, self.result_image)
            messagebox.showinfo("Success", f"Saved to:\n{path}")
            self.status_label.config(text=f"Saved: {os.path.basename(path)}")
        except Exception as exc:  # noqa: BLE001
            messagebox.showerror("Error", f"Save failed: {exc}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _show_image(self, canvas: tk.Canvas, image) -> None:
        self.root.update_idletasks()
        cw = max(canvas.winfo_width(), 1)
        ch = max(canvas.winfo_height(), 1)
        h, w = image.shape[:2]
        scale = min(cw / w, ch / h)
        nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
        resized = cv2.resize(image, (nw, nh))
        if resized.ndim == 2:
            pil = Image.fromarray(resized, mode="L")
        else:
            pil = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        tk_img = ImageTk.PhotoImage(pil)
        canvas.delete("all")
        canvas.create_image((cw - nw) // 2, (ch - nh) // 2, anchor=tk.NW, image=tk_img)
        canvas.image = tk_img  # keep reference


def main() -> None:
    root = tk.Tk()
    BrakeDiscApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
