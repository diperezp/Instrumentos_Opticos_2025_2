# ==============================================================
# punto2_4f_sim.py
# Simulación difractiva (Punto 2) con formalismo ABCD + Fresnel
# - Selección de imagen (solo imágenes) vía Tkinter
# - Sistemas 4f anterior y posterior (propagación-foco)
# - Filtros en plano de Fourier: binario circular / rendija / gaussiano
# - Gráficas de todas las etapas (intensidad)
# ==============================================================

from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# ----------------------------- Configuración -----------------------------
CONFIG = {
    # Longitud de onda (m)
    "lambda_0": 532e-9,

    # Focos de las lentes (m)
    "f_lente_1": 500e-3,
    "f_lente_2": 500e-3,

    # Sensor (tamaño y muestreo)
    "px_cols": 2448,
    "px_rows": 2048,
    "pitch": 3.45e-6,  # tamaño de píxel (m)

    # Filtro del plano de Fourier (elige: "circular", "circular_rendija", "gaussiano")
    "tipo_filtro": "gaussiano_sumado_invertido",   # <– usa este tipo
    "sigma_gauss": 300e-6,                         # = lado en tu código
    "centros_gauss": [                             # (x0, y0) en metros
        (1.58e-3,  1.01e-3),
        (-7.75e-4,   6.04e-4),
        ( -4.67e-3,   9.8e-4),
        (-1.44e-3,  1.20e-3),
        ( 1.67e-3,  -9.2e-4),
        ( -1.50e-3, -9.4e-4),
        ( 1.62e-3,  -1.15e-3),
        (4.73e-3, -1.04e-3),
    ],
    "diam_apertura": 100e-3,   # sigue controlando el diafragma físico


    # Visualización (porcentaje del máximo)
    "clim": {
        "objeto": (0.0, 1.0),
        "antes_diafragma": (0.0, 1e-5),
        "filtro": (0.0, 1.0),
        "tras_diafragma": (0.0, 1e-5),
        "sensor": (0.2, 0.7),
    }
}

# --------------------------- Utilidades básicas ---------------------------
def seleccionar_imagen_en_grises(inicio_en_escritorio: bool = True) -> np.ndarray:
    """Abre un diálogo para seleccionar una imagen. Devuelve matriz [0,1] en escala de grises."""
    root = tk.Tk(); root.withdraw()
    initial_dir = os.path.join(os.path.expanduser("~"), "Desktop") if inicio_en_escritorio else None
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen (PNG/JPG/JPEG/BMP/TIFF)",
        initialdir=initial_dir,
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
    )
    root.update(); root.destroy()
    if not ruta:
        raise RuntimeError("No se seleccionó ninguna imagen.")
    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("No fue posible leer la imagen seleccionada.")
    return img.astype(np.float64) / 255.0

def redimensionar_con_padding(img: np.ndarray, nuevo_ancho: int, nuevo_alto: int) -> np.ndarray:
    """Redimensiona manteniendo aspecto y rellena con bordes para encajar exactamente (ancho x alto)."""
    h, w = img.shape
    ratio = min(nuevo_ancho / w, nuevo_alto / h)
    nuevo_w, nuevo_h = int(w * ratio), int(h * ratio)
    img_rs = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)
    pad_w = max(nuevo_ancho - nuevo_w, 0)
    pad_h = max(nuevo_alto - nuevo_h, 0)
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    return cv2.copyMakeBorder(img_rs, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)

def malla_cartesiana(n_col: int, Lx: float, n_row: int, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    """Crea malla cartesiana centrada en 0 con tamaños físicos Lx × Ly."""
    x = np.linspace(-Lx/2, Lx/2, n_col)
    y = np.linspace(-Ly/2, Ly/2, n_row)
    return np.meshgrid(x, y)

def mostrar_intensidad(campo: np.ndarray, Lx: float, Ly: float, titulo: str,
                       vmin_frac: float, vmax_frac: float):
    """Muestra |campo|^2 con ejes físicos (m). vmin/vmax como fracción del máximo."""
    I = np.abs(campo)**2
    vmax = np.max(I) if np.max(I) > 0 else 1.0
    extent = (-Lx/2, Lx/2, -Ly/2, Ly/2)
    plt.imshow(I, extent=extent, origin='lower', cmap='gray',
               vmin=vmin_frac * vmax, vmax=vmax_frac * vmax)
    plt.colorbar(label="Intensidad (u.a.)")
    plt.xlabel("x (m)"); plt.ylabel("y (m)"); plt.title(titulo)
    plt.show()

# --------------------------- Muestreo de Fresnel --------------------------
def muestreo_entrada_para_Fresnel(Nx: int, Lx_out: float, B: float, lambda_0: float,
                                  Ny: int, Ly_out: float) -> tuple[float, float, float, float]:
    """
    Calcula el muestreo requerido en el plano de ENTRADA dado el tamaño del plano de SALIDA
    para una propagación tipo Fresnel asociada al término B de la matriz ABCD.
    """
    dx_out, dy_out = Lx_out / Nx, Ly_out / Ny
    dx_in = (lambda_0 * B) / (Nx * dx_out)
    dy_in = (lambda_0 * B) / (Ny * dy_out)
    Lx_in, Ly_in = dx_in * Nx, dy_in * Ny
    return dx_in, dy_in, Lx_in, Ly_in

# ------------------------------ Filtros 4f --------------------------------
def mascara_circular(xx: np.ndarray, yy: np.ndarray, radio: float) -> np.ndarray:
    """Máscara binaria circular centrada (1 adentro, 0 afuera)."""
    return ((xx**2 + yy**2) <= radio**2).astype(float)

def mascara_rendija_central(xx: np.ndarray, ancho: float) -> np.ndarray:
    """Rendija vertical centrada (1 dentro de la rendija, 0 fuera)."""
    return (np.abs(xx) <= (ancho/2)).astype(float)

def filtro_plano_fourier(xx: np.ndarray, yy: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Devuelve filtro en el plano de Fourier según CONFIG['tipo_filtro'].
    - "circular": disco central transmitivo.
    - "circular_rendija": disco + rendija vertical.
    - "gaussiano": gaussiana central (demo).
    - "gaussiano_sumado_invertido": SUMA de gaussianas en (x0,y0), luego 1 - suma, y
                                    finalmente se aplicará el diafragma circular fuera.
    """
    tipo = cfg["tipo_filtro"]

    if tipo == "circular":
        return mascara_circular(xx, yy, cfg["radio_pasa_centro"])

    if tipo == "circular_rendija":
        disco = mascara_circular(xx, yy, cfg["radio_pasa_centro"])
        slit  = mascara_rendija_central(xx, cfg["ancho_rendija"])
        return np.maximum(disco, slit)

    if tipo == "gaussiano":
        sigma = cfg.get("sigma_gauss", 300e-6)
        g = np.exp(-((xx**2 + yy**2)/(2*sigma**2)))
        return g / np.max(g)

    if tipo == "gaussiano_sumado_invertido":
        sigma = cfg.get("sigma_gauss", 300e-6)
        centros = cfg.get("centros_gauss", [])
        # Suma de gaussianas en los centros indicados
        gsum = np.zeros_like(xx, dtype=float)
        inv_2sig2 = 1.0 / (2.0 * sigma**2)
        for (x0, y0) in centros:
            gsum += np.exp(-(((xx - x0)**2 + (yy - y0)**2) * inv_2sig2))
        # Invertir como en tu flujo original: filtro = 1 - suma
        # (se recorta al rango [0,1] para evitar valores negativos)
        filtro = 1.0 - gsum
        filtro = np.clip(filtro, 0.0, 1.0)
        return filtro

    raise ValueError("tipo_filtro no reconocido.")


# ------------------------------- ABCD -------------------------------------
@dataclass
class SistemaABCD:
    """Contenedor para una cadena de elementos ABCD y metadatos necesarios."""
    M: np.ndarray           # Matriz 2x2 del sistema
    B_total: float          # Término B neto (m)
    camino_optico: float    # z efectivo para fase global

def T(d: float, n: float = 1.0) -> np.ndarray:
    """Traslación en medio índice n: [[1, d/n],[0,1]]."""
    return np.array([[1.0, d/n],
                     [0.0, 1.0]], dtype=float)

def L(f: float) -> np.ndarray:
    """Lente delgada: [[1,0],[-1/f,1]]."""
    return np.array([[1.0, 0.0],
                     [-1.0/f, 1.0]], dtype=float)

def encadenar_elementos(elementos: list[np.ndarray]) -> SistemaABCD:
    """Multiplica en orden de propagación y acumula B y camino óptico (Aprox.)."""
    Msys = np.eye(2)
    B_total = 0.0
    for E in elementos:
        Msys = Msys @ E
        B_total += E[0, 1]
    # En régimen paraxial, se usa B_total como z efectivo de Fresnel.
    return SistemaABCD(M=Msys, B_total=B_total, camino_optico=B_total)

# ---------------- Propagación Fresnel con formalismo ABCD -----------------
def propagar_Fresnel_ABCD(objeto: np.ndarray, sistema: SistemaABCD,
                          Lx_out: float, Ly_out: float, lambda_0: float) -> tuple[np.ndarray, float, float]:
    """
    Propagación escalar de Fresnel usando la forma de Collins/ABCD discreta.
    Devuelve el campo en salida y tamaños físicos Lx_out, Ly_out.
    """
    Ny, Nx = objeto.shape
    A, B, C, D = sistema.M[0,0], sistema.M[0,1], sistema.M[1,0], sistema.M[1,1]
    k = 2*np.pi / lambda_0

    # Muestreo requerido en la ENTRADA para obtener ventanas Lx_out×Ly_out en la SALIDA.
    dx_in, dy_in, Lx_in, Ly_in = muestreo_entrada_para_Fresnel(Nx, Lx_out, B, lambda_0, Ny, Ly_out)

    # Mallas de entrada y salida
    xx_in, yy_in   = malla_cartesiana(Nx, Lx_in, Ny, Ly_in)
    xx_out, yy_out = malla_cartesiana(Nx, Lx_out, Ny, Ly_out)

    # Fases cuadráticas (Collins/Fourier Fresnel)
    fase_const = np.exp(1j * k * sistema.camino_optico)
    fase_in  = np.exp(1j * k * (A/(2*B)) * (xx_in**2 + yy_in**2))
    fase_out = np.exp(1j * k * (D/(2*B)) * (xx_out**2 + yy_out**2))

    # Transformada de Fresnel discreta (FFT2)
    # Nota: el factor (dx_out*dy_out) se maneja vía ventanas físicas de salida.
    campo_out = (fase_const * fase_out *
                 np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(objeto * fase_in))) *
                 (dx_in * dy_in) * (1/(1j * lambda_0 * B)))

    return campo_out, Lx_out, Ly_out

# --------------------------------- Main -----------------------------------
def main():
    # ------------------- Parámetros físicos del sensor -------------------
    nx, ny = CONFIG["px_cols"], CONFIG["px_rows"]
    pitch = CONFIG["pitch"]
    Lx_sensor, Ly_sensor = nx * pitch, ny * pitch

    # ------------------- Selección y preparación del objeto --------------
    objeto = seleccionar_imagen_en_grises(inicio_en_escritorio=True)
    objeto = redimensionar_con_padding(objeto, nx, ny)

    # ------------------- Definición de sistemas 4f -----------------------
    f1, f2 = CONFIG["f_lente_1"], CONFIG["f_lente_2"]

    # Brazo anterior: S --f1--> L1 --f1--> plano de Fourier
    sistema_anterior = encadenar_elementos([T(f1), L(f1), T(f1)])

    # Brazo posterior: plano de Fourier --f2--> L2 --f2--> sensor
    sistema_posterior = encadenar_elementos([T(f2), L(f2), T(f2)])

    # ------------------- Ventanas físicas y muestreo ---------------------
    # Ventana deseada en el plano de Fourier (se calcula por muestreo de Fresnel)
    # Usamos el sensor final como referencia de salida del brazo posterior:
    # Primero obtenemos la ventana que "debe" existir en el plano de Fourier
    # para terminar en el tamaño del sensor tras el brazo posterior.
    _, _, Lx_fourier, Ly_fourier = muestreo_entrada_para_Fresnel(
        nx, Lx_sensor, sistema_posterior.B_total, CONFIG["lambda_0"], ny, Ly_sensor
    )

    # ------------------- Propagación hasta el plano de Fourier -----------
    campo_fourier, Lx_F, Ly_F = propagar_Fresnel_ABCD(
        objeto=objeto,
        sistema=sistema_anterior,
        Lx_out=Lx_fourier,
        Ly_out=Ly_fourier,
        lambda_0=CONFIG["lambda_0"]
    )

    # ------------------- Filtro en el plano de Fourier -------------------
    xxF, yyF = malla_cartesiana(nx, Lx_F, ny, Ly_F)

    # Ventana física del diafragma (aplica “recorte” global)
    diafragma = mascara_circular(xxF, yyF, CONFIG["diam_apertura"]/2)

    # Filtro seleccionado (circular central / rendija / gaussiano)
    filtro = filtro_plano_fourier(xxF, yyF, CONFIG)

    # Filtro total (aplica límite físico + transmitancia deseada)
    H = diafragma * filtro
    campo_fourier_filtrado = campo_fourier * H

    # ------------------- Propagación hasta el sensor ---------------------
    campo_sensor, Lx_out, Ly_out = propagar_Fresnel_ABCD(
        objeto=campo_fourier_filtrado,
        sistema=sistema_posterior,
        Lx_out=Lx_sensor,
        Ly_out=Ly_sensor,
        lambda_0=CONFIG["lambda_0"]
    )

    # ------------------- Visualizaciones --------------------------------
    # 1) Objeto
    mostrar_intensidad(objeto, Lx_F, Ly_F,
                       "Objeto (máscara seleccionada)",
                       *CONFIG["clim"]["objeto"])

    # 2) Campo en plano de Fourier (antes del filtro)
    mostrar_intensidad(campo_fourier, Lx_F, Ly_F,
                       "Campo antes del diafragma (plano de Fourier)",
                       *CONFIG["clim"]["antes_diafragma"])

    # 3) Filtro (transmitancia)
    mostrar_intensidad(H, Lx_F, Ly_F,
                       "Filtro en el plano de Fourier (diafragma × transmitancia)",
                       *CONFIG["clim"]["filtro"])

    # 4) Campo tras el diafragma
    mostrar_intensidad(campo_fourier_filtrado, Lx_F, Ly_F,
                       "Campo tras el diafragma",
                       *CONFIG["clim"]["tras_diafragma"])

    # 5) Campo en el sensor
    mostrar_intensidad(campo_sensor, Lx_out, Ly_out,
                       "Campo en el sensor (salida)",
                       *CONFIG["clim"]["sensor"])

# -------------------------------------------------------------------------
if __name__ == "__main__":
    main()
