# ==============================================================
# punto2_microscopio_ABCD.py
#
# Práctica 3 – Microscopía Óptica
# Punto 2: Simulación difractiva de un microscopio compuesto
#
# AHORA USANDO LA FUNCIÓN DE TRANSFERENCIA COHERENTE DEL PUNTO 1:
# U_img(x,y) = FT^{-1}{ H(fx,fy) * FT{U_obj(x,y)} }
# con H(fx,fy) = pupila circular de radio f_c = NA / λ
#
# - Objeto: imagen del test de resolución USAF 1951 (cargada por el usuario)
# - Se usa el tamaño de píxel del sensor y el aumento para definir el muestreo
#   en el plano objeto.
# - Se calcula el límite de Abbe y se visualiza la imagen simulada.
# ==============================================================

from __future__ import annotations
import os
from dataclasses import dataclass
import numpy as np
import cv2
import tkinter as tk
from tkinter import filedialog
import matplotlib.pyplot as plt

# ==============================================================
# 1. PARÁMETROS FÍSICOS DEL MICROSCOPIO (Punto 2, Práctica 3)
# ==============================================================

# Longitud de onda (λ) de la iluminación coherente
LAMBDA_0 = 533e-9       # [m] 533 nm

# Objetivo PLAN 20x/0.5
M_OBJ = 20.0            # Aumento del objetivo
NA_OBJ = 0.5            # Apertura numérica

# Lente de tubo
F_TL = 200e-3           # [m] f_TL = 200 mm

# Focal del objetivo (modelo de microscopio de conjugado infinito)
F_MO = F_TL / M_OBJ     # f_MO = f_TL / M

# Diámetro efectivo de la pupila (en el plano de Fourier) para NA dada
DIAM_PUPILA = 2.0 * F_MO * NA_OBJ   # D_pupila = 2 f_MO NA

# Límite teórico de Abbe (formación de imagen coherente)
F_C_COHERENTE = NA_OBJ / LAMBDA_0   # frecuencia de corte [ciclos/m]
D_ABBE = 1.0 / F_C_COHERENTE        # resolución mínima [m]

print("--- PARÁMETROS DE SIMULACIÓN (Punto 2 – Práctica 3) ---")
print(f"Lambda: {LAMBDA_0*1e9:.1f} nm")
print(f"Focal Objetivo (f_MO): {F_MO*1e3:.2f} mm")
print(f"Focal Lente Tubo (f_TL): {F_TL*1e3:.2f} mm")
print(f"Diámetro Pupila (2·f_MO·NA): {DIAM_PUPILA*1e3:.2f} mm")
print("--- LÍMITE TEÓRICO (ABBE, coherente) ---")
print(f"Frecuencia máxima (f_c): {F_C_COHERENTE/1e3:.2f} líneas/mm")
print(f"Resolución de Abbe (d_Abbe): {D_ABBE*1e6:.3f} μm")
print("--------------------------------------------------\n")

# Configuración general usada en la simulación del Punto 3
CONFIG = {
    # Longitud de onda
    "lambda_0": LAMBDA_0,

    # Focales de objetivo y lente de tubo (se mantienen por referencia)
    "f_lente_1": F_MO,    # Objetivo
    "f_lente_2": F_TL,    # Lente de tubo

    # Sensor (dimensiones y tamaño de píxel)
    "px_cols": 2448,
    "px_rows": 2048,
    "pitch": 3.45e-6,     # [m] tamaño de píxel en el sensor

    # Tipo de filtro en el plano de Fourier adicional (a la pupila)
    "tipo_filtro": "ninguno",

    # Diámetro físico de la pupila (controlado por la NA del objetivo)
    "diam_apertura": DIAM_PUPILA,

    # Rango de intensidades (fracción del máximo) para las gráficas
    "clim": {
        "objeto": (0.0, 1.0),
        "antes_diafragma": (0.0, 1e-5),
        "filtro": (0.0, 1.0),
        "tras_diafragma": (0.0, 1e-5),
        "sensor": (0.0, 1.0),
    }
}

# ==============================================================
# 2. UTILIDADES BÁSICAS
# ==============================================================

def seleccionar_imagen_en_grises(inicio_en_escritorio: bool = True) -> np.ndarray:
    """
    Abre un diálogo para seleccionar la imagen del test de resolución (USAF 1951).
    Devuelve una matriz en escala de grises normalizada a [0,1].
    """
    print("Por favor, selecciona una imagen del Test de Resolución USAF 1951.")
    root = tk.Tk()
    root.withdraw()

    initial_dir = os.path.join(os.path.expanduser("~"), "Desktop") if inicio_en_escritorio else None
    ruta = filedialog.askopenfilename(
        title="Selecciona una imagen (PNG/JPG/JPEG/BMP/TIFF)",
        initialdir=initial_dir,
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tiff *.gif")]
    )
    root.update()
    root.destroy()

    if not ruta:
        raise RuntimeError("No se seleccionó ninguna imagen.")

    img = cv2.imread(ruta, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError("No fue posible leer la imagen seleccionada.")

    return img.astype(np.float64) / 255.0


def redimensionar_con_padding(img: np.ndarray, nuevo_ancho: int, nuevo_alto: int) -> np.ndarray:
    """
    Redimensiona la imagen manteniendo su relación de aspecto y añade bordes
    negros para que encaje exactamente en (nuevo_ancho × nuevo_alto) píxeles.
    """
    h, w = img.shape
    ratio = min(nuevo_ancho / w, nuevo_alto / h)
    nuevo_w, nuevo_h = int(w * ratio), int(h * ratio)
    img_rs = cv2.resize(img, (nuevo_w, nuevo_h), interpolation=cv2.INTER_AREA)

    pad_w = max(nuevo_ancho - nuevo_w, 0)
    pad_h = max(nuevo_alto - nuevo_h, 0)
    left, right = pad_w // 2, pad_w - pad_w // 2
    top, bottom = pad_h // 2, pad_h - pad_h // 2

    return cv2.copyMakeBorder(img_rs, top, bottom, left, right,
                              cv2.BORDER_CONSTANT, value=0)


def malla_cartesiana(n_col: int, Lx: float, n_row: int, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    """Crea una malla cartesiana centrada en 0 con tamaño físico Lx × Ly."""
    x = np.linspace(-Lx/2, Lx/2, n_col)
    y = np.linspace(-Ly/2, Ly/2, n_row)
    return np.meshgrid(x, y)


def mostrar_intensidad(campo: np.ndarray, Lx: float, Ly: float, titulo: str,
                       vmin_frac: float, vmax_frac: float):
    """
    Muestra la intensidad |campo|^2 con ejes físicos en milímetros.
    vmin_frac y vmax_frac son fracciones del máximo para escalar el contraste.
    """
    I = np.abs(campo)**2
    vmax = np.max(I) if np.max(I) > 0 else 1.0
    extent = (-Lx/2 * 1e3, Lx/2 * 1e3, -Ly/2 * 1e3, Ly/2 * 1e3)  # ejes en mm

    plt.imshow(I, extent=extent, origin='lower', cmap='gray',
               vmin=vmin_frac * vmax, vmax=vmax_frac * vmax)
    plt.colorbar(label="Intensidad (u.a.)")
    plt.xlabel("x (mm)")
    plt.ylabel("y (mm)")
    plt.title(titulo)
    plt.show()

def generar_patron_sinusoidal(nx: int, ny: int,
                              dx_obj: float, dy_obj: float,
                              f_test: float) -> np.ndarray:
    """
    Genera un patrón sinusoidal 1D en el plano objeto, variando en x
    con frecuencia espacial f_test (ciclos/m).
    Se usa para probar el comportamiento del sistema por encima de f_c.
    (Punto 2 – Práctica 3)
    """
    x = (np.arange(nx) - nx/2) * dx_obj
    y = (np.arange(ny) - ny/2) * dy_obj
    X, Y = np.meshgrid(x, y)

    # Patrón sinusoidal (amplitud entre 0 y 1)
    patron = 0.5 + 0.5 * np.sin(2 * np.pi * f_test * X)
    return patron


# ==============================================================
# 3–6. UTILIDADES ABCD / FRESNEL
# (Se mantienen, pero ya no se usan en main)
# ==============================================================

def muestreo_entrada_para_Fresnel(Nx: int, Lx_out: float, B: float, lambda_0: float,
                                  Ny: int, Ly_out: float) -> tuple[float, float, float, float]:
    dx_out, dy_out = Lx_out / Nx, Ly_out / Ny
    dx_in = (lambda_0 * B) / (Nx * dx_out)
    dy_in = (lambda_0 * B) / (Ny * dy_out)
    Lx_in, Ly_in = dx_in * Nx, dy_in * Ny
    return dx_in, dy_in, Lx_in, Ly_in


def mascara_circular(xx: np.ndarray, yy: np.ndarray, radio: float) -> np.ndarray:
    """Máscara binaria circular centrada (1 dentro del disco, 0 fuera)."""
    return ((xx**2 + yy**2) <= radio**2).astype(float)


def mascara_rendija_central(xx: np.ndarray, ancho: float) -> np.ndarray:
    """Rendija vertical centrada (1 dentro de la rendija, 0 fuera)."""
    return (np.abs(xx) <= (ancho/2)).astype(float)


def filtro_plano_fourier(xx: np.ndarray, yy: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Filtro adicional en el plano de Fourier.
    En este punto 3 se deja como 'ninguno' (no se usa, pero se mantiene).
    """
    tipo = cfg["tipo_filtro"]

    if tipo == "ninguno":
        return 1.0

    if tipo == "circular":
        radio = cfg.get("radio_pasa_centro", 1e-3)
        return mascara_circular(xx, yy, radio)

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
        gsum = np.zeros_like(xx, dtype=float)
        inv_2sig2 = 1.0 / (2.0 * sigma**2)
        for (x0, y0) in centros:
            gsum += np.exp(-(((xx - x0)**2 + (yy - y0)**2) * inv_2sig2))
        filtro = 1.0 - gsum
        return np.clip(filtro, 0.0, 1.0)

    raise ValueError("tipo_filtro no reconocido.")


@dataclass
class SistemaABCD:
    M: np.ndarray
    B_total: float
    camino_optico: float


def T(d: float, n: float = 1.0) -> np.ndarray:
    return np.array([[1.0, d/n],
                     [0.0, 1.0]], dtype=float)


def L(f: float) -> np.ndarray:
    return np.array([[1.0, 0.0],
                     [-1.0/f, 1.0]], dtype=float)


def encadenar_elementos(elementos: list[np.ndarray]) -> SistemaABCD:
    Msys = np.eye(2)
    B_total = 0.0
    for E in elementos:
        Msys = Msys @ E
        B_total += E[0, 1]
    return SistemaABCD(M=Msys, B_total=B_total, camino_optico=B_total)


def propagar_Fresnel_ABCD(objeto: np.ndarray, sistema: SistemaABCD,
                          Lx_out: float, Ly_out: float, lambda_0: float) -> tuple[np.ndarray, float, float]:
    Ny, Nx = objeto.shape
    A, B, C, D = sistema.M[0, 0], sistema.M[0, 1], sistema.M[1, 0], sistema.M[1, 1]
    k = 2 * np.pi / lambda_0

    dx_out, dy_out = Lx_out / Nx, Ly_out / Ny
    dx_in, dy_in, Lx_in, Ly_in = muestreo_entrada_para_Fresnel(
        Nx, Lx_out, B, lambda_0, Ny, Ly_out
    )

    xx_in, yy_in = malla_cartesiana(Nx, Lx_in, Ny, Ly_in)
    xx_out, yy_out = malla_cartesiana(Nx, Lx_out, Ny, Ly_out)

    fase_const = np.exp(1j * k * sistema.camino_optico)
    fase_in = np.exp(1j * k * (A/(2*B)) * (xx_in**2 + yy_in**2))
    fase_out = np.exp(1j * k * (D/(2*B)) * (xx_out**2 + yy_out**2))

    campo_out = (
        fase_const * fase_out *
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(objeto * fase_in))) *
        (dx_in * dy_in) * (1 / (1j * lambda_0 * B))
    )

    return campo_out, Lx_out, Ly_out

# ==============================================================
# 7. RUTINA PRINCIPAL USANDO LA FUNCIÓN DE TRANSFERENCIA (Punto 1)
# ==============================================================

def main():
    # 7.1 Parámetros del sensor y del plano objeto
    nx, ny = CONFIG["px_cols"], CONFIG["px_rows"]
    pitch = CONFIG["pitch"]
    lambda_0 = CONFIG["lambda_0"]

    # Tamaño efectivo de píxel en el plano objeto (aumento M_OBJ)
    dx_obj = pitch / M_OBJ
    dy_obj = pitch / M_OBJ
    Lx_obj = nx * dx_obj
    Ly_obj = ny * dy_obj

    # 7.2 Selección y preparación del objeto (Test USAF 1951)
    objeto = seleccionar_imagen_en_grises(inicio_en_escritorio=True)
    objeto = redimensionar_con_padding(objeto, nx, ny)

    # 7.3 Malla en el plano objeto (solo para ejes físicos de la figura)
    xx_obj, yy_obj = malla_cartesiana(nx, Lx_obj, ny, Ly_obj)

    # 7.4 Malla de frecuencias espaciales en el plano objeto
    #     (fx, fy en ciclos/m)
    fx = np.fft.fftfreq(nx, d=dx_obj)
    fy = np.fft.fftfreq(ny, d=dy_obj)
    FX, FY = np.meshgrid(fx, fy)

    # 7.5 FUNCIÓN DE TRANSFERENCIA COHERENTE DEL PUNTO 1
    #     H(fx,fy) = 1 dentro del círculo |f| <= NA/λ, 0 fuera.
    f_c = F_C_COHERENTE  # frecuencia de corte (NA/λ)
    H = (np.sqrt(FX**2 + FY**2) <= f_c).astype(np.complex128)

    # 7.6 FORMACIÓN DE IMAGEN: U_img = FT^{-1}{ H * FT{objeto} }
    O = np.fft.fft2(objeto)
    E_img = np.fft.ifft2(O * H)

    # 7.7 Visualización
    mostrar_intensidad(
        objeto, Lx_obj, Ly_obj,
        "Objeto de entrada (Test USAF 1951, plano objeto)",
        *CONFIG["clim"]["objeto"]
    )

    # Para mostrar H como mapa, usamos ejes en frecuencia (ciclos/mm)
    H_amp = np.abs(H)
    fx_mm = fx / 1e3  # ciclos/mm
    fy_mm = fy / 1e3
    FX_mm, FY_mm = np.meshgrid(fx_mm, fy_mm)
    extent_f = (fx_mm.min(), fx_mm.max(), fy_mm.min(), fy_mm.max())

    plt.imshow(H_amp, extent=extent_f, origin='lower', cmap='gray')
    plt.colorbar(label="|H(fx,fy)|")
    plt.xlabel("f_x (ciclos/mm)")
    plt.ylabel("f_y (ciclos/mm)")
    plt.title(f"Función de transferencia coherente\nCorte: f_c = {F_C_COHERENTE/1e3:.2f} líneas/mm")
    plt.show()

    # En el plano imagen, el campo tiene el mismo tamaño de matriz,
    # pero físicamente las coordenadas están ampliadas por M_OBJ.
    Lx_img = Lx_obj * M_OBJ
    Ly_img = Ly_obj * M_OBJ

    mostrar_intensidad(
        E_img, Lx_img, Ly_img,
        f"Imagen en el plano imagen (Punto 2)\nLímite de Abbe teórico: {D_ABBE*1e6:.3f} μm",
        *CONFIG["clim"]["sensor"]
    )

        # --------------------------------------------------------------
    # 7.8 Prueba sintética: patrón sinusoidal por encima de f_c
    #     (Punto 2 – Práctica 3)
    # --------------------------------------------------------------
    # Elegimos una frecuencia de prueba mayor que el límite de Abbe:
    factor_sobre_fc = 1.2  # 20% por encima de f_c
    f_test = factor_sobre_fc * f_c  # [ciclos/m]

    print("\n--- Prueba sintética (patrón sinusoidal) ---")
    print(f"f_c (teórico)      = {f_c/1e3:.2f} líneas/mm")
    print(f"f_test (>{' ' if factor_sobre_fc>=1 else ''}f_c) = {f_test/1e3:.2f} líneas/mm\n")

    # Patrón en el plano objeto (resolución espacial 'ideal')
    patron_sup = generar_patron_sinusoidal(nx, ny, dx_obj, dy_obj, f_test)

    mostrar_intensidad(
        patron_sup, Lx_obj, Ly_obj,
        f"Patrón sinusoidal en el objeto\n"
        f"f_test = {f_test/1e3:.2f} líneas/mm (> f_c)",
        *CONFIG["clim"]["objeto"]
    )

    # Formación de imagen a través del microscopio (mismo H)
    O_sup = np.fft.fft2(patron_sup)
    E_img_sup = np.fft.ifft2(O_sup * H)

    mostrar_intensidad(
        E_img_sup, Lx_img, Ly_img,
        "Imagen del patrón sinusoidal filtrada por la CTF\n"
        "(f_test > f_c: se espera pérdida casi total de modulación)",
        *CONFIG["clim"]["sensor"]
    )


# ==============================================================
# 8. EJECUCIÓN DEL PUNTO 2
# ==============================================================

if __name__ == "__main__":
    main()
