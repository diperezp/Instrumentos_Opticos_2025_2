# ==============================================================
# punto3_microscopio_ABCD.py
#
# Práctica 3 – Microscopía Óptica
# Punto 3: Simulación difractiva de un microscopio compuesto
#          usando formalismo ABCD + Fresnel
#
# - Modelo coherente de un microscopio 20x/0.5 con lente de tubo de 200 mm
# - Objeto: imagen del test de resolución USAF 1951 (cargada por el usuario)
# - Sistema equivalente 4f (objetivo + lente de tubo)
# - Pupila circular que emula la NA del objetivo (NA = 0.5)
# - Cálculo del límite de Abbe y visualización de la imagen simulada
#
# *** CÓDIGO ESPECÍFICO DEL PUNTO 3 DE LA PRÁCTICA 3 ***
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
# 1. PARÁMETROS FÍSICOS DEL MICROSCOPIO (Punto 3, Práctica 3)
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

print("--- PARÁMETROS DE SIMULACIÓN (Punto 3 – Práctica 3) ---")
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

    # Focales de objetivo y lente de tubo
    "f_lente_1": F_MO,    # Objetivo
    "f_lente_2": F_TL,    # Lente de tubo

    # Sensor (dimensiones y tamaño de píxel)
    # Aquí se modela el sensor físico de la cámara.
    "px_cols": 2448,
    "px_rows": 2048,
    "pitch": 3.45e-6,     # [m] tamaño de píxel

    # Tipo de filtro en el plano de Fourier
    # Para el Punto 3, solo se aplica la pupila circular (NA),
    # por eso el filtro adicional es "ninguno".
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
#    (Selección de imagen USAF, redimensionado, malla y ploteo)
# ==============================================================

def seleccionar_imagen_en_grises(inicio_en_escritorio: bool = True) -> np.ndarray:
    """
    Punto 3 – Práctica 3:
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

# ==============================================================
# 3. MUESTREO DE FRESNEL (ABCD) – Punto 3
# ==============================================================

def muestreo_entrada_para_Fresnel(Nx: int, Lx_out: float, B: float, lambda_0: float,
                                  Ny: int, Ly_out: float) -> tuple[float, float, float, float]:
    """
    Calcula el muestreo requerido en el plano de ENTRADA dado el tamaño del
    plano de SALIDA para una propagación tipo Fresnel asociada al término B
    de la matriz ABCD (Punto 3, Práctica 3).
    """
    dx_out, dy_out = Lx_out / Nx, Ly_out / Ny
    dx_in = (lambda_0 * B) / (Nx * dx_out)
    dy_in = (lambda_0 * B) / (Ny * dy_out)
    Lx_in, Ly_in = dx_in * Nx, dy_in * Ny
    return dx_in, dy_in, Lx_in, Ly_in

# ==============================================================
# 4. FILTROS EN EL PLANO DE FOURIER (Pupila del objetivo)
# ==============================================================

def mascara_circular(xx: np.ndarray, yy: np.ndarray, radio: float) -> np.ndarray:
    """Máscara binaria circular centrada (1 dentro del disco, 0 fuera)."""
    return ((xx**2 + yy**2) <= radio**2).astype(float)


def mascara_rendija_central(xx: np.ndarray, ancho: float) -> np.ndarray:
    """Rendija vertical centrada (1 dentro de la rendija, 0 fuera)."""
    return (np.abs(xx) <= (ancho/2)).astype(float)


def filtro_plano_fourier(xx: np.ndarray, yy: np.ndarray, cfg: dict) -> np.ndarray:
    """
    Devuelve el filtro en el plano de Fourier según CONFIG['tipo_filtro'].
    En el Punto 3 se usa 'ninguno' y la pupila real está dada por 'diam_apertura'.
    """
    tipo = cfg["tipo_filtro"]

    if tipo == "ninguno":
        # Filtro de paso total; la limitación está dada solo por el diafragma circular.
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

# ==============================================================
# 5. FORMALISMO ABCD (Matrices del sistema) – Punto 3
# ==============================================================

@dataclass
class SistemaABCD:
    """
    Contenedor para una cadena de elementos ABCD y metadatos.
    Usado en el Punto 3 para describir los tramos:
    Objeto → Pupila y Pupila → Sensor.
    """
    M: np.ndarray           # Matriz 2x2 del sistema completo
    B_total: float          # Término B neto (m)
    camino_optico: float    # z efectivo para la fase global


def T(d: float, n: float = 1.0) -> np.ndarray:
    """Traslación en un medio de índice n: matriz [[1, d/n],[0,1]]."""
    return np.array([[1.0, d/n],
                     [0.0, 1.0]], dtype=float)


def L(f: float) -> np.ndarray:
    """Lente delgada de focal f: matriz [[1,0],[-1/f,1]]."""
    return np.array([[1.0, 0.0],
                     [-1.0/f, 1.0]], dtype=float)


def encadenar_elementos(elementos: list[np.ndarray]) -> SistemaABCD:
    """
    Multiplica las matrices ABCD en orden de propagación y acumula el término B.
    Se usa en la simulación del Punto 3 para construir los brazos anterior y posterior.
    """
    Msys = np.eye(2)
    B_total = 0.0
    for E in elementos:
        Msys = Msys @ E
        B_total += E[0, 1]
    return SistemaABCD(M=Msys, B_total=B_total, camino_optico=B_total)

# ==============================================================
# 6. PROPAGACIÓN DE FRESNEL USANDO ABCD – Punto 3
# ==============================================================

def propagar_Fresnel_ABCD(objeto: np.ndarray, sistema: SistemaABCD,
                          Lx_out: float, Ly_out: float, lambda_0: float) -> tuple[np.ndarray, float, float]:
    """
    Propagación escalar de Fresnel usando la forma de Collins (ABCD) discreta.
    Devuelve el campo en el plano de salida y los tamaños físicos Lx_out, Ly_out.
    Esta rutina se usa tanto para Objeto→Pupila como para Pupila→Sensor.
    """
    Ny, Nx = objeto.shape
    A, B, C, D = sistema.M[0, 0], sistema.M[0, 1], sistema.M[1, 0], sistema.M[1, 1]
    k = 2 * np.pi / lambda_0

    # Muestreo requerido en ENTRADA para obtener la ventana Lx_out × Ly_out en SALIDA.
    dx_in, dy_in, Lx_in, Ly_in = muestreo_entrada_para_Fresnel(
        Nx, Lx_out, B, lambda_0, Ny, Ly_out
    )

    # Mallas de entrada y salida
    xx_in, yy_in = malla_cartesiana(Nx, Lx_in, Ny, Ly_in)
    xx_out, yy_out = malla_cartesiana(Nx, Lx_out, Ny, Ly_out)

    # Fases cuadráticas asociadas al sistema ABCD
    fase_const = np.exp(1j * k * sistema.camino_optico)
    fase_in = np.exp(1j * k * (A/(2*B)) * (xx_in**2 + yy_in**2))
    fase_out = np.exp(1j * k * (D/(2*B)) * (xx_out**2 + yy_out**2))

    # Transformada de Fresnel (implementada con FFT2)
    campo_out = (
        fase_const * fase_out *
        np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(objeto * fase_in))) *
        (dx_in * dy_in) * (1 / (1j * lambda_0 * B))
    )

    return campo_out, Lx_out, Ly_out

# ==============================================================
# 7. RUTINA PRINCIPAL (Punto 3 – Formación de imagen USAF)
# ==============================================================

def main():
    # 7.1 Parámetros físicos del sensor
    nx, ny = CONFIG["px_cols"], CONFIG["px_rows"]
    pitch = CONFIG["pitch"]
    Lx_sensor, Ly_sensor = nx * pitch, ny * pitch  # tamaño físico del sensor

    # 7.2 Selección y preparación del objeto (Test USAF 1951)
    objeto = seleccionar_imagen_en_grises(inicio_en_escritorio=True)
    objeto = redimensionar_con_padding(objeto, nx, ny)

    # 7.3 Definición de los sistemas 4f (objetivo + lente de tubo)
    f1, f2 = CONFIG["f_lente_1"], CONFIG["f_lente_2"]

    # Brazo anterior: Objeto --f1--> L_MO --f1--> plano de Fourier/pupila
    sistema_anterior = encadenar_elementos([T(f1), L(f1), T(f1)])

    # Brazo posterior: plano de Fourier --f2--> L_TL --f2--> sensor
    sistema_posterior = encadenar_elementos([T(f2), L(f2), T(f2)])

    # 7.4 Ventanas físicas y muestreo del plano de Fourier
    _, _, Lx_fourier, Ly_fourier = muestreo_entrada_para_Fresnel(
        nx, Lx_sensor, sistema_posterior.B_total, CONFIG["lambda_0"], ny, Ly_sensor
    )

    # 7.5 Propagación Objeto → Plano de Fourier
    print("Propagando Objeto -> Plano de Fourier (Punto 3)...")
    campo_fourier, Lx_F, Ly_F = propagar_Fresnel_ABCD(
        objeto=objeto,
        sistema=sistema_anterior,
        Lx_out=Lx_fourier,
        Ly_out=Ly_fourier,
        lambda_0=CONFIG["lambda_0"]
    )
    print("Propagación 1/2 completa.\n")

    # 7.6 Aplicación de la pupila (NA) en el plano de Fourier
    xxF, yyF = malla_cartesiana(nx, Lx_F, ny, Ly_F)

    # Pupila circular del objetivo (NA = 0.5)
    diafragma = mascara_circular(xxF, yyF, CONFIG["diam_apertura"] / 2)

    # Filtro adicional (aquí: "ninguno" → todo pasa)
    filtro = filtro_plano_fourier(xxF, yyF, CONFIG)

    # Filtro total en el plano de Fourier
    H = diafragma * filtro
    campo_fourier_filtrado = campo_fourier * H

    # 7.7 Propagación Plano de Fourier → Sensor
    print("Propagando Plano de Fourier -> Sensor (Punto 3)...")
    campo_sensor, Lx_out, Ly_out = propagar_Fresnel_ABCD(
        objeto=campo_fourier_filtrado,
        sistema=sistema_posterior,
        Lx_out=Lx_sensor,
        Ly_out=Ly_sensor,
        lambda_0=CONFIG["lambda_0"]
    )
    print("Propagación 2/2 completa.\n")

    # 7.8 Visualización de los diferentes planos
    mostrar_intensidad(
        objeto, Lx_F, Ly_F,
        "Objeto de entrada (Test USAF 1951)",
        *CONFIG["clim"]["objeto"]
    )

    mostrar_intensidad(
        campo_fourier, Lx_F, Ly_F,
        "Campo antes de la pupila (plano de Fourier)",
        *CONFIG["clim"]["antes_diafragma"]
    )

    mostrar_intensidad(
        H, Lx_F, Ly_F,
        f"Pupila del objetivo (NA = 0.5, D = {DIAM_PUPILA*1e3:.2f} mm)",
        *CONFIG["clim"]["filtro"]
    )

    mostrar_intensidad(
        campo_fourier_filtrado, Lx_F, Ly_F,
        "Campo tras la pupila",
        *CONFIG["clim"]["tras_diafragma"]
    )

    mostrar_intensidad(
        campo_sensor, Lx_out, Ly_out,
        f"Imagen en el sensor (Punto 3)\nLímite de Abbe teórico: {D_ABBE*1e6:.3f} μm",
        *CONFIG["clim"]["sensor"]
    )

# ==============================================================
# 8. EJECUCIÓN DEL PUNTO 3
# ==============================================================

if __name__ == "__main__":
    # Llamada principal para el Punto 3 de la Práctica 3.
    main()
