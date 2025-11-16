# ==============================================================
# punto2_sim.py
# Simulación difractiva (Punto 2) con formalismo ABCD + Fresnel
# - Selección de imagen (solo imágenes) vía Tkinter
# - Sistemas 4f anterior y posterior (propagación-foco)
# - Filtros en plano de Fourier: binario circular / rendija / gaussiano
# - Gráficas de todas las etapas (intensidad)
# Practica 03 - Microscopía óptica
# - en este script se actualiza el Punto 2 de la práctica 02 para la practia 03 agregando:
# - microscopio optico coherente en configuracion de conjugado infinito
# - Simulación de test USAF 1951 en microscopio coherente
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
    # Longitud de onda (enunciado)
    "lambda_0": 533e-9,          # 533 nm

    # Lentes del sistema 4f ≈ microscopio
    # MO PLAN 20x/0.5  +  TL f_TL = 200 mm
    # f_MO ≈ f_TL / M = 200 mm / 20 = 10 mm
    "f_lente_1": 10e-3,          # f_MO = 10 mm
    "f_lente_2": 200e-3,         # f_TL = 200 mm

    # Sensor Alvium 1800 U-811m (mono)
    "px_cols": 2848,
    "px_rows": 2848,
    "pitch": 2.74e-6,            # tamaño de píxel (m)

    # Tipo de filtro plano de Fourier de tu punto 2 (lo dejo igual)
    "tipo_filtro": "gaussiano_sumado_invertido",
    "sigma_gauss": 300e-6,
    "centros_gauss": [
        (1.58e-3,  1.01e-3),
        (-7.75e-4,   6.04e-4),
        (-4.67e-3,   9.8e-4),
        (-1.44e-3,  1.20e-3),
        (1.67e-3,  -9.2e-4),
        (-1.50e-3, -9.4e-4),
        (1.62e-3,  -1.15e-3),
        (4.73e-3, -1.04e-3),
    ],
    "diam_apertura": 100e-3,

    # Visualización
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

# ------------------------ Utilidades para la pregunta 2 -------------------

def usaf_frequency_lpmm(group: int, element: int) -> float:
    """
    Frecuencia espacial (line pairs/mm) de un patrón USAF 1951.
    Fórmula estándar: f = 2^{g + (e-1)/6}.
    """
    return 2.0 ** (group + (element - 1) / 6.0)


def nearest_usaf_group_element(target_lpmm: float,
                               g_min: int = -2,
                               g_max: int = 9):
    """
    Busca el par (grupo, elemento) cuya frecuencia USAF está más cerca
    de target_lpmm (en lp/mm).
    """
    best = None
    for g in range(g_min, g_max + 1):
        for e in range(1, 7):
            f = usaf_frequency_lpmm(g, e)
            diff = abs(f - target_lpmm)
            if best is None or diff < best[0]:
                best = (diff, g, e, f)
    # devuelve (grupo, elemento, frecuencia_encontrada)
    return best[1], best[2], best[3]


def simular_usaf_microscopio(lambda0: float,
                             NA: float,
                             M: float,
                             sensor_nx: int,
                             sensor_pitch: float,
                             factor_rel_f_c: float = 1.0,
                             grupo: int | None = None,
                             elemento: int | None = None,
                             N: int = 1024,
                             mostrar: bool = True):
    """
    Modelo coherente del microscopio + test USAF.

    - Si 'grupo' y 'elemento' son None, se elige el patrón USAF cuya
      frecuencia está más cerca de: factor_rel_f_c * f_c_Abbe.
      (factor_rel_f_c < 1 → por debajo de Abbe, > 1 → por encima).

    Devuelve un dict con datos de resolución y contraste.
    """
    # tamaño físico del sensor y FoV en el objeto
    sensor_L = sensor_nx * sensor_pitch          # [m]
    L_obj = sensor_L / M                         # FoV en el objeto [m]
    dx = L_obj / N                               # muestreo en el objeto

    # frecuencia de corte coherente (Abbe) y resolución teórica
    F_c = NA / lambda0                           # [ciclos/m]
    d_abbe = 1.0 / F_c                           # [m]
    F_c_mm = F_c / 1e3                           # [lp/mm]
    d_abbe_um = d_abbe * 1e6                     # [µm]

    # frecuencia objetivo (por debajo o por encima de f_c)
    target_lpmm = factor_rel_f_c * F_c_mm

    # si no se especifica grupo/elemento, se escoge el USAF más cercano a target_lpmm
    if grupo is None or elemento is None:
        grupo, elemento, f_lpmm = nearest_usaf_group_element(target_lpmm)
    else:
        f_lpmm = usaf_frequency_lpmm(grupo, elemento)

    f_test = f_lpmm * 1e3                        # [ciclos/m]
    d_test = 1.0 / f_test                        # [m]
    d_test_um = d_test * 1e6                     # [µm]

    # malla espacial en el objeto
    xi = np.linspace(-L_obj / 2, L_obj / 2, N)
    XI, ETA = np.meshgrid(xi, xi)

    # patrón sinusoidal tipo barras USAF
    U_obj = 0.5 + 0.5 * np.sin(2.0 * np.pi * f_test * XI)

    # malla de frecuencias espaciales
    fx = np.fft.fftfreq(N, d=dx)
    fy = np.fft.fftfreq(N, d=dx)
    FX, FY = np.meshgrid(fx, fy)
    FR = np.sqrt(FX**2 + FY**2)

    # función de transferencia coherente (pupila circular)
    CTF = (FR <= F_c).astype(np.complex128)
    CTF = np.fft.fftshift(CTF)

    # imagen formada (coherente)
    F_obj = np.fft.fft2(U_obj)
    F_obj_c = np.fft.fftshift(F_obj)
    F_img_c = F_obj_c * CTF
    F_img = np.fft.ifftshift(F_img_c)
    U_img = np.fft.ifft2(F_img)
    I_img = np.abs(U_img)**2

    # perfiles 1D en el centro
    perfil_obj = U_obj[N // 2, :]
    perfil_img = I_img[N // 2, :]
    coord_um = xi * 1e6

    # contraste (MTF) en una ventana central
    mid = N // 2
    ventana = slice(mid - 50, mid + 50)
    I_seg = perfil_img[ventana]
    Imax, Imin = I_seg.max(), I_seg.min()
    contraste = (Imax - Imin) / (Imax + Imin + 1e-12)

    if mostrar:
        # mapas 2D
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.imshow(U_obj, cmap="gray",
                   extent=[coord_um.min(), coord_um.max(),
                           coord_um.min(), coord_um.max()])
        plt.title(f"Objeto USAF simulado\nGrupo {grupo}, Elemento {elemento}\n"
                  f"{f_lpmm:.1f} lp/mm")
        plt.xlabel("ξ (µm)")
        plt.ylabel("η (µm)")

        plt.subplot(1, 2, 2)
        plt.imshow(I_img, cmap="gray",
                   extent=[coord_um.min(), coord_um.max(),
                           coord_um.min(), coord_um.max()])
        plt.title("Imagen simulada (microscopio coherente)")
        plt.xlabel("u (µm)")
        plt.ylabel("v (µm)")
        plt.tight_layout()
        plt.show()

        # perfiles 1D
        plt.figure()
        plt.plot(coord_um, perfil_obj, label="Objeto (amplitud)")
        plt.plot(coord_um,
                 perfil_img / np.max(perfil_img),
                 label="Imagen (intensidad)")
        plt.xlabel("x en el objeto (µm)")
        plt.ylabel("Valor normalizado")
        plt.title("Perfiles centrales – test de resolución")
        plt.legend()
        plt.grid(True)
        plt.show()

    return {
        "lambda0": lambda0,
        "NA": NA,
        "M": M,
        "F_c_mm": F_c_mm,
        "d_abbe_um": d_abbe_um,
        "factor_rel_f_c": factor_rel_f_c,
        "grupo": grupo,
        "elemento": elemento,
        "f_test_lpmm": f_lpmm,
        "d_test_um": d_test_um,
        "contraste": contraste,
    }



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
                       "Objeto",
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
    #main()

     # Parámetros del microscopio de la entrega
    NA = 0.5
    M = 20.0

    # ---------- Test 1: por DEBAJO del límite de Abbe (0.8 f_c) ----------
    print("\n==== Test 1: frecuencia por DEBAJO del límite de Abbe (0.8 f_c) ====")
    res_bajo = simular_usaf_microscopio(
        lambda0=CONFIG["lambda_0"],
        NA=NA,
        M=M,
        sensor_nx=CONFIG["px_cols"],
        sensor_pitch=CONFIG["pitch"],
        factor_rel_f_c=0.8,   # <--- 0.8 * f_c
        mostrar=True
    )

    print(f"f_c (Abbe): {res_bajo['F_c_mm']:.2f} lp/mm")
    print(f"d_Abbe: {res_bajo['d_abbe_um']:.3f} µm")
    print(f"USAF usado: Grupo {res_bajo['grupo']}, Elemento {res_bajo['elemento']}, "
          f"{res_bajo['f_test_lpmm']:.2f} lp/mm "
          f"(d ≈ {res_bajo['d_test_um']:.3f} µm)")
    print(f"Contraste (MTF) ≈ {res_bajo['contraste']:.2f}")

    # ---------- Test 2: por ENCIMA del límite de Abbe (1.2 f_c) ----------
    print("\n==== Test 2: frecuencia por ENCIMA del límite de Abbe (1.2 f_c) ====")
    res_alto = simular_usaf_microscopio(
        lambda0=CONFIG["lambda_0"],
        NA=NA,
        M=M,
        sensor_nx=CONFIG["px_cols"],
        sensor_pitch=CONFIG["pitch"],
        factor_rel_f_c=1.2,   # <--- 1.2 * f_c
        mostrar=True
    )

    print(f"f_c (Abbe): {res_alto['F_c_mm']:.2f} lp/mm")
    print(f"d_Abbe: {res_alto['d_abbe_um']:.3f} µm")
    print(f"USAF usado: Grupo {res_alto['grupo']}, Elemento {res_alto['elemento']}, "
          f"{res_alto['f_test_lpmm']:.2f} lp/mm "
          f"(d ≈ {res_alto['d_test_um']:.3f} µm)")
    print(f"Contraste (MTF) ≈ {res_alto['contraste']:.2f}")
