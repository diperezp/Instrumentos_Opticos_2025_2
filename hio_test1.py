"""
Plantilla: Recuperación de Fase usando HIO + Espectro Angular
Autor: Diego (basado en implementación GPT)
Fecha: [coloca la tuya]
"""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v3 as iio  # pip install imageio
import os
import cv2
from tkinter import filedialog
from tkinter import Tk

# ------------------------------------------------------------------
# 1. Funciones auxiliares (FFT, propagación, restricciones, errores)
# ------------------------------------------------------------------

def fft2c(x): return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))
def ifft2c(X): return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

def asm_transfer_function(nx, ny, dx, dy, wavelength, z):
    k = 2 * np.pi / wavelength
    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(fx, fy, indexing='xy')
    arg = 1 - (wavelength * FX)**2 - (wavelength * FY)**2
    mask = arg >= 0
    H = np.zeros_like(arg, dtype=complex)
    H[mask] = np.exp(1j * k * z * np.sqrt(arg[mask]))
    return H

def asm_propagate(u0, wavelength, dx, dy, z, H_cache):
    ny, nx = u0.shape
    key = (nx, ny, dx, dy, wavelength, z)
    if key not in H_cache:
        H_cache[key] = asm_transfer_function(nx, ny, dx, dy, wavelength, z)
    H = H_cache[key]
    return ifft2c(fft2c(u0) * H)

def enforce_detector_intensity(U, I_meas_sqrt):
    return I_meas_sqrt * np.exp(1j * np.angle(U))

def object_constraints(gp, support_mask=None):
    valid = np.ones_like(gp, dtype=bool)
    if support_mask is not None:
        outside = ~support_mask
        valid[outside] = np.isclose(np.abs(gp[outside]), 0)
    return valid

def fourier_error(U, I_meas_sqrt):
    return np.linalg.norm(np.abs(U) - I_meas_sqrt) / (np.linalg.norm(I_meas_sqrt) + 1e-12)

def object_support_error(g, support_mask):
    if support_mask is None: return np.nan
    outside = ~support_mask
    return np.sqrt(np.sum(np.abs(g[outside])**2) / (np.sum(np.abs(g)**2) + 1e-12))

# ------------------------------------------------------------------
# 2. Algoritmo HIO
# ------------------------------------------------------------------

def hio_asm(I_meas, wavelength, dx, dy, z, support_mask=None, beta=0.9, n_iters=200):
    I_meas_sqrt = np.sqrt(np.maximum(I_meas, 0))
    ny, nx = I_meas.shape
    # Inicialización: fase aleatoria
    g = np.exp(1j * 2 * np.pi * np.random.rand(ny, nx)) * I_meas_sqrt
    H_cache = {}
    errF_hist, errO_hist = [], []
    for k in range(n_iters):
        # Propaga al detector
        U = asm_propagate(g, wavelength, dx, dy, z, H_cache)
        # Impone intensidad medida
        Uc = enforce_detector_intensity(U, I_meas_sqrt)
        # Propaga de vuelta
        gp = asm_propagate(Uc, wavelength, dx, dy, -z, H_cache)
        # Restricciones objeto
        valid = object_constraints(gp, support_mask)
        g_new = g.copy()
        g_new[valid] = gp[valid]
        g_new[~valid] = g[~valid] - beta * gp[~valid]
        g = g_new
        # Errores
        errF_hist.append(fourier_error(U, I_meas_sqrt))
        errO_hist.append(object_support_error(g, support_mask))
    return g, np.array(errF_hist), np.array(errO_hist)

# ------------------------------------------------------------------
# 3. PLANTILLA DE USO
# ------------------------------------------------------------------
if __name__ == "__main__":
    # === 1. Cargar imagen de intensidad medida ===
    Tk().withdraw()  # evita que aparezca la ventana principal de Tkinter
    ruta = filedialog.askopenfilename(
    title="Selecciona una imagen",
    filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
    if not os.path.exists(ruta):
        raise FileNotFoundError("No se encontró la imagen de intensidad medida.")
    I_meas=cv2.imread(ruta,cv2.IMREAD_UNCHANGED)
    if I_meas.ndim == 3:
        I_meas = np.mean(I_meas, axis=2)  # Convertir a escala de grises

    I_meas /= I_meas.max()  # Normalizar

    # === 2. Parámetros físicos ===
    wavelength = 633e-9   # 633 nm
    z_list = [0.005,0.01,0.05]  # Distancias candidatas (m)
    dx = dy = 5.4e-6       # Tamaño de píxel (ajusta al sensor real)

    # === 3. Máscara de soporte ===
    ny, nx = I_meas.shape
    Y, X = np.ogrid[:ny, :nx]
    cy, cx = ny // 2, nx // 2
    R = min(nx, ny) * 0.3
    support_mask = (X - cx)**2 + (Y - cy)**2 <= R**2

    # === 4. Barrido de distancia ===
    resultados = []
    for z in z_list:
        g_rec, errF, errO = hio_asm(I_meas, wavelength, dx, dy, z,
                                    support_mask=support_mask, beta=0.9, n_iters=300)
        resultados.append((z, g_rec, errF, errO))
        print(f"z = {z:.3f} m → Error final: {errF[-1]:.4f}")

    # === 5. Seleccionar mejor distancia ===
    mejor = min(resultados, key=lambda r: r[2][-1])
    z_best, g_best, errF_best, errO_best = mejor
    print(f"\nMejor reconstrucción: z = {z_best:.3f} m")

    # === 6. Mostrar resultados ===
    amp = np.abs(g_best)
    phase = np.angle(g_best)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0, 0].imshow(I_meas, cmap='gray')
    axs[0, 0].set_title("Intensidad medida")
    axs[0, 1].imshow(amp, cmap='gray')
    axs[0, 1].set_title("Amplitud reconstruida")
    axs[1, 0].imshow(phase, cmap='twilight')
    axs[1, 0].set_title("Fase reconstruida")
    axs[1, 1].plot(errF_best, label="Error Fourier")
    axs[1, 1].plot(errO_best, label="Error Objeto")
    axs[1, 1].legend()
    axs[1, 1].set_title("Evolución del error")
    plt.tight_layout()
    plt.show()

    # === 7. Guardar resultados ===
    iio.imwrite("amplitud_recuperada.png", (amp / amp.max() * 255).astype(np.uint8))
    iio.imwrite("fase_recuperada.png", ((phase + np.pi) / (2 * np.pi) * 255).astype(np.uint8))
    np.savez("historial_errores.npz", errF=errF_best, errO=errO_best)
