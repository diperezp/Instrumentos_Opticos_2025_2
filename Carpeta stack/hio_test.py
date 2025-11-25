import numpy as np
import matplotlib.pyplot as plt
from optic_util import *

# ---------- Utilidades FFT centradas ----------
def fft2c(x):
    """FFT2 centrada (shift antes y después)"""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(x)))

def ifft2c(X):
    """IFFT2 centrada (shift antes y después)"""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(X)))

# ---------- Propagación por Espectro Angular ----------
def asm_transfer_function(nx, ny, dx, dy, wavelength, z, evanescent_cut=True):
    """
    Devuelve H(fx, fy, z) para ASM.
    evanescent_cut=True anula los componentes evanescentes (estabiliza).
    """
    k = 2 * np.pi / wavelength

    fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # [1/m]
    fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))
    FX, FY = np.meshgrid(fx, fy, indexing='xy')

    # Termino bajo la raíz: 1 - (λ fx)^2 - (λ fy)^2
    arg = 1.0 - (wavelength * FX)**2 - (wavelength * FY)**2

    # Manejo de evanescentes
    if evanescent_cut:
        # Para arg < 0, apagamos el término (no propagamos evanescentes)
        mask_prop = arg >= 0
        H = np.zeros_like(arg, dtype=np.complex128)
        H[mask_prop] = np.exp(1j * k * z * np.sqrt(arg[mask_prop]))
    else:
        # Permitimos valores complejos (crecimiento/atenuación evanescente)
        H = np.exp(1j * k * z * np.sqrt(arg + 0j))

    return H

def asm_propagate(u0, wavelength, dx, dy, z, H_cache=None, evanescent_cut=True):
    """
    Propaga u0 una distancia z usando ASM.
    - H_cache: puedes pasar un dict para cachear H por (z) y ganar velocidad.
    """
    ny, nx = u0.shape  # Nota: shape es (rows, cols) -> (y, x)
    key = (nx, ny, dx, dy, wavelength, z, evanescent_cut)

    if H_cache is not None and key in H_cache:
        H = H_cache[key]
    else:
        H = asm_transfer_function(nx, ny, dx, dy, wavelength, z, evanescent_cut)
        if H_cache is not None:
            H_cache[key] = H

    U0 = fft2c(u0)
    Uz = ifft2c(U0 * H)
    return Uz

# ---------- Proyección en el dominio del detector ----------
def enforce_detector_intensity(U, I_meas_sqrt):
    """
    Sustituye la amplitud por la medida, conserva fase.
    I_meas_sqrt = sqrt(I_medida)
    """
    phase = np.exp(1j * np.angle(U))
    return I_meas_sqrt * phase

# ---------- Comprobación de restricciones en el objeto ----------
def object_constraints(gp, support_mask=None, enforce_nonneg=False):
    """
    Devuelve:
      valid_mask: puntos que CUMPLEN las restricciones (True/False)
      gp_proj: proyección de gp a las restricciones (para ER, si la quisieras)
    Restricciones:
      - support_mask: fuera del soporte -> debe ser 0
      - enforce_nonneg: Re(g) >= 0 (útil para objetos puramente absorbentes)
    """
    gp_proj = gp.copy()

    valid_mask = np.ones_like(gp, dtype=bool)

    if support_mask is not None:
        # Fuera del soporte: debe ser 0
        outside = ~support_mask
        gp_proj[outside] = 0.0 + 0.0j
        # Puntos fuera del soporte NO cumplen (a menos que ya sean ~0)
        # Consideramos violación si |gp| > 0 fuera
        valid_mask[outside] = np.isclose(np.abs(gp[outside]), 0.0)

    if enforce_nonneg:
        # No negatividad sobre la parte real (típico para objetos de amplitud)
        neg = np.real(gp_proj) < 0
        gp_proj[neg] = 0.0 + 1j * np.imag(gp_proj[neg])
        valid_mask[neg] = False

    return valid_mask, gp_proj

# ---------- Métricas de error ----------
def fourier_error(U, I_meas_sqrt):
    """Error relativo de amplitud en detector."""
    num = np.linalg.norm(np.abs(U) - I_meas_sqrt)
    den = np.linalg.norm(I_meas_sqrt) + 1e-12
    return num / den

def object_support_error(g, support_mask):
    """Fracción de energía fuera del soporte."""
    if support_mask is None:
        return np.nan
    outside = ~support_mask
    num = np.sum(np.abs(g[outside])**2)
    den = np.sum(np.abs(g)**2) + 1e-12
    return np.sqrt(num / den)

# ---------- Algoritmo HIO con ASM ----------
def hio_asm(
    I_meas,                  # Intensidad medida en el detector (2D, float)
    wavelength,              # [m]
    dx, dy,                  # paso espacial en el objeto [m]
    z,                       # distancia objeto->detector [m]
    support_mask=None,       # máscara booleana Ω; True adentro
    beta=0.9,                # parámetro HIO (0.5-1.0 típico)
    n_iters=300,
    init_obj=None,           # condición inicial en el objeto (2D complejo)
    enforce_nonneg=False,    # aplicar no-negatividad en Re(g)
    evanescent_cut=True,     # cortar evanescentes en ASM
    callback=None            # función opcional: callback(k, g, errF, errO)
):
    """
    Devuelve:
      g: objeto reconstruido (complejo)
      history: dict con 'errF', 'errO' por iteración
    """
    I_meas = np.asarray(I_meas, dtype=np.float64)
    I_meas_sqrt = np.sqrt(np.maximum(I_meas, 0.0))

    ny, nx = I_meas.shape

    # Inicialización del objeto
    if init_obj is None:
        # amplitud inicial uniforme dentro del soporte, fase aleatoria
        amp0 = np.ones_like(I_meas_sqrt)
        phase0 = np.exp(1j * 2 * np.pi * np.random.rand(ny, nx))
        g = amp0 * phase0
        if support_mask is not None:
            g = g * support_mask.astype(g.dtype)
    else:
        g = init_obj.astype(np.complex128)

    # Cache para H
    H_cache = {}

    errF_hist, errO_hist = [], []

    for k in range(1, n_iters + 1):
        # 1) Propaga al detector
        U = asm_propagate(g, wavelength, dx, dy, z, H_cache, evanescent_cut)

        # 2) Impone amplitud medida
        Uc = enforce_detector_intensity(U, I_meas_sqrt)

        # 3) Propaga de vuelta al objeto
        gp = asm_propagate(Uc, wavelength, dx, dy, -z, H_cache, evanescent_cut)

        # 4) Restricciones en el objeto + regla HIO
        valid_mask, _ = object_constraints(gp, support_mask, enforce_nonneg)

        g_new = g.copy()
        # Donde CUMPLE → aceptar g'
        g_new[valid_mask] = gp[valid_mask]
        # Donde VIOLA → feedback HIO
        g_new[~valid_mask] = g[~valid_mask] - beta * gp[~valid_mask]

        g = g_new

        # 5) Métricas
        errF = fourier_error(U, I_meas_sqrt)
        errO = object_support_error(g, support_mask)
        errF_hist.append(errF)
        errO_hist.append(errO)

        if callback is not None:
            callback(k, g, errF, errO)

    history = {"errF": np.array(errF_hist), "errO": np.array(errO_hist)}
    return g, history

# ---------- Ejemplo de uso (plantilla) ----------
if __name__ == "__main__":
    # Parámetros (ajústalos a tu experimento)
    wavelength = 633e-9        # 633 nm
    z = 10e-2                   # 5 cm (ejemplo)
    ny, nx = 512, 512
    dx = dy = 5.8e-3      # paso de píxel (p.ej. 6.5 µm)

    # Carga o genera I_meas (2D). Aquí solo un placeholder:
    # I_meas = ...  # Debe provenir de tu sensor (normalizado o en unidades arbitrarias)
    I_meas = import_image(512)  # <--- Reemplaza por tus datos reales
    print(I_meas.shape)

    # Soporte: por ejemplo, un círculo central (ajusta según tu óptica)

    # Ejecuta HIO
    g_rec, hist = hio_asm(
        I_meas=I_meas,
        wavelength=wavelength,
        dx=dx, dy=dy,
        z=z,
        support_mask=None,
        beta=0.9,
        n_iters=100,
        init_obj=None,
        enforce_nonneg=False,  # pon True si tu objeto es puramente absorbente
        evanescent_cut=True,
        callback=lambda k, g, eF, eO: (None)  # puedes imprimir o registrar si deseas
    )

    # g_rec es complejo: |g_rec| = amplitud, np.angle(g_rec) = fase
    # hist['errF'], hist['errO'] contienen la evolución de los errores

    plot_image(g_rec)
