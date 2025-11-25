import numpy as np
import matplotlib.pyplot as plt

def ifft2c(F):
    return np.fft.ifftshift(np.fft.ifft2(np.fft.ifftshift(F)))

def fft2c(f):
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(f)))

def gerchberg_saxton(measured_mag, support=None, num_iters=500, enforce_positive=False, verbose=False):
    # measured_mag: array real >=0 (same shape as FT)
    M = measured_mag
    # init: random phase
    phi = np.exp(1j * 2*np.pi*np.random.rand(*M.shape))
    F_est = M * phi

    errors = []
    for k in range(num_iters):
        img = np.real(ifft2c(F_est))  # imagen espacial (se asume real)
        if support is not None:
            img = img * support
        if enforce_positive:
            img = np.clip(img, 0, None)

        F_est = fft2c(img)
        # impose measured magnitude
        F_est = M * np.exp(1j * np.angle(F_est))

        # error
        err = np.linalg.norm(np.abs(F_est) - M) / np.linalg.norm(M)
        errors.append(err)
        if verbose and (k % 50 == 0 or k == num_iters-1):
            print(f"GS iter {k+1}/{num_iters} err={err:.4e}")

    return np.real(ifft2c(F_est)), errors


def hio(measured_mag, support=None, num_iters=1000, beta=0.8, enforce_positive=False, verbose=False):
    M = measured_mag
    # init
    phi = np.exp(1j * 2*np.pi*np.random.rand(*M.shape))
    F_est = M * phi
    img = np.real(ifft2c(F_est))
    img_prev = img.copy()

    errors = []
    for k in range(num_iters):
        F_est = fft2c(img)
        # impose measured magnitude
        F_est = M * np.exp(1j * np.angle(F_est))
        img_new = np.real(ifft2c(F_est))

        # Apply HIO update
        if support is None:
            # if no support, treat whole image as support (degenerates a bit)
            support_mask = np.ones_like(img_new, dtype=bool)
        else:
            support_mask = support.astype(bool)

        # inside support: enforce constraint (e.g., realness and positivity)
        inside = support_mask
        outside = ~support_mask

        # positivity enforcement inside support
        if enforce_positive:
            # provisional projection inside support: clip negatives to zero
            proj = img_new.copy()
            proj[inside] = np.clip(proj[inside], 0, None)
        else:
            proj = img_new.copy()

        # HIO rule: outside support: g_{n+1} = g_n - beta * proj_n (or use previous)
        img_hio = img.copy()
        img_hio[inside] = proj[inside]
        img_hio[outside] = img[outside] - beta * proj[outside]

        img_prev = img
        img = img_hio

        # optional: also enforce global reality (we already take real part)
        img = np.real(img)

        # compute Fourier-domain error
        F_current = fft2c(img)
        err = np.linalg.norm(np.abs(F_current) - M) / np.linalg.norm(M)
        errors.append(err)
        if verbose and (k % 100 == 0 or k == num_iters-1):
            print(f"HIO iter {k+1}/{num_iters} err={err:.4e}")

    # final: optionally do a few GS iterations inside final support for polishing
    return img, errors


# -------------------------
# Ejemplo de uso (simulado)
# -------------------------
if __name__ == "__main__":
    # Supongamos que tienes measured_mag como numpy array (2D)
    # Para demo, generamos una imagen y su magnitud:
    from skimage.data import camera
    img_true = camera().astype(float)
    # central crop / normalize small example
    img_true = img_true / img_true.max()

    F_true = fft2c(img_true)
    measured_mag = np.abs(F_true)  # lo que tendrías (o sqrt de power spectrum)

    # construir un soporte aproximado: por ejemplo, un rectángulo central
    H, W = img_true.shape
    support = np.zeros_like(img_true)
    support[H//8: -H//8, W//8: -W//8] = 1

    # 1) GS (rápido)
    rec_gs, err_gs = gerchberg_saxton(measured_mag, support=support, num_iters=300, enforce_positive=True, verbose=True)

    # 2) HIO (mejor)
    rec_hio, err_hio = hio(measured_mag, support=support, num_iters=800, beta=0.8, enforce_positive=True, verbose=True)

    # Graficar
    plt.figure(figsize=(10,6))
    plt.subplot(2,3,1); plt.title("Original"); plt.imshow(img_true, cmap='gray'); plt.axis('off')
    plt.subplot(2,3,2); plt.title("Recon GS"); plt.imshow(rec_gs, cmap='gray'); plt.axis('off')
    plt.subplot(2,3,3); plt.title("Recon HIO"); plt.imshow(rec_hio, cmap='gray'); plt.axis('off')
    plt.subplot(2,3,4); plt.title("Error GS"); plt.plot(err_gs); plt.xlabel('iter'); plt.grid(True)
    plt.subplot(2,3,5); plt.title("Error HIO"); plt.plot(err_hio); plt.xlabel('iter'); plt.grid(True)
    plt.subplot(2,3,6); plt.title("Soporte"); plt.imshow(support, cmap='gray'); plt.axis('off')
    plt.tight_layout()
    plt.show()
