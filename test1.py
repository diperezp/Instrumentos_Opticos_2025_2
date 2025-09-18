import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from opticlibrary.angular_espectrum import AngularSpectrum


wavelngth = 500e-9  # Longitud de onda [m]
length_side=40e-6 #dimensiones de la imagen [m]
N=1024  # Número de pixeles

x_image = np.linspace(-length_side / 2, length_side / 2, N)
y_image = np.linspace(-length_side / 2, length_side / 2, N)
x_image, y_image = np.meshgrid(x_image, y_image)

slit_width = 5e-6  # Ancho de la rendija [m]
slit_height = 10e-6  # Alto de la rendija [m]

U0=np.where((np.abs(x_image) <= slit_width / 2) & (np.abs(y_image) <= slit_height / 2), 1, 0)


print(U0.shape[1])



asys=AngularSpectrum(U0,wavelngth,length_side)

asys.plot_magnitude_spectrum()
asys.plot_image()