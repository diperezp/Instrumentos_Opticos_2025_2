import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from opticlibrary.angular_espectrum import AngularSpectrum
from optic_util import *
from backpropagation_angular_espectrum import BackpropagationAngularSpectrum


wavelngth = 633e-9  # Longitud de onda [m]
length_side=5.8e-3 #dimensiones de la imagen [m]
N=1080  # Número de pixeles

x_image = np.linspace(-length_side / 2, length_side / 2, N)
y_image = np.linspace(-length_side / 2, length_side / 2, N)
x_image, y_image = np.meshgrid(x_image, y_image)

slit_width = 5e-6  # Ancho de la rendija [m]
slit_height = 10e-6  # Alto de la rendija [m]

U0=np.where((np.abs(x_image) <= slit_width / 2) & (np.abs(y_image) <= slit_height / 2), 1, 0)
asys=AngularSpectrum(U0,wavelngth,length_side)

#Instanciamos la clase de espectro angular
asys.import_image(512)  #seleccionamos la transmittancia
asys.plot_image()   #mostramos la transmittancia




img=asys.get_propagation(5e-2) #propagamos a 5 cm
# print(img.shape)
# #mostramos la imagen propagada
asys.plot_propagation(5e-2,True)
plot_image(np.abs(img), title='Campo propagado a 5 mm', cmap_type='gray')



#extraemos el espectro de magnitud y fase
esperctrum_magnitude=np.abs(img)
print(f"Dimensión del espectro de magnitud: {esperctrum_magnitude.shape}")
plot_image(esperctrum_magnitude, title='Espectro de magnitud', cmap_type='gray')
esperctrum_phase=np.angle(img)
print(f"Dimensión del espectro de fase: {esperctrum_phase.shape}")
plot_image(esperctrum_phase, title='Espectro de fase', cmap_type='gray')

angular_spectrum=esperctrum_magnitude*np.exp(1j*esperctrum_phase)



#instanciamos la clase de backpropagation
bpas=BackpropagationAngularSpectrum(angular_spectrum,5e-2,length_side,wavelngth)


#calculamos la transmitancia
transmittance=bpas.compute_transmittance(True)
print(f"Dimensión de la transmitancia: {transmittance.shape}")
#mostramos la transmitancia
plot_image(np.abs(transmittance), title='Transmitancia reconstruida', cmap_type='gray')
plot_image(np.angle(transmittance),title="Espectro de phase reconstruida", cmap_type='gray')


