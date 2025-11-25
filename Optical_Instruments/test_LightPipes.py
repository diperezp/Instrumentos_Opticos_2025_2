from LightPipes import *

#parameters de illuminacion
wavelength = 633*nm


#parametros de la imagen
size = 5*cm
N = 500
F = 20*cm

#creacion del campo
U = Begin(size, wavelength, N)

print(type(U))