import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from LightPipes import *
from utilopctic import *

path_image=r"C:\Users\luzad\OneDrive\Documentos\GitHub\Instrumentos_Opticos_2025_2\Entrega03\recursos\MuestrasBio\MuestraBio_E02.csv"

def import_image_complex(path):
    """
    Esta funcion carga una imagen desde un archivo.csv dada por el path
    y la almacena en un array
    """
    #inicialmente leemos el archivo como strings
    data_str=np.genfromtxt(path,delimiter=",",dtype=str)
    print(data_str.shape)


    # Reemplazar 'i' por 'j' en toda la matriz
    data_str = np.char.replace(data_str, 'i', 'j')

    #Convertimos cada celda en un numero complejo
    data_complex = data_str.astype(np.complex128)

    return data_complex

img=import_image_complex(path_image)
#img=import_image()[:,:,0]
print(img.shape)

#inciamos definiendo la dimensiones de la imagen

#caracteristicas de la imagen
img_width=390*um
N=img.shape[0]
pixel_size=img_width/N
#imagen
amplitud=np.abs(img)
amplitud=amplitud/amplitud.max()
angle=np.angle(img)
#caracteristicas de la iluminacion
wave_lenght=533*nm

#Caracteristicas del microscopio
f_MO=10*mm
f_TL=200*mm


#transformada de fourier
espectro=np.fft.fftshift(np.fft.fft2(img))

I=np.abs(espectro)
P=np.angle(espectro)

img1=angle*np.exp(1j*amplitud)

export_image(amplitud)
export_image(angle)




fig=plt.figure(figsize=(10,12))
axis1=fig.add_subplot(1,4,1)
axis1.imshow(amplitud,cmap='gray')
axis2=fig.add_subplot(1,4,2)
axis2.imshow(angle,cmap='gray')
axis3=fig.add_subplot(1,4,3)
axis3.imshow(np.abs(img1),cmap='gray')
axis4=fig.add_subplot(1,4,4)
axis4.imshow(np.angle(img1),cmap='gray')
plt.show()
