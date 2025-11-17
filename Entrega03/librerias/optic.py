import numpy as np
import matplotlib.pyplot as plt
from LightPipes import *

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


#inciamos definiendo la dimensiones de la imagen

#caracteristicas de la imagen
img_width=390*um
N=img.shape[0]
pixel_size=img_width/N
#caracteristicas de la iluminacion
wave_lenght=533*nm

#Caracteristicas del microscopio
f_MO=10*mm
f_TL=200*mm

#iniciamos la simulacion del microscopio

U=Begin(pixel_size,wave_lenght,N)

U=MultIntensity(U,np.abs(img))
U=MultPhase(U,np.angle(img))

U=Fresnel(U,f_MO)
U=Lens(U,f_MO)
U=Fresnel(U,f_MO)
U=Fresnel(U,f_TL)
U=Lens(U,f_TL)
U=Fresnel(U,f_TL)


I=Intensity(U)





fig=plt.figure(figsize=(10,12))
axis1=fig.add_subplot(1,3,1)
axis1.imshow(np.abs(img),cmap='gray')
axis2=fig.add_subplot(1,3,2)
axis2.imshow(np.angle(img))
axis3=fig.add_subplot(1,3,3)
axis3.imshow(I)
plt.show()
