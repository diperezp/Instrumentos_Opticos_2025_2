import numpy as np
from numpy import *
import matplotlib.pyplot as plt
from LightPipes import *
from utilopctic import *

path_image=r"C:\Users\diego_0hh0fmb\OneDrive\Documents\GitHub\Instrumentos_Opticos_2025_2\Entrega03\recursos\MuestrasBio\MuestraBio_E10.csv"

def import_image_complex(path):
    """
    Esta funcion carga una imagen desde un archivo.csv dada por el path
    y la almacena en un array
    """
    print("Cargando imagen desde:", path)
    #inicialmente leemos el archivo como strings
    data_str = np.loadtxt(path, delimiter=',', dtype=str)
    print(data_str.shape)


    # Reemplazar 'i' por 'j' en toda la matriz
    data_str = np.char.replace(data_str, 'i', 'j')

    #Convertimos cada celda en un numero complejo
    data_complex = data_str.astype(np.complex128)

    return data_complex

img=import_image_complex(path_image)
#img=import_image()[:,:,0]
print(img.shape)

img_sec=np.abs(img)*np.exp(1j*np.angle(img))

print(np.max(np.abs(img)))
print(np.min(np.abs(img)))
print(np.max(np.angle(img)))
print(np.min(np.angle(img)))


#inciamos definiendo la dimensiones de la imagen

fig=plt.figure(figsize=(10, 12))
ax1= fig.add_subplot(3,2, 1)
ax1.imshow(np.abs(img), cmap='gray', origin='upper')
ax1.set_title('Intensidad')
ax1.axis('off')
ax2= fig.add_subplot(3,2, 2)
ax2.imshow(np.angle(img), cmap='gray', origin='upper')
ax2.set_title('Fase')
ax2.axis('off')
ax3= fig.add_subplot(3,2, 3)
ax3.imshow(np.real(img), cmap='gray', origin='upper')
ax3.set_title('Parte Real')
ax3.axis('off')
ax4= fig.add_subplot(3,2, 4)
ax4.imshow(np.imag(img), cmap='gray', origin='upper')
ax4.set_title('Parte Imaginaria')
ax4.axis('off')
ax5= fig.add_subplot(3,2, 5)
ax5.imshow(np.abs(img_sec), cmap='gray', origin='upper')
ax5.set_title('Reconstrucción')
ax5.axis('off')
ax6= fig.add_subplot(3,2, 6)
ax6.imshow(np.angle(img_sec), cmap='gray', origin='upper')
ax6.set_title('Fase Reconstrucción')
ax6.axis('off')
plt.show()






