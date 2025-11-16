import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import zoom
from typing import Tuple
import cv2


def mesh_image(image, delta_x, delta_y):
    """
    Esta funcion devulve la grilla relacionada 
    """
    rows, cols = image.shape
    x = np.arange(-cols/2, cols/2) * delta_x
    y = np.arange(-rows/2, rows/2) * delta_y
    X, Y = np.meshgrid(x, y)
    return X, Y

def plot_image(image,X,Y):
  """Esta funcion grafica la funcion con su malla real

  image:np.array imagen a graficar
  X:np.array malla real en el eje x
  Y:np.array malla real en el eje y
  """
  
  fig=plt.figure(figsize=(10,10))
  plt.imshow(image,extent=[X.min(),X.max(),Y.min(),Y.max()], cmap='gray', origin='upper')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.show()

def resize_with_pad(image:np.ndarray,new_size:Tuple[int,int],pad_value:int=255)->np.ndarray:
    """
    Redimensionamos la imagen y luego aplicamos padding.
    """
    # Redimensionar la imagen
    image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)

    # Calcular el tamaño del padding necesario
    pad_height = (new_size[0]) // 2
    pad_width = (new_size[1]) // 2
    print("pad_width,pad_height:", pad_width, pad_height)

    # Aplicar padding a la imagen
    padded_image = cv2.copyMakeBorder(image, pad_height, pad_height, pad_width, pad_width, cv2.BORDER_CONSTANT, value=pad_value)

    return padded_image
def resize_with_pad_complex(image)->np.ndarray:
   """
   Esta funcion redimensiona una imagen de datos complex
   image:funcion compleja
   new_size: nuevas dimensiones
   pad_value: color del padding
   
   return
   image_padding_complex
   """
   #extraemos la dimension de la imagen la cual se supone cuadrada
   N=image.shape[0]

   #hacemos el zoom de la imagen, esto elimina informacion de la imagen 
   imagen=zoom(image,zoom=0.5,order=3) #esta funcion 

   #luego hacemos padding
   pad=N/4
   imagen=np.pad(imagen,((4,4),(4,4)),'constant')
   imagen_padding_complex=imagen
   return imagen
   


