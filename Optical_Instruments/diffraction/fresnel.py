import numpy as np
from utilopctic import import_image
import cv2

class Field():
    def __init__(self, grid_zize,wavelenght, N):
        self._grid_size = grid_zize
        self._wavelenght = wavelenght
        self._N = N
        self._k = 2 * np.pi / wavelenght
        self._dx = grid_zize / N
        self._x = np.linspace(-grid_zize / 2, grid_zize / 2, N)
        self._y = np.linspace(-grid_zize / 2, grid_zize / 2, N)
        self._X, self._Y = np.meshgrid(self._x, self._y)
        self._E = np.zeros((N, N), dtype=complex)
        self._phase = np.zeros((N, N), dtype=float)
    
    def Begin(self, grid_size, wavelength, N):
        self._grid_size = grid_size
        self._wavelength = wavelength
        self._N = N
        return Field(grid_size, wavelength, N)
    def import_Intensity(self,path:None):

        if path is None:
            #llamamos a la funcion de importar imagen
            A = import_image()
        else:
            # Cargar la imagen como un arreglo numpy
            if ruta_imagen:
                #cargamos el primer 
                imagen = Image.open(ruta_imagen)
                return np.array(imagen)
            else:
                return None
        
        #redimensionamos la imagen al tamaño del campo
        A_resized = cv2.resize(A, (self._N, self._N), interpolation=cv2.INTER_CUBIC)

        

    
    




