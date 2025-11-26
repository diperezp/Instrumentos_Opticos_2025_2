import numpy as np
import matplotlib.pyplot as plt
from diffraction.utilopctic import import_image
import cv2
from PIL import Image



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
        self.__E = None  # Campo eléctrico inicializado en cero
    
    def Begin(self, grid_size, wavelength, N):
        self._grid_size = grid_size
        self._wavelength = wavelength
        self._N = N
        return Field(grid_size, wavelength, N)
    def import_Intensity(self,path:None):

        if path is None:
            #llamamos a la funcion de importar imagen
            imagen = import_image()
        else:
            # Cargar la imagen como un arreglo numpy
            if path:
                #cargamos el primer 
                imagen = Image.open(path)
                imagen = np.array(imagen)
            else:
                return None
        #redimensionamos la imagen al tamaño del campo sin utilizar cv2
        A_resized = cv2.resize(imagen, (self._N, self._N), interpolation=cv2.INTER_CUBIC)
        #cambio a float32
        A_resized = A_resized.astype(np.float32)
        self.__E = A_resized * np.exp(1j * 0)

    def import_Phase(self,path:None):

        if path is None:
            #llamamos a la funcion de importar imagen
            imagen = import_image()
        else:
            # Cargar la imagen como un arreglo numpy
            if path:
                #cargamos el primer 
                imagen = Image.open(path)
                imagen = np.array(imagen)
            else:
                return None
        
        #redimensionamos la imagen al tamaño del campo sin utilizar cv2
        imagen_resized = cv2.resize(imagen, (self._N, self._N), interpolation=cv2.INTER_CUBIC)
        #normalizamoos la fase entre 0 y 2pi
        imagen_resized = (imagen_resized/255)* 2 * np.pi
        self.__E=self.__E*np.exp(1j * imagen_resized)

    def padding2N_field(self):
        """
        Agrega padding al campo eléctrico para alcanzar un nuevo tamaño físico.
        
        Parámetros:
            new_size : nuevo tamaño físico (en metros)
        """
        new_size = 2 * self._grid_size
        N_new = int(new_size / self._dx)
        pad_x = (N_new - self._N) // 2
        pad_y = (N_new - self._N) // 2
        
        E_padded = np.pad(self.__E, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant', constant_values=0)
        
        self.__E = E_padded
        self._N = N_new
        self._grid_size = new_size
        self._x = np.linspace(-new_size / 2, new_size / 2, N_new)
        self._y = np.linspace(-new_size / 2, new_size / 2, N_new)
        self._X, self._Y = np.meshgrid(self._x, self._y)
    def fresnel_limit(self):
        """
        Calcula la distancia mínima de propagación para evitar aliasing.
        
        Retorna:
            distancia mínima de propagación (en metros)
        """
        dx = self._dx
        wavelength = self._wavelenght
        N = self._N
        
        z_min = (N * dx**2) / wavelength
        return z_min

    def fresnel_propagation(self, z):
        """
        Docstring for fresnel_propagation
        
        :param self: Description
        :param z: Description
        """
        #aplicamos padding en la imagen
        self.padding2N_field()

        k = self._k
        dx = self._dx
        N = self._N
        
        # Crear coordenadas del espacio de frecuencia
        fx = np.fft.fftfreq(N, d=dx)
        fy = np.fft.fftfreq(N, d=dx)
        FX, FY = np.meshgrid(fx, fy)
        
        # Calcular el kernel de transferencia de Fresnel
        H = np.exp(-1j * (np.pi * self._wavelenght * z) * (FX**2 + FY**2))
        
        # Transformada de Fourier del campo eléctrico inicial
        E_fft = np.fft.fft2(self.__E)
        
        # Multiplicar en el dominio de la frecuencia
        E_fft_propagated = E_fft * H
        
        # Transformada inversa de Fourier para obtener el campo propagado
        E_propagated = np.fft.ifft2(E_fft_propagated)

        
        self.__E = E_propagated

        #aplicamos crop para volver al tamaño original
        self.crop_field()

    
    def crop_field(self):
        """
        Recorta el campo eléctrico al tamaño original después de la propagación con padding.
        """
        original_N = self._N // 2
        start = (self._N - original_N) // 2
        end = start + original_N
        
        E_cropped = self.__E[start:end, start:end]
        
        self.__E = E_cropped
        self._N = original_N
        self._grid_size = self._grid_size / 2
        self._x = np.linspace(-self._grid_size / 2, self._grid_size / 2, original_N)
        self._y = np.linspace(-self._grid_size / 2, self._grid_size / 2, original_N)
        self._X, self._Y = np.meshgrid(self._x, self._y)

    def lens(self, f):
        """
        Aplica una lente delgada al campo eléctrico.
        
        Parámetros:
            f : distancia focal de la lente (en metros)
        """
        k = self._k
        R = self._X**2 + self._Y**2
        lens_phase = np.exp(-1j * (k / (2 * f)) * R)
        
        self.__E = self.__E * lens_phase
    
    def pupila(self, radius):
        """
        Aplica una pupila circular al campo eléctrico.
        
        Parámetros:
            radius : radio de la pupila (en metros)
        """
        R = np.sqrt(self._X**2 + self._Y**2)
        aperture = np.where(R <= radius, 1, 0)
        
        self.__E = self.__E * aperture
    def zernike_filterself( b, phase):
        """
        Aplica un filtro de Zernike al campo eléctrico.
        
        Parámetros:
            n : orden radial
            m : orden azimutal
        """
        pass
    

    def show_intensity(self):
        intensity= np.abs(self.__E)**2
        plt.imshow(intensity, cmap='gray', extent=(-self._grid_size/2*1e3, self._grid_size/2*1e3, -self._grid_size/2*1e3, self._grid_size/2*1e3))
        plt.colorbar(label='Intensity')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('Intensity of Field')
        plt.show()
        
    def show_phase(self):
        phase = np.angle(self.__E)
        plt.imshow(phase, cmap='gray', extent=(-self._grid_size/2*1e3, self._grid_size/2*1e3, -self._grid_size/2*1e3, self._grid_size/2*1e3))
        plt.colorbar(label='Phase (radians)')
        plt.xlabel('X (mm)')
        plt.ylabel('Y (mm)')
        plt.title('Phase of Field')
        plt.show()
    

        



        

    
    




