import numpy as np
import matplotlib.pyplot as plt
from diffraction.utilopctic import import_image
import cv2
from PIL import Image
from diffraction.utilopctic import export_image
from diffraction.filterZernike import filter_Zernike



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
    def get_field(self):
        """
        Funcion que devuelve el campo de la simulacion
        """
        return self.__E 
    def import_Intensity(self,path=None,Title="Seleccionar imagen"):

        if path is None:
            #llamamos a la funcion de importar imagen
            imagen = import_image(Title)
        else:
            # Cargar la imagen como un arreglo numpy
            if path:
                #cargamos el primer 
                imagen = Image.open(path)
                imagen = np.array(imagen)
                if (len(imagen.shape) >= 3):
                    imagen = imagen[:, :, 0]  # Usar solo el canal rojo si es RGB
            else:
                return None
        #redimensionamos la imagen al tamaño del campo sin utilizar cv2
        A_resized = cv2.resize(imagen, (self._N, self._N), interpolation=cv2.INTER_CUBIC)
        #cambio a float32
        A_resized = A_resized.astype(np.float32)
        #normalizamos la intensidad entre 0 y 1
        A_resized = A_resized / A_resized.max()
        self.__E = A_resized * np.exp(1j * 0)

    def import_Phase(self,path:None,Title="Seleccionar imagen"):

        if path is None:
            #llamamos a la funcion de importar imagen
            imagen = import_image(Title)
        else:
            # Cargar la imagen como un arreglo numpy
            if path:
                #cargamos el primer 
                imagen = Image.open(path)
                imagen = np.array(imagen)
                if (len(imagen.shape) >= 3):
                    imagen = imagen[:, :, 0]  # Usar solo el canal rojo si es RGB
            else:
                return None
        
        #redimensionamos la imagen al tamaño del campo sin utilizar cv2
        imagen_resized = cv2.resize(imagen, (self._N, self._N), interpolation=cv2.INTER_CUBIC)
        #normalizamoos la fase entre 0 y 2pi
        imagen_resized = (imagen_resized/imagen_resized.max())* 2 * np.pi
        self.__E=self.__E*np.exp(1j * imagen_resized)
    def padding2N_field(self,factor=2):
        """
        Agrega padding al campo eléctrico para alcanzar un nuevo tamaño físico.
        
        Parámetros:
            new_size : nuevo tamaño físico (en metros)
        """
        new_size = factor * self._grid_size
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
        # print("Calculando límite de Fresnel...")
        # print("Grid size (m):", self._grid_size)
        # print("Wavelength (m):", self._wavelenght)
        # print("N:", self._N)
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

    def zernike_filter(self,radius_pupil,radius_filter,b, phase):
        """
        Aplica un filtro de Zernike al campo eléctrico.
        
        Parámetros:
            n : orden radial
            m : orden azimutal
        """
        filter = filter_Zernike(self._X, self._Y, radius_pupil, radius_filter, phase, b)
        self.__E = self.__E * filter
        pass

    def export_field(self):
        """
        Exporta el campo eléctrico como una matriz numpy.
        
        Retorna:
            matriz numpy del campo eléctrico
        """
        export_image(self.__E)
        return self.__E
    

    def show_intensity(self,axes):
        intensity= np.abs(self.__E)**2
        axes.imshow(intensity, cmap='gray', extent=(-self._grid_size/2*1e3, self._grid_size/2*1e3, -self._grid_size/2*1e3, self._grid_size/2*1e3))
        
    def show_phase(self,axes):
        phase = np.angle(self.__E)
        axes.imshow(phase, cmap='gray', extent=(-self._grid_size/2*1e3, self._grid_size/2*1e3, -self._grid_size/2*1e3, self._grid_size/2*1e3))
    

        



        

    
    




