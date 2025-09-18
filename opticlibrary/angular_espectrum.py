
import numpy as np
from scipy.fft import fft2, fftshift

class AngularSpectrum:
    def __init__(self, image=None, wavelength=500e-9, N=1024, length_side=40e-9):
        self.__wavelenght = wavelength  #longitud de onda [m]
        self.__length_side = length_side #tamaño de la imagen [m]
        self.__image = image #imagen de entrada  
        self.__N=N #numero de pixeles

    def set_wavelength(self,wavelength):
        """Set the wavelength in meters.
         Args:
            wavelength (float): Wavelength in meters.
        """
        self.__wavelenght = wavelength
    def get_wavelength(self):
        return self.__wavelenght
    def set_length_side(self,length_side):
        self.__length_side = length_side
    def get_length_side(self):
        return self.__length_side
    def set_image(self,image):
        self.__image = image
    def get_image(self):
        return self.__image
    def set_N(self,N):
        self.__N = N
    def get_N(self):
        return self.__N
    
