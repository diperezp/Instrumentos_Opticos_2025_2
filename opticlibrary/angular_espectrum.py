
import numpy as np
from scipy.fft import fft2, fftshift

class AnSpectrum:
    def __init__(self, image, lambda_0=500e-9, pixel_size=10e-6):
        self.lambda_0 = lambda_0
        self.pixel_size = pixel_size
        self.image = image
    
    def pixel_number(self):
        return self.pixel_size*self.image.shape[0]

    def calcular_espectro(self):
