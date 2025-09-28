import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift
from test2 import fft2c, ifft2c
class BackpropagationAngularSpectrum:
    def __init__(self, espectrum_magnitude, spectrum_phase, z, side_length, wavelength):
        #variables de analisis
        self.__espectrum_magnitude = espectrum_magnitude
        self.__espectrum_phase = spectrum_phase
        self.__z = z
        self.__side_length = side_length
        self.__wavelength = wavelength
        self.__angular_spectrum = self.__espectrum_magnitude * np.exp(1j * self.__espectrum_phase)

        #variable globales
        self.__N=self.__angular_spectrum.shape[0]
        self.__dx=self.__side_length/self.__N
        self.__k=2*np.pi/self.__wavelength
        

        #grilla de coordenadas
        self.__fx=np.fft.fftshift(np.fft.fftfreq(self.__N, d=self.__dx))
        self.__fy=np.fft.fftshift(np.fft.fftfreq(self.__N, d=self.__dx))
        self.__FX, self.__FY=np.meshgrid(self.__fx, self.__fy)
        self.__kz = self.__k*np.sqrt((1 - (self.__wavelength**2)*((self.__FX) ** 2 + (self.__FY) ** 2)).astype(complex))
    
    def compute_transmittance(self,only_propagating=True):
        """
        Esta funcion calcula la transmitancia utiliada dada el espectro_magnitude y el espectrum_phase
        """
        #inicialmente calculamos la transformada de fourier del campo de salida
        U1=np.fft.fft2(self.__angular_spectrum)

        #calculamos la funcion de transferencia          
        if only_propagating:
            prop_mask = ((2*np.pi*self.__FX)**2 + (2*np.pi*self.__FY)**2) <= self.__k**2
            H = np.zeros_like(self.__kz, dtype=complex)
            print(f"Dimensiones de H {H.shape}")
            # sqrt para mask (ya es seguro porque mask selecciona positivos)
            kz_prop = np.sqrt(self.__k**2 - (2*np.pi*self.__FX[prop_mask])**2 - (2*np.pi*self.__FY[prop_mask])**2)
            H[prop_mask] = np.exp(-1j * kz_prop * self.__z)
        else:
            H = np.exp(-1j * self.__kz * self.__z)
            # opcional: truncar componentes con decaimiento numéricamente irrelevante
            alpha = np.maximum(0, np.imag(self.__kz))  # alpha >= 0 for evanescent parts
            too_small = np.exp(-alpha * self.__z) < 1e-12
            H[too_small] = 0
        
        #multiplicamos en el dominio de la frecuencia
        U2=U1*H
        
        #calculamos la transformada inversa de fourier para obtener el campo de entrada
        u2=np.fft.ifft2(U2)
        u2=np.fft.ifftshift(u2)



        return u2
    

    

class metodo_gerchberg_saxton:
    def __init__(self, espectrum_magnitude, num_iters=500):
        self.__espectrum_magnitude = espectrum_magnitude
        self.__num_iters = num_iters

    def reconstruct(self):
        M = self.__espectrum_magnitude
        # init: random phase
        phi = np.exp(1j * 2*np.pi*np.random.rand(*M.shape))
        F_est = M * phi

        errors = []
        for k in range(self.__num_iters):
            img = np.real(ifft2c(F_est))  # imagen espacial (se asume real)
            #aplicamos las restricciones 
            img = np.clip(img, 0, None)  # no negatividad  
            F_est = fft2c(img)
            # impose measured magnitude
            F_est = M * np.exp(1j * np.angle(F_est))

            # error
            err = np.linalg.norm(np.abs(F_est) - M) / np.linalg.norm(M)
            print(f"GS iter {k+1}/{self.__num_iters} err={err:.4e}")
            errors.append(err)

        return np.real(ifft2c(F_est))
