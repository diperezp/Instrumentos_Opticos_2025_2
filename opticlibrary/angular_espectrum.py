
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, fftshift, ifftshift

class AngularSpectrum:
    def __init__(self, image:any, wavelength=500e-9, length_side=40e-9):
        #variables de entrada
        self.__wavelenght = wavelength  #longitud de onda [m]
        self.__length_side = length_side #tamaño de la imagen [m]
        self.__image = image #imagen de entrada 
        self.__N= image.shape[1] #número de pixeles

        #variables de entorno y de uso
        self.__pixel_size=length_side/self.__N #tamaño del pixel [m]
        self.__k = 2 * np.pi / self.__wavelenght  # Número de onda [1/m]

        #coordenadas de la imagen
        self.__x_img=np.linspace(-length_side/2,length_side/2,self.__N) #coordenadas x [m]
        self.__y_img=np.linspace(-length_side/2,length_side/2,self.__N) #
        self.__X_img, self.__Y_img = np.meshgrid(self.__x_img, self.__y_img) #malla

        #coordenadas en el plano de Fourier
        self.__fx = np.fft.fftshift(np.fft.fftfreq(self.__N, d=self.__pixel_size))
        self.__fy = np.fft.fftshift(np.fft.fftfreq(self.__N, d=self.__pixel_size))
        self.__FX, self.__FY = np.meshgrid(self.__fx, self.__fy)
        self.__kz = self.__k*np.sqrt((1 - (self.__wavelenght**2)*((self.__FX) ** 2 + (self.__FY) ** 2)).astype(complex))



    def set_wavelength(self,wavelength):
        """Set the wavelength in meters.
         Args:
            wavelength (float): Wavelength in meters.
        """
        self.__wavelenght = wavelength
    def get_wavelength(self):
        """Get the wavelength in meters.
        Returns:
            float: Wavelength in meters.
        """
        return self.__wavelenght
    def set_length_side(self,length_side):
        """
        Set the length of the side of the image in meters.
        Args:
            length_side (float): Length of the side of the image in meters.
        """
        self.__length_side = length_side
    def get_length_side(self):
        """
        Get the length of the side of the image in meters.
        Returns:
            float: Length of the side of the image in meters.
        """
        return self.__length_side
    def set_image(self,image):
        """Set the input image.
        Args:
            image (any): Input image.
            """
        self.__image = image
    def get_image(self):
        """Get the input image.
        Returns:
            any: Input image.
        """
        return self.__image
    def get_N(self):
        """Get the number of pixels in one dimension of the image.
        Returns:
            int: Number of pixels in one dimension of the image.
        """
        return self.__N
    
    def fft_image(self):
        """Compute the Fourier Transform of the input image.
        Returns:
            any: Fourier Transform of the input image.
        """
        return fftshift(fft2(ifftshift(self.__image)))
    def plot_image(self):
        """Plot the input image."""
        plt.figure(figsize=(6, 6))
        plt.imshow(self.__image, cmap='gray',extent=[self.__x_img[0]*1e6, self.__x_img[-1]*1e6, self.__y_img[0]*1e6, self.__y_img[-1]*1e6])
        plt.title("Imagen de Entrada")
        plt.xlabel("x [µm]")
        plt.ylabel("y [µm]")
        plt.tight_layout()
        plt.show()
        return self.__image
    
    def plot_magnitude_spectrum(self):
        """Plot the magnitude spectrum of the Fourier Transform of the input image."""
        U0_ft = self.fft_image()
        magnitude_spectrum = 20 * np.log(1 + np.abs(U0_ft))
        plt.figure(figsize=(6, 6))
        plt.imshow(magnitude_spectrum, cmap='gray',extent=[self.__fx[0]*1e-6, self.__fx[-1]*1e-6, self.__fy[0]*1e-6, self.__fy[-1]*1e-6])
        plt.title("Espectro de Magnitud")
        plt.xlabel("f_x [1/µm]")
        plt.ylabel("f_y [1/µm]")
        plt.tight_layout()
        plt.show()
        return magnitude_spectrum
    
    def propagate(self,z,only_propagating=True):
        """Propagate the input image using the Angular Spectrum method.
        Args:
            z (float): Propagation distance in meters.
            only_propagating (bool, optional): If True, only propagating components are considered. Defaults to True.
        Returns:
            any: Propagated image in the spatial domain.
        """
        U0_ft = self.fft_image()
        if only_propagating:
            prop_mask = ((2*np.pi*self.__FX)**2 + (2*np.pi*self.__FY)**2) <= self.__k**2
            H = np.zeros_like(self.__kz, dtype=complex)
            # sqrt para mask (ya es seguro porque mask selecciona positivos)
            kz_prop = np.sqrt(self.__k**2 - (2*np.pi*self.__FX[prop_mask])**2 - (2*np.pi*self.__FY[prop_mask])**2)
            H[prop_mask] = np.exp(1j * kz_prop * z)
        else:
            H = np.exp(1j * self.__kz * z)
            # opcional: truncar componentes con decaimiento numéricamente irrelevante
            alpha = np.maximum(0, np.imag(self.__kz))  # alpha >= 0 for evanescent parts
            too_small = np.exp(-alpha * z) < 1e-12
            H[too_small] = 0

        U1 = U0_ft * H
        return U1
        

    
