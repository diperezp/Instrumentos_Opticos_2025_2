from scipy.fft import fft2, ifft2
import numpy as np
import matplotlib.pyplot as plt

class Fourier:
    def __init__(self,image):
        self.image = image
        self.fourier_transform=None
        self.magnitude_spectrum=None
    def transform_scipy(self):
        self.fourier_transform = fft2(self.image)
        self.magnitude_spectrum = 20*np.log(1+np.abs(self.fourier_transform))
        self.magnitude_spectrum = np.fft.fftshift(self.magnitude_spectrum)
    def transform_numpy(self):
        self.fourier_transform = np.fft.fft2(self.image)
        self.magnitude_spectrum = 20*np.log(1+np.abs(self.fourier_transform))
        self.magnitude_spectrum = np.fft.fftshift(self.magnitude_spectrum)
    def transform(self, method):
        if method == 'scipy':
            self.transform_scipy()
        elif method == 'numpy':
            self.transform_numpy()
        else:
            raise ValueError("Method must be 'scipy' or 'numpy'")
    def graficar(self):
        plt.subplot(121),plt.imshow(self.image, cmap = 'gray')
        plt.title('Input Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122),plt.imshow(self.magnitude_spectrum, cmap = 'gray')
        plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
        plt.show()