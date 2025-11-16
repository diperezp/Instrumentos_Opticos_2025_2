import numpy as np
from tkinter import Tk, filedialog
import cv2
from matplotlib import pyplot as plt
#importamos la librerias necesarias para hacer transformadas de Fresnel
from scipy.fft import fft2, ifft2, fftshift
from scipy.fftpack import fftfreq




class ProcesamientoOptico():
    def __init__(self, length_wave,pixel_size,transmitance=None):
        """
        Inicializa la instancia del objeto óptico.

        Parámetros
        ----------
        length_wave : float
            Longitud de onda utilizada en los cálculos (por ejemplo, en metros).
        pixel_size : float
            Tamaño de píxel del plano de muestreo (por ejemplo, en metros). Define la resolución espacial.
        transmitance : float | ndarray | callable | None, opcional
            Representa la transmitancia del elemento óptico. Puede ser:
              - Un escalar (transmitancia uniforme),
              - Un array 2D que define una máscara espacial,
              - Una función que reciba coordenadas (x, y) y devuelva la transmitancia local.
            Si es None, se considera que no hay máscara aplicada hasta que se asigne una.

        Atributos inicializados
        -----------------------
        __length_wave : float
            Almacena la longitud de onda proporcionada.
        __pixel_size : float
            Almacena el tamaño de píxel proporcionado.
        __transmitance : float | ndarray | callable | None
            Almacena la transmitancia proporcionada (o None si no se suministró).
        __field_propagated : None
            Reserva para almacenar el campo óptico propagado; inicializado en None
            y se actualizará cuando se ejecute la propagación correspondiente.

        Notas
        -----
        - El constructor no realiza comprobaciones estrictas de tipo/forma; es responsabilidad
          del usuario proporcionar valores coherentes con el resto de operaciones de la clase.
        - Asegúrese de que las unidades de longitud_wave y pixel_size sean coherentes.
        """
        self.__length_wave = length_wave
        self.__pixel_size = pixel_size
        self.__transmitance = transmitance
        self.__field_propagated = None

    @property
    def length_wave(self) -> float:
        """Obtener la longitud de onda (en las mismas unidades con que se creó)."""
        return self.__length_wave

    @length_wave.setter
    def length_wave(self, value):
        """Establecer la longitud de onda. Debe ser un número positivo."""
        if not isinstance(value, (int, float)):
            raise TypeError("length_wave debe ser un número (int o float).")
        if value <= 0:
            raise ValueError("length_wave debe ser mayor que 0.")
        self.__length_wave = float(value)

    @property
    def pixel_size(self) -> float:
        """Obtener el tamaño de píxel (en las mismas unidades con que se creó)."""
        return self.__pixel_size

    @pixel_size.setter
    def pixel_size(self, value):
        """Establecer el tamaño de píxel. Debe ser un número positivo."""
        if not isinstance(value, (int, float)):
            raise TypeError("pixel_size debe ser un número (int o float).")
        if value <= 0:
            raise ValueError("pixel_size debe ser mayor que 0.")
        self.__pixel_size = float(value)
    
    def import_transmitance(self,N=None):
        """
        Abre un diálogo de selección de archivo para cargar una imagen de transmitancia, la procesa y la almacena como un campo complejo en self.__image.
        Descripción
        -----------
        - Muestra un cuadro de diálogo (tkinter.filedialog.askopenfilename) para elegir una imagen.
        - Carga la imagen con cv2.imread(..., cv2.IMREAD_UNCHANGED) y toma el primer canal (img[:,:,0]).
        - Si se proporciona N, redimensiona la imagen a NxN (interpolación bicúbica) y guarda N en self.__N; si no, usa el tamaño original (img.shape[0]) y lo asigna a self.__N.
        - Aplica padding de self.__N//2 píxeles con valor constante 255.
        - Normaliza la imagen dividiendo por np.max(img) para llevar los valores a [0, 1].
        - Construye el campo complejo u0 = img * exp(1j*0) (fase inicialmente cero), lo almacena en self.__image y llama a self.update_parameters().
        Parámetros
        ----------
        N : int o None, opcional
            Tamaño objetivo (número de píxeles por lado) para redimensionar la imagen a NxN.
            Si es None (valor por defecto), se emplea el tamaño original de la imagen.
        Valores devueltos
        -----------------
        None
            La función no devuelve el arreglo procesado; en caso de éxito el resultado queda almacenado en self.__image.
            Si la selección de archivo se cancela o la lectura de la imagen falla, la función puede devolver None o provocar una excepción dependiendo del estado (no hay manejo explícito de errores en la implementación).
        Efectos secundarios y notas
        ---------------------------
        - Requisitos: tkinter (Tk, filedialog), OpenCV (cv2), NumPy (np).
        - El diálogo filtra por extensiones: .png, .jpg, .jpeg, .bmp, .tif, .tiff.
        - El padding usa constant_values=255 (se asume fondo blanco).
        - Actualmente no hay comprobaciones explícitas si el usuario cancela (ruta vacía) o si cv2.imread devuelve None; se recomienda añadir validación para evitar errores al indexar la imagen.
        """
        # Seleccionar archivo de imagen
        # abrir un cuadro de diálogo para seleccionar la imagen
        Tk().withdraw()  # evita que aparezca la ventana principal de Tkinter
        ruta = filedialog.askopenfilename(
        title="Selecciona una imagen",
        filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
        #cargar imagen desde las imagenes de grises
        img=cv2.imread(ruta,cv2.IMREAD_UNCHANGED)
        img=img[:,:,0]
        if N is None:
            N = img.shape[0]  # Usar el tamaño original si N no se proporciona
        else:
            img = cv2.resize(img, dsize=(N,N), interpolation=cv2.INTER_CUBIC) # Redimensionar la imagen a NxN píxeles
        img = np.pad(img, N//2, mode='constant',constant_values=255)
        
        # Padding para evitar efectos de borde
        img = img / np.max(img) #normalizar la imagen para que los valores estén entre 0 y 1

        u0 = img * np.exp(1j * 0)  # Asumiendo fase cero inicialmente
        self.__transmitance=u0
        return self.__transmitance
    
    def show_transmitance(self):
        """
        Mostrar la transmitancia almacenada en self.__transmitance usando matplotlib.
        Soporta:
          - escalar: muestra una imagen constante del tamaño de __N (si existe) o 256x256.
          - ndarray: muestra el array (si es complejo, muestra su magnitud).
          - callable: evalúa la función sobre una malla centrada usando __N y pixel_size si están definidos.
        """

        if self.__transmitance is None:
            raise ValueError("No hay ninguna transmitance asignada para mostrar.")

        t = self.__transmitance

        # Determinar tamaño de muestreo
        try:
            N = int(self.__N)
        except Exception:
            N = 256
        
        #creamos la malla de coordenadas centrada
        x = (np.arange(N) - N//2) * self.__pixel_size
        y = (np.arange(N) - N//2) * self.__pixel_size
        X, Y = np.meshgrid(x, y)


        

        # Si es complejo, mostrar magnitud
        if np.iscomplexobj(t):
            t = np.abs(t)

        # Normalizar visualización opcionalmente para mejorar contraste si valores fuera de [0,1]
        # (no modificar los datos)
        vmin = None
        vmax = None

        plt.figure()
        plt.imshow(t, cmap='gray', origin='lower', vmin=vmin, vmax=vmax)
        plt.colorbar(label='Transmitancia')
        plt.title('Transmitancia')
        plt.xlabel('x (pix)')
        plt.ylabel('y (pix)')
        plt.tight_layout()
        plt.show()
    
    def propagate(self, z=0,mode='fresnel'):
        """
        Propaga el campo óptico a una distancia z utilizando la Transformada de Fresnel.
        Parámetros
        ----------
        z : float
            Distancia de propagación en metros.
        mode : str
            Modo de propagación. Actualmente solo 'fresnel' está implementado.
        
        """
        if self.__transmitance is None:
            raise ValueError("No hay ninguna transmitance asignada para propagar.")
        if mode.lower() != 'fresnel':
            raise ValueError("Modo no reconocido. Actualmente solo 'fresnel' está implementado.")
        
        #creamos la malla de coordenadas centrada
        N = self.__transmitance.shape[0]
        x = (np.arange(N) - N//2) * self.__pixel_size
        y = (np.arange(N) - N//2) * self.__pixel_size
        X, Y = np.meshgrid(x, y)

        # Creamos la malla en el espacio de frecuencias
        dx = self.__pixel_size
        k = 2 * np.pi / self.__length_wave  # número de onda
        u0 = self.__transmitance  # campo de entrada
        # Aplicar fase esférica
        u1 = u0 * np.exp(1j * (k / (2 * z)) * (X**2 + Y**2))
        # Transformada de Fourier del campo con fase esférica
        U2 = fftshift(fft2(fftshift(u1))) * (dx**2)
        # Escalar el resultado de la Transformada de Fresnel
        Kernel_escalamiento = (np.exp(1j * k * z) / (1j * self.__length_wave * z)) * np.exp(1j * (np.pi * self.__length_wave * z) * (X**2 + Y**2) / (self.__length_wave * z)**2)
        u3 = U2 * Kernel_escalamiento

        self.__field_propagated = u3
    
    def show_propagated_field(self):
        """
        Muestra el campo óptico propagado almacenado en self.__field_propagated.
        """
        if self.__field_propagated is None:
            raise ValueError("No hay ningún campo propagado para mostrar. Realice una propagación primero.")
        
        N = self.__field_propagated.shape[0]
        x = (np.arange(N) - N//2) * self.__pixel_size
        y = (np.arange(N) - N//2) * self.__pixel_size
        x_um = x * 1e3  # Convertir a mm
        y_um = y * 1e3  # Convertir a mm

        plt.figure(figsize=(8, 6))
        plt.imshow(np.abs(self.__field_propagated), cmap='inferno', extent=[x_um[0], x_um[-1], y_um[0], y_um[-1]])
        plt.title('Campo Óptico Propagado |u|')
        plt.xlabel('x [mm]')
        plt.ylabel('y [mm]')
        plt.colorbar(label='Amplitud')
        plt.tight_layout()
        plt.show()

        

def import_image_example(N=None):
    """
    Abre un diálogo de selección de archivo para cargar una imagen de transmitancia, la procesa y la almacena como un campo complejo en self.__image.
    Descripción
    -----------
    - Muestra un cuadro de diálogo (tkinter.filedialog.askopenfilename) para elegir una imagen.
    - Carga la imagen con cv2.imread(..., cv2.IMREAD_UNCHANGED) y toma el primer canal (img[:,:,0]).
    - Si se proporciona N, redimensiona la imagen a NxN (interpolación bicúbica) y guarda N en self.__N; si no, usa el tamaño original (img.shape[0]) y lo asigna a self.__N.
    - Aplica padding de self.__N//2 píxeles con valor constante 255.
    - Normaliza la imagen dividiendo por np.max(img) para llevar los valores a [0, 1].
    - Construye el campo complejo u0 = img * exp(1j*0) (fase inicialmente cero), lo almacena en self.__image y llama a self.update_parameters().
    Parámetros
    ----------
    N : int o None, opcional
        Tamaño objetivo (número de píxeles por lado) para redimensionar la imagen a NxN.
        Si es None (valor por defecto), se emplea el tamaño original de la imagen.
    Valores devueltos
    -----------------
    None
        La función no devuelve el arreglo procesado; en caso de éxito el resultado queda almacenado en self.__image.
        Si la selección de archivo se cancela o la lectura de la imagen falla, la función puede devolver None o provocar una excepción dependiendo del estado (no hay manejo explícito de errores en la implementación).
    Efectos secundarios y notas
    ---------------------------
    - Requisitos: tkinter (Tk, filedialog), OpenCV (cv2), NumPy (np).
    - El diálogo filtra por extensiones: .png, .jpg, .jpeg, .bmp, .tif, .tiff.
    - El padding usa constant_values=255 (se asume fondo blanco).
    - Actualmente no hay comprobaciones explícitas si el usuario cancela (ruta vacía) o si cv2.imread devuelve None; se recomienda añadir validación para evitar errores al indexar la imagen.
    """
    # Seleccionar archivo de imagen
    # abrir un cuadro de diálogo para seleccionar la imagen
    Tk().withdraw()  # evita que aparezca la ventana principal de Tkinter
    ruta = filedialog.askopenfilename(
title="Selecciona una imagen",
filetypes=[("Imágenes", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")])
    #cargar imagen desde las imagenes de grises
    img=cv2.imread(ruta,cv2.IMREAD_UNCHANGED)
    img=img[:,:]
    if N is None:
        N = img.shape[0]  # Usar el tamaño original si N no se proporciona
    else:
        img = cv2.resize(img, dsize=(N,N), interpolation=cv2.INTER_CUBIC) # Redimensionar la imagen a NxN píxeles
    #img = np.pad(img, N//2, mode='constant',constant_values=255)

    # Padding para evitar efectos de borde
    img = img / np.max(img) #normalizar la imagen para que los valores estén entre 0 y 1
    return img

        