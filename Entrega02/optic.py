import numpy as np
from tkinter import Tk, filedialog
import cv2


class ProcesamientoOptico():
    def __init__(self, length_wave,pixel_size,transmitance:None):
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
                self.__N = img.shape[0]  # Usar el tamaño original si N no se proporciona
            else:
                self.__N=N
                img = cv2.resize(img, dsize=(self.__N,self.__N), interpolation=cv2.INTER_CUBIC) # Redimensionar la imagen a NxN píxeles
            img = np.pad(img, self.__N//2, mode='constant',constant_values=255)
        
            # Padding para evitar efectos de borde
            img = img / np.max(img) #normalizar la imagen para que los valores estén entre 0 y 1

            u0 = img * np.exp(1j * 0)  # Asumiendo fase cero inicialmente
            self.__image=u0
            self.update_parameters()
            return None