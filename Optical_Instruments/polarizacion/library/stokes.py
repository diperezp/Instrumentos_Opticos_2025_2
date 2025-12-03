from diffraction.fresnel import *
from diffraction.utilopctic import *

class simple_stokes():
    def __init__(self,state_pol    #estado de polarizacion
                 ,wavelenght       #longitud de onda de simulacion
                 ,grid_size        #dimensiones del campo inicial
                 ,N                #cantidad de pixeles de la imagen
                 ):
        """
        vector de Jones de la iluminacion
        """
        self.__wavelenght=wavelenght
        self.__state_polarization=state_pol
        self.__grid_size=grid_size
        self.__N=N

        #instanciamos dos clases Field
        self.__field_p=Field(self.__grid_size,self.__wavelenght,self.__N)
        self.__field_s=Field(self.__grid_size,self.__wavelenght,self.__N)
    def import_phase_field_p(self,path=None):
        """
        Esta funcion importa la imagen relacionada a la fase del campo TE
        opcional se le puede ingresar el path de la imagen si no se despliega
        el administrador de archivos para selecionar el archivo a importar
        """
        self.__field_p.import_Phase(path)
        return True
    def import_intensity_field_p(self,path=None):
        """
        Esta funcion importa la imagen relacionada a la intensidad del campo TE
        opcional se le puede ingresar el path de la imagen si no se despliega
        el administrador de archivos para selecionar el archivo a importar
        """
        self.__field_p.import_Intensity(path)
        return True
    def import_intensity_field_s(self,path=None):
        """
        Esta funcion importa la imagen relacionada a la intensidad del campo TM
        opcional se le puede ingresar el path de la imagen si no se despliega
        el administrador de archivos para selecionar el archivo a importar
        """
        self.__field_s.import_Intensity(path)
        return True
    def import_phase_field_s(self,path=None):
        """
        Esta funcion importa la imagen relacionada a la fase del campo TM
        opcional se le puede ingresar el path de la imagen si no se despliega
        el administrador de archivos para selecionar el archivo a importar
        """
        self.__field_s.import_Phase(path)

    def padding2N_field(self,factor=2):

        """
        Esta funcion agrega padding al campo para sobrellevar el alias que se puede 
        generar

        @parameter:
            factor de padding
        """

        #padding al campo TE
        self.__field_p.padding2N_field()
        #padding al campo TM
        self.__field_s.padding2N_field()

        return True
    def limit_propagate(self):
        """
        Devolvemos el limite de propagacion para que la simulacion sigua el modelo 
        matematico
        """
        return self.__field_p.fresnel_limit()

    def fresnel_propagate(self,z):
        """
        Propagamos los campos TE y TM independientemente

        Esto significa que esta simulacion es consistente para experimentos donde los
        campos TE y TM estan desacoplados durante la propagacion
        """
        #propagacion del campo TE
        self.__field_p.fresnel_propagation(z)
        #propagacion del campo TM
        self.__field_s.fresnel_propagation(z)
    def lens(self,f):
        """
        Pasamos el campo vectorial por una lente de distancia focal
        f
        @parameter:
            f->distancia focal de la lente
        Se asume que la lente no es un elemento que pueda modificar 
        el estado de polarizacion del campo. Por lo tanto es de esperar que 
        al igual que a la entrada el campo este igualmente desacoplado
        """
        
        #pasamos el campo TE y TM por la lente. 
        self.__field_p.lens(f)
        self.__field_s.lens(f)
        return True
    
    def crop_field(self,factor=2):
        """
        Recortamos de la imagen el paddig añadido anteriormente
        """

        #recortamos del campo TE y TM
        self.__field_p.crop_field(factor)
        self.__field_s.crop_field(factor)
        return True
    def pupila(self,radius:float):
        """
        Aplicacion de la pupila segun sea necesario durante la propagacion 
        """
        #aplicamos la pupila al campo TE y TM
        self.__field_p.pupila(radius)
        self.__field_s.pupila(radius)
    def zernike_filter(self,radius_pupil,radius_filter,b, phas):
        """
        Implementacion del filtro de Zernike al campo
        @parameters
            radius_pupila-> radio de la pupila donde se aplica el filtro
            radius_filter-> radio del filtro
            b-> coeficiente de transmision del filtro
            phase-> añadidura de fase al campo
        """
        #aplicamso el filtro al campo TE y TM
        self.__field_p.zernike_filter(radius_pupil,radius_filter,b,phas)
        self.__field_s.zernike_filter(radius_pupil,radius_filter,b,phas)
        return True
    def __field_intensity(self): #@methodostatic
        """
        Esta funcion devuleve la intensidad del campo vectorial
        """
        Intensity_field_p=np.abs(self.__field_p.get_field())**2
        Intensity_field_s=np.abs(self.__field_s.get_field())**2
        return Intensity_field_p+Intensity_field_s
    def __field_phase(self):
        """
        Esta funcion devuelve la phase del campo vectorial
        """
        Phase_field_p=np.angle(self.__field_p.get_field())
        Phase_field_s=np.angle(self.__field_s.get_field())
        return Phase_field_p+Phase_field_s

    
    def show_field(self,fig):
        """
        Esta funcion muestra un collage del campo y sus componentes de polarizacion.
        En el momento difractivo en el que este.
        """
        #creamos una lienzo de 2 columnas y 3 filas
        axis_inte_p=fig.add_subplot(3,2,1)
        axis_phas_p=fig.add_subplot(3,2,2)
        axis_inte_s=fig.add_subplot(3,2,3)
        axis_phas_s=fig.add_subplot(3,2,4)
        axis_inte_sp=fig.add_subplot(3,2,5)
        axis_phas_sp=fig.add_subplot(3,2,6)

        #ahora mostramos en la primera fila el campo TE
        self.__field_p.show_intensity(axis_inte_p)
        axis_inte_p.set_title("espectro de intensidad del campo TE")
        axis_inte_p.set_xlabel('X (mm)')
        axis_inte_p.set_ylabel('Y (mm)')
        self.__field_p.show_phase(axis_phas_p)
        axis_phas_p.set_title("espectro de fase del campo TE")
        axis_phas_p.set_xlabel('X (mm)')
        axis_phas_p.set_ylabel('Y (mm)')

        #ahora mostramos en la segunda fila el campo TM
        self.__field_s.show_intensity(axis_inte_s)
        axis_inte_s.set_title("espectro de intensidad del campo TM")
        axis_inte_s.set_xlabel('X (mm)')
        axis_inte_s.set_ylabel('Y (mm)')
        self.__field_s.show_phase(axis_phas_s)
        axis_phas_s.set_title("espectro de fase del campo TM")
        axis_phas_s.set_xlabel('X (mm)')
        axis_phas_s.set_ylabel('Y (mm)')

        #calculamos el campo neto
        intensity_net=self.__field_intensity()
        phase_net=self.__field_phase()
        #campo total.
        axis_inte_sp.imshow(intensity_net,
                            cmap='gray',
                            extent=(-self.__grid_size/2*1e3, self.__grid_size/2*1e3, -self.__grid_size/2*1e3, self.__grid_size/2*1e3)
                            )
        axis_inte_sp.set_title("espectro de intensidad del campo neto")
        axis_inte_sp.set_xlabel('X (mm)')
        axis_inte_sp.set_ylabel('Y (mm)')

        axis_phas_sp.imshow(phase_net,
                            cmap='gray',
                            extent=(-self.__grid_size/2*1e3, self.__grid_size/2*1e3, -self.__grid_size/2*1e3, self.__grid_size/2*1e3)
                            )
        axis_phas_sp.set_title("espectro de fase del campo neto")
        axis_phas_sp.set_xlabel('X (mm)')
        axis_phas_sp.set_ylabel('Y (mm)')