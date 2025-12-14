from diffraction.fresnel import *
from diffraction.utilopctic import *
import cv2
import numpy as np
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

    def set_field_p(self,Field_p=None):
        self.__field_p.set_field(Field_p)
    
    def set_field_s(self,Field_s=None):
        self.__field_s.set_field(Field_s)
    
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
    def  analyzer_polarizador(self,angle_trans):
        """
        Esta funcion simula un polarizador lineal.
        @parameters
        angle_trans->angulo de transmision del polarizador
        """
        #creamos los elementos de la matriz
        self.__field_p.set_field(self.__field_p.get_field()*np.cos(angle_trans)**2+self.__field_s.get_field()*np.sin(angle_trans)*np.cos(angle_trans))
        self.__field_s.set_field(self.__field_s.get_field()*np.sin(angle_trans)**2+self.__field_p.get_field()*np.sin(angle_trans)*np.cos(angle_trans))
        return True
                                 
        return True
    def __field_intensity(self): 
        """
        Esta funcion devuleve la intensidad del campo vectorial
        """
        Intensity_field_p=np.abs(self.__field_p.get_field())**2
        Intensity_field_s=np.abs(self.__field_s.get_field())**2
        Intensity_neta= Intensity_field_p + Intensity_field_s
        return Intensity_neta
    def __field_phase(self):
        """
        Esta funcion devuelve la phase del campo vectorial
        """
        Phase_field_p=np.angle(self.__field_p.get_field())
        Phase_field_s=np.angle(self.__field_s.get_field())
        return Phase_field_p+Phase_field_s

    
    def show_field(self,axis_inte_sp):
        """
        Esta funcion muestra un collage del campo y sus componentes de polarizacion.
        En el momento difractivo en el que este.
        """


        #calculamos el campo neto
        intensity_net=self.__field_intensity()
        phase_net=self.__field_phase()
        #campo total.
        axis_inte_sp.imshow(intensity_net,
                            cmap='gray',
                            extent=(-self.__grid_size/2*1e3, self.__grid_size/2*1e3, -self.__grid_size/2*1e3, self.__grid_size/2*1e3)
                            )
        axis_inte_sp.set_title("espectro de intensidad del campo neto")
        axis_inte_sp.set_xlabel('(mm)')
    def export_field(self):


        #calculamos la intensidad del campo
        Intensity=self.__field_intensity()
        Intensity=Intensity/Intensity.max()
        
        #exportamos la imagen
        export_image(Intensity)




class birrefrigente():
    def __init__(self,path_delta_map:str=None,path_theta_map:str=None,N:int=1024):
        self.__N=N
        
        self.__theta_map=self.import_theta()
        self.__delta_map=self.import_delta()
    def import_theta(self,Path:str=None):
        image=import_image(path=Path)
        A_resized = cv2.resize(image, (self.__N, self.__N), interpolation=cv2.INTER_CUBIC)
        A_resized=A_resized.astype(np.float32)
        #normalizamos la intensidad entre 0 y 1
        if A_resized.max()!=0:
            A_resized = (A_resized / A_resized.max())*2*np.pi
        else:
            A_resized=A_resized

        self.__theta_map=A_resized
        return A_resized
    
    def import_delta(self,Path:str=None):
        image=import_image(path=Path)

        A_resized = cv2.resize(image, (self.__N, self.__N), interpolation=cv2.INTER_CUBIC)
        A_resized=A_resized.astype(np.float32)
        #normalizamos la intensidad entre 0 y 1
        if A_resized.max()!=0:
            A_resized = (A_resized / A_resized.max())*2*np.pi
        else:
            A_resized=A_resized
        self.__delta_map=A_resized
        return A_resized
    def apply_initial_retardo(self,jones_in=[1,0]):
        """
        aplica la muesta a un estado de polarizacion conocido
        uniforme
        
        :param jones_in: Estado de polarizacion de la iluminacion

        :Return
        -------
        Tupla: Ex_out,Ey_out
            Camplo transmitido pixel por pixel
        """

        H,W =self.__N,self.__N


        #Convertimos el vector de Jones a forma compleja
        jones_in = np.asarray(jones_in,dtype=complex).reshape(2)


        #expandimos el estado uniforme al tamaño de la imagen
        Ex_in = np.full((H,W),jones_in[0],dtype=complex)
        Ey_in = np.full((H,W),jones_in[1],dtype=complex)

        # Rotación según theta(x,y)
        cos = np.cos(self.__theta_map)
        sin = np.sin(self.__theta_map)

        # Terminos de la Jones local
        exp_delta = np.exp(1j * self.__delta_map)

        J11 = cos**2 + exp_delta * sin**2
        J12 = (1 - exp_delta) * cos * sin
        J21 = J12
        J22 = sin**2 + exp_delta * cos**2

        # Campo transmitido
        Ex_out = J11 * Ex_in + J12 * Ey_in
        Ey_out = J21 * Ex_in + J22 * Ey_in

        return Ex_out, Ey_out
    

class polarizador():
    def __init__(self,path_theta_map:str=None,N:int=1024):
        self.__N=N
        
        self.__theta_map=self.import_theta()
    def import_theta(self,Path:str=None):
        image=import_image(path=Path)
        A_resized = cv2.resize(image, (self.__N, self.__N), interpolation=cv2.INTER_CUBIC)
        A_resized=A_resized.astype(np.float32)
        #normalizamos la intensidad entre 0 y 1
        if A_resized.max()!=0:
            A_resized = (A_resized / A_resized.max())*2*np.pi
        else:
            A_resized=A_resized

        self.__theta_map=A_resized
        return A_resized

    def apply_initial_polarization(self,jones_in=[(1/np.sqrt(2)),(1/np.sqrt(2))]):
        """
        aplica la muesta a un estado de polarizacion conocido
        uniforme
        
        :param jones_in: Estado de polarizacion de la iluminacion

        :Return
        -------
        Tupla: Ex_out,Ey_out
            Camplo transmitido pixel por pixel
        """

        H,W =self.__N,self.__N


        #Convertimos el vector de Jones a forma compleja
        # jones_in = np.asarray(jones_in,dtype=complex).reshape(2)


        #expandimos el estado uniforme al tamaño de la imagen
        Ex_in = np.full((H,W),jones_in[0],dtype=complex)
        Ey_in = np.full((H,W),jones_in[1],dtype=complex)

        # Rotación según theta(x,y)
        cos = np.cos(self.__theta_map)
        sin = np.sin(self.__theta_map)

        # Terminos de la Jones local

        J11 = cos**2
        J12 = cos * sin
        J21 = J12
        J22 = sin**2 

        # Campo transmitido
        Ex_out = J11 * Ex_in + J12 * Ey_in
        Ey_out = J21 * Ex_in + J22 * Ey_in

        return Ex_out, Ey_out

