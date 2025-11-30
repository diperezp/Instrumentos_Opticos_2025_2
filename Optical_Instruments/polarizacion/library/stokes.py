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
        field_s=Field(self.__grid_size,self.__wavelenght,self.__N)
        field_s=Field(self.__grid_size,self.__wavelenght,self.__N)
    def import_field(self,path_field_p=None,path_phase_p=None,path_field_s=None,path_phase_s=None):

        """
        Importamos la imagen que corresponda a cada elemento del campo s y p.
        """



