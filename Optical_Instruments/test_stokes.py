from polarizacion.library.stokes import *


#caracteristicas de la iluminacion y del campo incial
wavelenght=633e-9 # 633 nm
grid_size=1e-2    # 1 cm
N=1024            #numero de pixeles
#creamos el lienzo de matplotlib
fig=plt.figure(figsize=(10,12))
#instanciamos la clase simple_stokes
simple_stokes=simple_stokes(None,wavelenght,grid_size,N)

#importamos las imagenes que representan cada elemento o caracteristica del campo
simple_stokes.import_intensity_field_p()
#simple_stokes.import_phase_field_p()
simple_stokes.import_intensity_field_s()
#simple_stokes.import_phase_field_s()

#adicionamos padding a la  imagen para evitar aliasing
simple_stokes.padding2N_field()

z=0.2
diametro_pupila=0.014
foco_lente=z

simple_stokes.fresnel_propagate(z)
simple_stokes.lens(foco_lente)
simple_stokes.fresnel_propagate(z)

simple_stokes.fresnel_propagate(z)
simple_stokes.lens(foco_lente)
simple_stokes.fresnel_propagate(z)

simple_stokes.crop_field()

simple_stokes.show_field(fig=fig)


plt.show()