from polarizacion.library.stokes import *


#caracteristicas de la iluminacion y del campo incial
wavelength = 633e-9  # Wavelength in meters
grid_size = 200e-6     # Grid size in meters
N = 1024              # Number of grid points
State_polar=[1,0]

#creamos el lienzo de matplotlib
fig=plt.figure(figsize=(10,12))
ax1=fig.add_subplot(1,2,1)
ax2=fig.add_subplot(1,2,2)
plt.title("Polarizador-Analizador")
#instanciamos la clase simple_stokes
simple_stokes=simple_stokes(None,wavelength,grid_size,N)
#instanciamos la muestra
Muestra=birrefrigente(N=N)

Field_p,Field_s=Muestra.apply_initial_retardo()
#importamos las imagenes que representan cada elemento o caracteristica del campo
simple_stokes.set_field_p(Field_p=Field_p)
#simple_stokes.import_phase_field_p()
simple_stokes.set_field_s(Field_s=Field_s)
#simple_stokes.import_phase_field_s()

simple_stokes.show_field(ax1)
simple_stokes.export_field()
#adicionamos padding a la  campo para evitar aliasing
simple_stokes.padding2N_field()

fTL=2e-4
MX=20
fMO=fTL
diametro_pupila=10e-3
diametro_del_lente=25.4e-3



simple_stokes.fresnel_propagate(fMO)
simple_stokes.lens(fMO)
simple_stokes.pupila(diametro_del_lente/2)
simple_stokes.fresnel_propagate(fMO)

simple_stokes.pupila(diametro_pupila/2)

simple_stokes.fresnel_propagate(fTL)
simple_stokes.lens(fTL)
simple_stokes.pupila(diametro_del_lente/2)
simple_stokes.fresnel_propagate(fTL)

#angulo de tranmision
Beta=np.pi/2
simple_stokes.analyzer_polarizador(Beta)

simple_stokes.crop_field()

simple_stokes.show_field(ax2)
simple_stokes.export_field()


plt.show()