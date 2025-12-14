from diffraction.fresnel import Field
import numpy as np
import matplotlib.pyplot as plt


wavelength = 633e-9  # Wavelength in meters
grid_size = 200e-6     # Grid size in meters
N = 1024              # Number of grid points


field_p = Field(grid_size, wavelength, N)
# field.import_Intensity(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/amplitud.png')
# field.import_Phase(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/fase.png')
field_p.import_Intensity()

#creamos un lienzo de matplotlib
fig=plt.figure(figsize=(6,8))
#creamos 6 axes
ax1=fig.add_subplot(1,2,1)
ax1.set_title('Muestra')
ax1.set_xlabel("(mm)")
ax2=fig.add_subplot(1,2,2)
ax2.set_title("Imagen")
ax2.set_xlabel("(mm)")




#mostramos las entradas
field_p.show_intensity(ax1)




# field.show_intensity()
# field.show_phase()

#calculamos el limite de Fresnel
print("Límite de Fresnel:", field_p.fresnel_limit())
fTL=2e-4
MX=20
fMO=fTL
diametro_pupila=10e-3
diametro_del_lente=25.4e-3


field_p.padding2N_field(2)
field_p.show_intensity(ax1)

#Lente Objetivo
field_p.fresnel_propagation(fMO)
field_p.lens(fMO)
field_p.pupila(diametro_del_lente/2)
field_p.pupila(diametro_pupila)
field_p.fresnel_propagation(fMO)

#Pupila del objetivo
field_p.pupila(diametro_pupila/2)

#Lente de Tubo
field_p.fresnel_propagation(fTL)
field_p.lens(fTL)
field_p.pupila(diametro_del_lente/2)
field_p.pupila(diametro_pupila/2)
field_p.fresnel_propagation(fTL)

field_p.crop_field()

field_p.set_Magnificacion(MX)
field_p.show_intensity(ax2)

plt.show()











