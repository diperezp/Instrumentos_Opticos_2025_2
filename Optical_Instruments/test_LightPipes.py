from diffraction.fresnel import Field
import numpy as np

wavelength = 500e-9  # Wavelength in meters
grid_size = 1e-4     # Grid size in meters
N = 1024              # Number of grid points


field = Field(grid_size, wavelength, N)
# field.import_Intensity(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/amplitud.png')
# field.import_Phase(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/fase.png')
field.import_Intensity(path=None)
field.import_Phase(path=None)

field.show_intensity()
field.show_phase()

# field.show_intensity()
# field.show_phase()

#calculamos el limite de Fresnel
print("Límite de Fresnel:", field.fresnel_limit())

z=0.01
diametro_pupila=0.0001
foco_lente=z

field.padding2N_field(2)


field.fresnel_propagation(z)
field.lens(foco_lente)
field.fresnel_propagation(foco_lente)

field.zernike_filter(diametro_pupila, diametro_pupila/100, 0.3, np.pi/2)
field.show_intensity()
field.show_phase()

field.fresnel_propagation(z)
field.lens(foco_lente)
field.fresnel_propagation(foco_lente)

field.crop_field()



#calculamos el limite de Fresnel
print("Límite de Fresnel:", field.fresnel_limit())




field.show_intensity()
field.show_phase()