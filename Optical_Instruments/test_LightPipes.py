from diffraction.fresnel import Field
import numpy as np

wavelength = 500e-9  # Wavelength in meters
grid_size = 1e-4     # Grid size in meters
N = 1024              # Number of grid points


field_p = Field(grid_size, wavelength, N)
field_s = Field(grid_size, wavelength, N)
# field.import_Intensity(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/amplitud.png')
# field.import_Phase(path='C:/Users/diego_0hh0fmb/OneDrive/Documents/GitHub/Instrumentos_Opticos_2025_2/Entrega03/recursos/fase.png')
field_p.import_Intensity(path=None)
field_s.import_Intensity(path=None)


# field.show_intensity()
# field.show_phase()

#calculamos el limite de Fresnel
print("Límite de Fresnel:", field_p.fresnel_limit())
print("Límite de Fresnel:", field_s.fresnel_limit())

z=0.01
diametro_pupila=0.0001
foco_lente=z


field_p.padding2N_field(2)
field_s.padding2N_field(2)



field_p.fresnel_propagation(z)
field_p.lens(foco_lente)
field_p.fresnel_propagation(foco_lente)

# field_p.zernike_filter(diametro_pupila, diametro_pupila/100, 0.3, np.pi/2)
# field_p.show_intensity()
# field_p.show_phase()

field_p.fresnel_propagation(z)
field_p.lens(foco_lente)
field_p.fresnel_propagation(foco_lente)

field_p.crop_field()


field_s.fresnel_propagation(z)
field_s.lens(foco_lente)
field_s.fresnel_propagation(foco_lente)

# field_s.zernike_filter(diametro_pupila, diametro_pupila/100, 0.3, np.pi/2)
# field_s.show_intensity()
# field_s.show_phase()

field_s.fresnel_propagation(z)
field_s.lens(foco_lente)
field_s.fresnel_propagation(foco_lente)

field_s.crop_field()







