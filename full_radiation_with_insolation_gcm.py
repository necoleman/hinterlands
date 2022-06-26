import climt
from sympl import (
    PlotFunctionMonitor,
    TimeDifferencingWrapper, UpdateFrequencyWrapper,
)
import numpy as np
from datetime import timedelta
from pyglet import shapes, clock
import pyglet


nx = 128
ny = 64  # 62??
nz = 20

window = pyglet.window.Window(10 * nx, 10 * ny)
batch = pyglet.graphics.Batch()

SQUARES = []


def draw_state(state,
               to_draw="surface_air_pressure",
               units="mbar"):
    air_pressure = state[to_draw].to_units(units)
    max_pr = np.max(air_pressure)
    min_pr = np.min(air_pressure)
    print(f"Updating {to_draw} drawing")
    for i in range(nx):
        for j in range(ny):
            this_air_pr = air_pressure[j, i]
            if max_pr != min_pr:
                hue = int(255 * (this_air_pr - min_pr) / (max_pr - min_pr))
            else:
                hue = 170
            # print(i, j, this_air_pr, max_pr, min_pr, hue)
            square = shapes.Rectangle(
                10 * i,
                10 * j,
                10,
                10,
                color=(hue, hue, hue),
                batch=batch
            )
            SQUARES.append(square)
    return


climt.set_constants_from_dict({
    'stellar_irradiance': {'value': 1200, 'units': 'W m^-2'}})

model_time_step = timedelta(seconds=600)
# Create components

# convection = climt.EmanuelConvection()
simple_physics = TimeDifferencingWrapper(climt.SimplePhysics())

constant_duration = 6

radiation_lw = UpdateFrequencyWrapper(
    climt.RRTMGLongwave(), constant_duration*model_time_step)

radiation_sw = UpdateFrequencyWrapper(
    climt.RRTMGShortwave(), constant_duration*model_time_step)

slab_surface = climt.SlabSurface()
insolation = climt.Instellation()

dycore = climt.GFSDynamicalCore([
    simple_physics,
    slab_surface,
    radiation_sw,
    radiation_lw,
    # convection
],
    number_of_damped_levels=5
)
grid = climt.get_grid(nx=nx, ny=ny)

# Create model state
my_state = climt.get_default_state([dycore], grid_state=grid)

# Set initial/boundary conditions
latitudes = my_state['latitude'].values
longitudes = my_state['longitude'].values

zenith_angle = np.radians(latitudes)
surface_shape = [len(longitudes), len(latitudes)]

my_state['zenith_angle'].values = zenith_angle
my_state['eastward_wind'].values[:] = np.random.randn(
    *my_state['eastward_wind'].shape)
my_state['ocean_mixed_layer_thickness'].values[:] = 1

surf_temp_profile = 290 - (40*np.sin(zenith_angle)**2)
my_state['surface_temperature'].values = surf_temp_profile

for i in range(1500*24*6):
    diagnostics = insolation(my_state)
    my_state.update(diagnostics)
    diag, my_state = dycore(my_state, model_time_step)
    my_state.update(diag)
    my_state['time'] += model_time_step

    print(my_state['time'])
