import climt
from sympl import (
    TimeDifferencingWrapper, UpdateFrequencyWrapper,
)
from datetime import timedelta, datetime
import numpy as np


class Simulation:
    """Encapsulates a climt climate simulation: allows building
    and updating"""

    def __init__(self, nx=64, ny=32, nz=10):
        self.simple_physics = TimeDifferencingWrapper(climt.SimplePhysics())
        self.constant_duration = 6
        self.time_step = timedelta(seconds=600)

        # Set model components
        self.radiation_longwave = UpdateFrequencyWrapper(
            climt.RRTMGLongwave(),
            self.constant_duration * self.time_step
        )
        self.radiation_shortwave = UpdateFrequencyWrapper(
            climt.RRTMGShortwave(),
            self.constant_duration * self.time_step
        )
        self.convection = climt.EmanuelConvection()
        self.slab_surface = climt.SlabSurface()
        self.insolation = climt.Instellation()
        self.hydrology = climt.BucketHydrology()

        # Simulation core
        self.dynamical_core = climt.GFSDynamicalCore(
            tendency_component_list=[
                self.simple_physics,
                self.slab_surface,
                self.radiation_longwave,
                self.radiation_shortwave,
                self.convection,
                self.insolation,
                self.hydrology
            ],
            number_of_damped_levels=5
        )

        # Update initial conditions
        self.dcmip = climt.DcmipInitialConditions(add_perturbation=True)

        # Get grid
        self.grid = climt.get_grid(nx=nx, ny=ny, nz=nz)

        # Get state
        self.state = climt.get_default_state(
            self.dynamical_core,
            grid_state=self.grid
        )

        # Adjust initial conditions
        latitudes = self.state["latitude"]
        longitudes = self.state["longitude"]
        zenith_angle = np.radians(latitudes)

        self.state["zenith_angle"].values = zenith_angle
        self.state["eastward_wind"].values[:] = np.random.randn(
            *self.state["eastward_wind"].shape
        )
        self.state["ocean_mixed_layer_thickness"].values[:] = 1

        self.state["surface_temperature"].values = 290 - \
            (40 * np.sin(zenith_angle)**2)

        self.state.update(self.dcmip(self.state))

        self.state["time"] = datetime(2022, 5, 1, 9, 0, 0)
        self.state["counter"] = 0

        return

    def update(self):
        pass
