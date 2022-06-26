from dataclasses import dataclass, asdict
from multiprocessing.dummy import Array
import climt
from sympl import (
    PlotFunctionMonitor,
    TimeDifferencingWrapper, UpdateFrequencyWrapper,
)
from datetime import timedelta, datetime
# import matplotlib.pyplot as plt
import numpy as np
import os
import psutil
import matplotlib.cm as cmaps
from matplotlib.colors import Normalize

from pyglet import shapes, clock, gl
import pyglet
from math import sin, cos, radians
import gc

from numba import jit

#########################################################################
# opengl pyglet sphere
# https://gist.github.com/davepape/7324958
# step = 1

# vlists = []
# for lat in range(-90, 90, step):
#     verts = []
#     texc = []
#     for lon in range(-180, 181, step):
#         x = -cos(radians(lat)) * cos(radians(lon))
#         y = sin(radians(lat))
#         z = cos(radians(lat)) * sin(radians(lon))
#         s = (lon+180) / 360.0
#         t = (lat+90) / 180.0
#         verts += [x, y, z]
#         texc += [s, t]
#         x = -cos(radians((lat+step))) * cos(radians(lon))
#         y = sin(radians((lat+step)))
#         z = cos(radians((lat+step))) * sin(radians(lon))
#         s = (lon+180) / 360.0
#         t = ((lat+step)+90) / 180.0
#         verts += [x, y, z]
#         texc += [s, t]
#     vlist = pyglet.graphics.vertex_list(
#         int(len(verts)/3), ('v3f', verts), ('t2f', texc))
#     vlists.append(vlist)

# ##########################################################################

nx = 32  # 128
ny = 16  # 64
nz = 5  # 20
width = 25

window = pyglet.window.Window(
    width * nx,
    width * ny
)  # (1024, 512)  # width * nx, width * ny)
batch = pyglet.graphics.Batch()

PRIMITIVES = []


class ArrayImage:
    """Dynamic pyglet image of a 2d numpy array using matplotlib colormaps."""

    def __init__(self, array, cmap=cmaps.viridis, norm=None, rescale=True):
        self.array = array
        self.cmap = cmap
        if norm is None:
            norm = Normalize()
        self.norm = norm
        self.rescale = rescale

        self._array_normed = np.zeros(array.shape+(4,), dtype=np.uint8)
        # this line below was the bottleneck...
        # we have removed it by setting the _tex_data array to share the buffer
        # of the normalised data _array_normed
        # self._tex_data = (pyglet.gl.GLubyte * self._array_normed_data.size)( *self._array_normed_data )
        self._tex_data = (
            pyglet.gl.GLubyte * self._array_normed.size).from_buffer(self._array_normed)
        self._update_array()

        format_size = 4
        bytes_per_channel = 1
        self.pitch = array.shape[1] * format_size * bytes_per_channel
        self.image = pyglet.image.ImageData(
            array.shape[1],
            array.shape[0],
            "RGBA",
            self._tex_data
        )
        self._update_image()

    def set_array(self, data):
        self.array = data
        self.update()

    def _update_array(self):
        if self.rescale:
            self.norm.autoscale(self.array)
        self._array_normed[:] = self.cmap(self.norm(self.array), bytes=True)
        # don't need the below any more as _tex_data points to _array_normed memory
        # self._tex_data[:] = self._array_normed

    def _update_image(self):
        self.image.set_data("RGBA", self.pitch, self._tex_data)

    def update(self):
        self._update_array()
        self._update_image()


arr_img = None

# def draw_vector(state,
# to_draw=[""])


def get_hue(
    value,
    min_value,
    max_value,
    how=None
):
    scaled_value = value
    if how == "ratio":
        scaled_value
    elif how == "cap":
        scaled_value = max(
            min(
                scaled_value, max_value
            ), min_value
        )
    if max_value == min_value:
        return 170
    hue = int(
        255 * (scaled_value - min_value) / (max_value - min_value)
    )
    return hue


def draw_vector(
    state,
    x_variation,
    y_variation,
    units=None,
    height=None
):
    """Draw a vector field
    """
    x_values = state[x_variation].values if units is None else state[x_variation].to_units(
        units)
    y_values = state[y_variation].values if units is None else state[y_variation].to_units(
        units)
    max_len = np.sqrt(np.max(x_values**2 + y_values**2))
    scale_factor = 2 * width / max_len
    shape = len(x_values.shape)
    # print(f"Drawing vector of {x_variation}, {y_variation}")
    if shape != 2:
        k = height if height is not None else 0
    print("    looping over vector values")
    for i in range(nx):
        for j in range(ny):
            x_coord = x_values[k, j, i] if shape != 2 else x_values[j, i]
            y_coord = y_values[k, j, i] if shape != 2 else y_values[j, i]
            anchor_x = width * (i + 0.5)
            anchor_y = width * (j + 0.5)
            head_x = anchor_x + x_coord * scale_factor
            head_y = anchor_y + y_coord * scale_factor
            line = shapes.Line(
                anchor_x,
                anchor_y,
                anchor_x + x_coord * scale_factor,
                anchor_y + y_coord * scale_factor,
                color=(0, 0, 255),
                batch=batch
            )
            PRIMITIVES.append(line)
            arrow_width = 0.08
            arrow_length = 0.8
            left_x = anchor_x + arrow_length * x_coord * \
                scale_factor - arrow_width * y_coord * scale_factor
            left_y = anchor_y + arrow_length * y_coord * \
                scale_factor + arrow_width * x_coord * scale_factor
            right_x = anchor_x + arrow_length * x_coord * \
                scale_factor + arrow_width * y_coord * scale_factor
            right_y = anchor_y + arrow_length * y_coord * \
                scale_factor - arrow_width * x_coord * scale_factor
            triangle = shapes.Triangle(
                head_x,
                head_y,
                left_x,
                left_y,
                right_x,
                right_y,
                color=(0, 0, 255),
                batch=batch
            )
            PRIMITIVES.append(triangle)
    return


def draw_X(
    loc_x,
    loc_y
):
    line_1 = shapes.Line(
        loc_x,
        loc_y,
        loc_x + width,
        loc_y + width,
        color=(255, 0, 0),
        batch=batch
    )
    PRIMITIVES.append(line_1)
    line_2 = shapes.Line(
        loc_x,
        loc_y + width,
        loc_x + width,
        loc_y,
        color=(255, 0, 0),
        batch=batch
    )
    PRIMITIVES.append(line_2)
    return


def draw_state(state,
               to_draw,
               units=None,
               min_value=None,
               max_value=None,
               how="ratio",
               height=None):
    """Draw the state
    Example: to_draw="surface_air_pressure" with units="mbar"

    if how="ratio" then draw ratio between lower and upper
    else take max/min
    """
    # print(state[to_draw])
    air_pressure = state[to_draw].values
    shape = len(air_pressure.shape)
    if units is not None:
        air_pressure = state[to_draw].to_units(units)
    if max_value is None:
        max_value = np.nanmax(air_pressure)
    if min_value is None:
        min_value = np.nanmin(air_pressure)
    k = None
    # print(f"Updating {to_draw} drawing")
    if shape != 2:
        k = height if height is not None else 0
    print("    looping over scalar values")
    for i in range(nx):
        for j in range(ny):
            if shape != 2:
                this_air_pr = air_pressure[k, j, i]
            else:
                this_air_pr = air_pressure[j, i]
            if np.isnan(this_air_pr):
                return
                # draw_X(width * i, width * j)
            else:
                hue = get_hue(
                    this_air_pr,
                    min_value=min_value,
                    max_value=max_value,
                    how=how
                )
                square = shapes.Rectangle(
                    width * i,
                    width * j,
                    width,
                    width,
                    color=(hue, hue, hue),
                    batch=batch
                )
                PRIMITIVES.append(square)
    return


def setup():
    print("setting up...")

    simple_physics = TimeDifferencingWrapper(climt.SimplePhysics())

    constant_duration = 6

    model_time_step = timedelta(seconds=600)

    radiation_lw = UpdateFrequencyWrapper(
        climt.RRTMGLongwave(), constant_duration*model_time_step)

    radiation_sw = UpdateFrequencyWrapper(
        climt.RRTMGShortwave(), constant_duration*model_time_step)

    # convection = climt.DryConvectiveAdjustment()
    convection = climt.EmanuelConvection()

    slab_surface = climt.SlabSurface()
    insolation = climt.Instellation()
    hydrology = climt.BucketHydrology()

    climt.set_constants_from_dict({
        'stellar_irradiance': {'value': 1200, 'units': 'W m^-2'},
        "reference_air_pressure": {"value": 1e5, "units": "Pa"}
    })

    dycore = climt.GFSDynamicalCore([
        simple_physics,
        slab_surface,
        radiation_sw,
        radiation_lw,
        convection
    ],
        number_of_damped_levels=5
    )
    dcmip = climt.DcmipInitialConditions(add_perturbation=True)

    grid = climt.get_grid(nx=nx, ny=ny, nz=nz)

    my_state = climt.get_default_state(
        [dycore, hydrology],
        grid_state=grid
    )

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

    out = dcmip(my_state)

    my_state.update(out)

    print(my_state["cloud_area_fraction_in_atmosphere_layer"].shape,
          my_state["cloud_area_fraction_in_atmosphere_layer"].units)
    my_state["time"] = datetime(2022, 5, 1, 9, 0, 0)
    my_state["counter"] = 0

    viz_array = my_state["surface_air_pressure"].values

    return my_state, dycore, insolation, viz_array, hydrology


def get_memory_size(key, state):
    try:
        return state[key].values.nbytes / 1048576
    except Exception:
        return None


def update(dt, state, dynamical_core,
           insolation, hydrology, timestep,
           global_state,  arr_img):
    global angle
    angle = angle + 1
    print("===== NEXT SIMULATION STEP =====")
    print(f"Current time: {state['time']}")
    print(f"Simulation step: {state['counter']}")
    print(
        f"Memory use: {psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2}M")
    paused = global_state.PAUSED
    print(f"Paused? {paused}")
    if paused:
        print("    ... PAUSED")
    else:
        diagnostics = insolation(state)
        state.update(diagnostics)
        diag, new_state = dynamical_core(state, timestep)
        state.update(diag)
        state.update(new_state)

        diag, new_state = hydrology(state, timestep)

        state['time'] = state["time"] + timestep
        state["counter"] = state["counter"] + 1

        # this prints a list of all fields
        # for k in sorted(my_state.keys()):
        #     try:
        #         print(k, my_state[k].units, sep=" "*(80 - len(k)))
        #     except Exception:
        #         print(k, "no units/wrong type", sep=" "*(80 - len(k)))

    try:
        print(global_state.asdict())
        if global_state.UNITS is not None:
            arr_img.array = state[global_state.SCALAR_TO_PLOT].to_units(
                global_state.UNITS)
        else:
            arr_img.array = state[global_state.SCALAR_TO_PLOT].values
        if len(state[global_state.SCALAR_TO_PLOT].values.shape) > 2:
            arr_img.array = arr_img.array[global_state.Z_PLOT, :, :]
        arr_img.cmap = global_state.SCALAR_CMAP
        arr_img.update()
    except Exception as err:
        print("unable to update arr_img due to", err)

    PRIMITIVES.clear()

    print("State memory survey:")

    state_memory_info = [
        (key, get_memory_size(key, state))
        for key in state
    ]

    print(
        "Memory utilization in current state (M):",
        np.sum([s[1] for s in state_memory_info if s[1] is not None])
    )
    return


def vizupdate(dt, state, global_state):
    PRIMITIVES = []
    print("  Updating the visualization...")

    scalar_to_plot = global_state.SCALAR_TO_PLOT
    units = global_state.UNITS
    min_value = global_state.SCALAR_MIN
    max_value = global_state.SCALAR_MAX
    how = global_state.SCALAR_HOW
    height = global_state.Z_PLOT

    draw_state(
        state,
        to_draw=scalar_to_plot,
        units=units,
        min_value=min_value,
        max_value=max_value,
        how=how,
        height=height
    )

    draw_vector(
        state,
        x_variation="eastward_wind",
        y_variation="northward_wind",
        units=None,
        height=None
    )
    print("  Visualization updated")
    return


def change_temp(loc_x, loc_y, state, percent=0.1, sign=1):
    """Change the temperatue at location [loc_x, loc_y] in state by sign*percent
    (default to +10%)
    """
    id_x = int(loc_x / width)
    id_y = int(loc_y / width)
    old_val = state["surface_temperature"].values[id_y, id_x]
    state["surface_temperature"].values[id_y,
                                        id_x] = old_val * (1 + sign * percent)
    return


def info_label(state, infobatch):
    labelstring = ""
    for k in sorted(state.keys()):
        s = ""
        try:
            s += (k + " "*(80 - len(k)) + state[k].units + "\n")
        except Exception:
            s += (k + " "*(80 - len(k)) + "no units/wrong type")
        labelstring += s
    label = pyglet.text.Label(
        text=labelstring,
        font_size=12,
        anchor_x="left",
        anchor_y="top",
        align="left",
        # multiline=True,
        batch=infobatch
    )
    return label


class GlobalState:
    def __init__(self):
        self.PAUSED = False
        self.Z_PLOT = 0
        self.SCALAR_TO_PLOT = "surface_air_pressure"
        self.UNITS = "mbar"
        self.SCALAR_MIN = None
        self.SCALAR_MAX = None
        self.SCALAR_HOW = "ratio"
        self.SCALAR_CMAP = cmaps.magma

    def asdict(self):
        return {
            "PAUSED": self.PAUSED,
            "Z_PLOT": self.Z_PLOT,
            "SCALAR_TO_PLOT": self.SCALAR_TO_PLOT,
            "UNITS": self.UNITS,
            "SCALAR_MIN": self.SCALAR_MIN,
            "SCALAR_MAX": self.SCALAR_MAX,
            "SCALAR_HOW": self.SCALAR_HOW
        }


def handle_key_press(symbol, global_state):
    print(f"Handling key press {symbol}")
    if symbol == pyglet.window.key.SPACE:
        global_state.PAUSED = not global_state.PAUSED
        print(f"Updated PAUSED to {global_state.PAUSED}")
    if symbol == pyglet.window.key.PLUS:
        global_state.Z_PLOT = min(nz - 1, global_state.Z_PLOT + 1)
        print(f"Changed vertical plot level to {global_state.Z_PLOT}")
    if symbol == pyglet.window.key.MINUS:
        global_state.Z_PLOT = max(0, global_state.Z_PLOT - 1)
    if symbol == pyglet.window.key.T:
        # surface temp
        global_state.SCALAR_TO_PLOT = "surface_temperature"
        global_state.UNITS = "degC"
        global_state.SCALAR_MIN = -40
        global_state.SCALAR_MAX = 60
        global_state.SCALAR_HOW = "cap"
        global_state.CMAP = cmaps.magma
    if symbol == pyglet.window.key.P:
        # surface pressure
        global_state.SCALAR_TO_PLOT = "surface_air_pressure"
        global_state.UNITS = "mbar"
        global_state.SCALAR_MIN = None
        global_state.SCALAR_MAX = None
        global_state.CMAP = cmaps.viridis
    if symbol == pyglet.window.key.M:
        # physical map...
        pass
    if symbol == pyglet.window.key.C:
        global_state.SCALAR_TO_PLOT = "cloud_area_fraction_in_atmosphere_layer"
        global_state.UNITS = None
        global_state.CMAP = cmaps.binary
        pass
    if symbol == pyglet.window.key.R:
        global_state.SCALAR_TO_PLOT = "convective_precipitation_rate"
        global_state.UNITS = None
        global_state.CMAP = cmaps.Blues
    return


my_state, dycore, insolation, viz_array, hydrology = setup()
# label = info_label(my_state, infobatch)

global_state = GlobalState()

arr_img = ArrayImage(
    viz_array,
    cmap=cmaps.magma
)


@ window.event
def on_mouse_press(x, y, button, param):
    sign = 1
    if button == pyglet.window.mouse.LEFT:
        sign = 1
        print("left mouse button pressed!")
    elif button == pyglet.window.mouse.RIGHT:
        sign = -1
        print("right mouse button pressed!")
    change_temp(x, y, my_state, sign=sign)


@ window.event
def on_key_press(symbol, modifiers):
    handle_key_press(symbol, global_state)
    print(global_state.asdict())


# @ window.event
# def on_draw():
#     print("-- draw event --")
#     window.clear()
#     batch.draw()


angle = 0
# gl.glEnable(gl.GL_DEPTH_TEST)


@window.event
def on_draw():
    global arr_img
    arr_img.update()

    # https://gamedev.stackexchange.com/questions/20297/how-can-i-resize-pixel-art-in-pyglet-without-making-it-blurry

    texture = arr_img.image.get_texture()
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D,
        gl.GL_TEXTURE_MAG_FILTER,
        # gl.GL_LINEAR
        gl.GL_NEAREST
    )
    texture.blit(
        0,
        0,
        width=(nx * width),
        height=(ny * width)
    )
    # gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
    # gl.glMatrixMode(gl.GL_PROJECTION)
    # gl.glLoadIdentity()
    # gl.glOrtho(-2, 2, -1, 1, -1, 1)
    # gl.glMatrixMode(gl.GL_MODELVIEW)
    # gl.glLoadIdentity()
    # gl.glRotatef(angle, 0, 1, 0)
    # gl.glColor3f(1, 1, 1)
    # gl.glEnable(gl.GL_TEXTURE_2D)
    # gl.glBindTexture(gl.GL_TEXTURE_2D, texture.id)
    # for v in vlists:
    #     v.draw(gl.GL_TRIANGLE_STRIP)


# clock.schedule_interval(
#     vizupdate,
#     interval=0.1,
#     state=my_state,
#     global_state=global_state
# )
clock.schedule_interval(
    update,
    interval=0.5,
    state=my_state,
    dynamical_core=dycore,
    insolation=insolation,
    timestep=timedelta(minutes=10),
    global_state=global_state,
    hydrology=hydrology,
    arr_img=arr_img
)

pyglet.app.run()
