import numpy as np
from pyglet import shapes, clock, window, graphics, app
import pyglet

import matplotlib.cm as cmaps
from matplotlib.colors import Normalize


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
            array.shape[0], array.shape[1], "RGBA", self._tex_data)
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


nx = 25
ny = 7
width = 50

window = window.Window(width * nx, width * ny)
batch = graphics.Batch()

template = [
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
    [0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
]


def hue(val):
    return [255 * val, 255 * val, 255 * val]


batch_data = np.array([[hue(val) for val in row] for row in template])
batch_img = ArrayImage(np.array(template))

PRIMITIVES = []

for i in range(ny):
    for j in range(nx):
        square = shapes.Rectangle(
            width * j,
            width * (ny - i - 1),
            width,
            width,
            batch=batch
        )
        square.color = batch_data[i, j, :]

        #     color=(batch_data[i, j, 0],
        #            batch_data[i, j, 1],
        #            batch_data[i, j, 2]),
        #     batch=batch
        # )
        PRIMITIVES.append(square)


def update(dt):
    global batch_data
    print("update event")
    batch_data = np.roll(batch_data, -1, axis=1)
    j = int(PRIMITIVES[26].x / width)
    i = int(ny - PRIMITIVES[26].y / width + 1)
    print("Square info:",
          PRIMITIVES[26].x / width,
          ny - PRIMITIVES[26].y / width + 1,
          PRIMITIVES[26].color,

          )
    print("Numpy array:", batch_data[i, j, :])
    print(batch_data.sum(axis=2).sum(axis=0))


@window.event
def on_mouse_press(x, y, button, param):
    if button == pyglet.window.mouse.LEFT:
        id_x = int(x / width)
        id_y = int(y / width)
        batch_data[id_y, id_x, 0] = 255
        batch_data[id_y, id_x, 1] = 255
        batch_data[id_y, id_x, 2] = 255
        print("Left mouse button press")
        return
    if button == pyglet.window.mouse.RIGHT:
        id_x = int(x / width)
        id_y = int(y / width)
        batch_data[id_y, id_x, 0] = 0
        batch_data[id_y, id_x, 1] = 0
        batch_data[id_y, id_x, 2] = 0
        print("Right mouse putton press")
        return


@window.event
def on_draw():
    global batch_data
    print(f"-- drawing {len(PRIMITIVES)}")
    window.clear()
    batch.draw()
    batch.invalidate()
    # print(batch_data.sum(axis=2).sum(axis=0))
    pyglet.gl.glFlush()
    print("finished drawing")


clock.schedule_interval(
    update,
    interval=2
)

app.run()
