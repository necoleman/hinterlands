import numpy as np
import matplotlib.cm as cmaps
from matplotlib.colors import Normalize
import pyglet
import pyglet.gl
from pyglet import gl


# pyglet.gl.glEnable(pyglet.gl.GL_TEXTURE_2D)
# pyglet.gl.glTexParameteri(pyglet.gl.GL_TEXTURE_2D,
#                           pyglet.gl.GL_TEXTURE_MAG_FILTER,
#                           pyglet.gl.GL_NEAREST)


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


nx = 25
ny = 7
width = 50

window = pyglet.window.Window(width * nx, width * ny)
batch = pyglet.graphics.Batch()


@window.event
def on_draw():
    global arr_img
    arr_img.update()

    # https://gamedev.stackexchange.com/questions/20297/how-can-i-resize-pixel-art-in-pyglet-without-making-it-blurry

    texture = arr_img.image.get_texture()
    gl.glTexParameteri(
        gl.GL_TEXTURE_2D,
        gl.GL_TEXTURE_MAG_FILTER,
        gl.GL_NEAREST
    )
    texture.blit(
        0,
        0,
        width=(nx * width),
        height=(ny * width)
    )


batch_data = np.array([
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0,
        1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1,
        0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
])


def update(dt):
    global arr_img
    arr_img.array = np.roll(arr_img.array, -1, axis=1)


arr_img = ArrayImage(batch_data, rescale=True)

pyglet.clock.schedule_interval(
    update,
    interval=0.3
)

pyglet.app.run()
