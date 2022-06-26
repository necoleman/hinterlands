from pyglet import shapes
import pyglet

window = pyglet.window.Window(800, 600)
batch = pyglet.graphics.Batch()


class Plot:

    def __init__(self, ordered_coordinate_list, color):
        self.ordered_coordinate_list = ordered_coordinate_list
        self.color = color

        self.x = self.ordered_coordinate_list[0][0]
        self.y = self.ordered_coordinate_list[0][1]
        self.height = self.ordered_coordinate_list[2][1] - \
            self.ordered_coordinate_list[0][1]
        self.width = self.ordered_coordinate_list[2][0] - \
            self.ordered_coordinate_list[0][0]
        return

    def draw(self, batch):
        """draw self in a pyglet graphics batch"""
        print(
            f"drawing {self.color} square of {self.width}x{self.height} at {self.x}, {self.y}")
        square = shapes.Rectangle(
            self.x,
            self.y,
            self.width,
            self.height,
            color=self.color,
            batch=batch
        )
        return square


def gen_plot(col, row, max_row=60, max_col=80):
    return Plot([
        [col * 10, row * 10],
        [(col + 1) * 10, row * 10],
        [(col + 1) * 10, (row + 1) * 10],
        [col * 10, (row + 1) * 10]
    ], (170, 170, 170)
    )


plots = [
    gen_plot(col, row, max_row=60, max_col=80)
    for col in range(80)
    for row in range(60)
]


drawsquares = [
    plot.draw(batch)
    for plot in plots
]


@window.event
def on_draw():
    window.clear()
    batch.draw()


if __name__ == "__main__":
    pyglet.app.run()
