import numpy as np
import ffmpeg
from datetime import datetime
from vispy import app, scene
from vispy.scene.widgets import Console
from vispy.scene.visuals import Text

from funcs import *

# %%

app.use_app('pyglet')

asp = 16 / 9
h = 720
w = int(asp * h)

canvas = scene.SceneCanvas(keys='interactive',
                           show=True,
                           size=(w, h))

view = canvas.central_widget.add_view()
view.camera = scene.PanZoomCamera(rect=(0, 0, asp, 1),
                                  aspect=1)

face = "Times New Roman"
# %%

video = True
N = 10000
dt = 0.05
power = 0.75
perception = 100. / (20 * N ** power)
vrange = np.array([0.05, 0.1])
arange = np.array([0., 1.0])

coeffs = np.array([0.7,  # alignment
                   0.2,  # cohesion
                   0.1,  # separation
                   0.001,  # walls
                   0.024  # noise
                   ])

# x, y, vx, vy, ax, ay
frame_counter = 0
boids = np.zeros((N, 6), dtype=np.float64)
D = np.zeros((N, N), dtype=np.float64)
init_boids(boids, asp, vrange)
# nb0 = calc_neighbors(boids, perception)

arrows = scene.Arrow(arrows=directions(boids),
                     arrow_color=(1, 1, 1, 1),
                     arrow_size=5,
                     connect='segments',
                     parent=view.scene
                     )

console = Console(text_color='w', font_size=12.)
view.add_widget(console)

scene.Line(pos=np.array([[0, 0],
                         [asp, 0],
                         [asp, 1],
                         [0, 1],
                         [0, 0]
                         ]),
           color=(1, 0, 0, 1),
           connect='strip',
           method='gl',
           parent=view.scene
           )
# %%

if video:
    fname = str((power, 'perception', N, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4])) + f"boids_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
    print(fname)

    process = (
        ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{w}x{h}", r=60)
            .output(fname, pix_fmt='yuv420p', preset='slower', r=60)
            .overwrite_output()
            .run_async(pipe_stdin=True)
    )


# %%
def update(event):
    global process, boids, D, frame_counter
    console.clear()
    console.write('N: %d\nalignment:%.5f,\ncohesion:%.5f,\nseparation:%.5f,\nwalls:%.5f,\nnoise:%.5f,\nFPS:%.2f' %
                  (N, coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4], canvas.fps)
                  )
    simulate(boids, D, perception, asp, coeffs)
    propagate(boids, dt, vrange)
    periodic_walls(boids, asp)
    arrows.set_data(arrows=directions(boids))
    if video:
        frame_counter += 1
        frame = canvas.render(alpha=False)
        process.stdin.write(frame.tobytes())
        if frame_counter == 2700:
            app.quit()

    else:
        canvas.update(event)
    # print(f"{canvas.fps:0.1f}")


# %%

timer = app.Timer(interval=0, start=True, connect=update)

if __name__ == '__main__':
    canvas.measure_fps()
    app.run()
    if video:
        process.stdin.close()
        process.wait()
