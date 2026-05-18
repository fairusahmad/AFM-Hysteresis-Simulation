import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from sample_generation import sample, width_um, height_um, artifact_layer
from hysteresis import NanoPositioner
from afm_state import AFMState
from afm_data import TrajectoryData
from afm_utils import get_tip_position, update_title
from afm_ui import setup_figure, setup_probe_graphics, setup_buttons
from afm_animation import AFMAnimation
from afm_callbacks import AFMCallbacks
from artefact_detector import ArtefactDetector

# 初始化
state = AFMState(sample, width_um, height_um)
stage = NanoPositioner(log_file="movement_log.csv")
data = TrajectoryData()

# 创建界面
fig, ax = setup_figure()

img = ax.imshow(
    sample[int(state.y):int(state.y+state.fov_height), 
           int(state.x):int(state.x+state.fov_width)],
    cmap='gray',
    extent=[state.x, state.x+state.fov_width, 
            state.y+state.fov_height, state.y],
    origin='upper'
)

ideal_line, = ax.plot([], [], 'g-', linewidth=2, alpha=0.7, label='Ideal Path')
hyst_line, = ax.plot([], [], 'b-', linewidth=2, alpha=0.7, label='Hysteresis Path')
ax.legend(loc='upper right')

cantilever, rod, tip, center_x_ax = setup_probe_graphics(ax)
button_objects, radio_step = setup_buttons()

# 辅助函数
def get_tip_wrapper():
    return get_tip_position(tip, ax)

def update_title_wrapper():
    tip_x, _ = get_tip_wrapper()
    update_title(ax, fig, state.pi_mode, state.target_x, tip_x)

# 回调管理
callbacks = AFMCallbacks(
    state, stage, fig, ax, tip, cantilever, rod, center_x_ax,
    data, update_title_wrapper, get_tip_wrapper, button_objects, artifact_layer
)
callbacks.img = img

# 绑定按钮
button_objects['up'].on_clicked(callbacks.move_up)
button_objects['down'].on_clicked(callbacks.move_down)
button_objects['left'].on_clicked(callbacks.move_left)
button_objects['right'].on_clicked(callbacks.move_right)
button_objects['stop'].on_clicked(callbacks.toggle_pause)

button_objects['pi'].on_clicked(callbacks.toggle_pi)
button_objects['auto'].on_clicked(callbacks.start_auto_scan)
button_objects['clear'].on_clicked(callbacks.clear_trails)
button_objects['show_hyst'].on_clicked(callbacks.show_hysteresis_curve)

button_objects['save_ref'].on_clicked(callbacks.save_reference)
button_objects['remove_sample'].on_clicked(callbacks.remove_sample)
button_objects['relocate'].on_clicked(callbacks.relocate)

button_objects['zoom_in'].on_clicked(callbacks.zoom_in)
button_objects['zoom_out'].on_clicked(callbacks.zoom_out)
button_objects['coord'].on_clicked(callbacks.show_tip_coord)
button_objects['tilt'].on_clicked(callbacks.set_tilt)
button_objects['artifact'].on_clicked(lambda event: callbacks.toggle_artifact(event, artifact_layer))
button_objects['data_collect'].on_clicked(callbacks.start_data_collection)

# 步长选择器
def on_step_selected(label):
    step = int(label.split()[0])
    callbacks.set_step(step)

radio_step.on_clicked(on_step_selected)

# 动画
animation = AFMAnimation(
    state, stage, data, ax, img, ideal_line, hyst_line, artifact_layer, fig, get_tip_wrapper
)
ani = FuncAnimation(fig, animation.update, interval=30, cache_frame_data=False)

# 保存和关闭
def on_close(event):
    try:
        data.save()
        data.plot()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close('all')

fig.canvas.mpl_connect('close_event', on_close)

update_title_wrapper()
plt.show()