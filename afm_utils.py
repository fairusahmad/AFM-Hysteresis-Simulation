import numpy as np

def get_tip_position(tip, ax):
    """获取 tip 尖端坐标（考虑倾斜）"""
    vertices = tip.get_xy()
    tip_vertex_axes = vertices[2]
    tip_transform = tip.get_transform()
    tip_vertex_display = tip_transform.transform(tip_vertex_axes)
    tip_vertex_data = ax.transData.inverted().transform(tip_vertex_display)
    return tip_vertex_data[0], tip_vertex_data[1]

def update_title(ax, fig, pi_mode, target_x, tip_x):
    """更新标题显示误差"""
    if pi_mode:
        error = target_x - tip_x
        ax.set_title(f"Hysteresis Error: {error:+.1f} μm", fontsize=10)
    else:
        ax.set_title("Linear Mode", fontsize=10)
    fig.canvas.draw_idle()

def create_fov_image(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height):
    """生成视野图像（叠加artifact）"""
    ix = int(np.clip(x, 0, sample.shape[1] - fov_width))
    iy = int(np.clip(y, 0, sample.shape[0] - fov_height))
    fov = sample[iy:iy+fov_height, ix:ix+fov_width].copy()
    if show_artifact and artifact_layer is not None:
        artifact_fov = artifact_layer.get_display()[iy:iy+fov_height, ix:ix+fov_width]
        fov = np.maximum(fov, artifact_fov)
    return fov, ix, iy