import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from matplotlib.patches import Polygon, Rectangle

def setup_figure():
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.subplots_adjust(bottom=0.48, left=0.08, right=0.92)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('Distance (μm)')
    ax.set_ylabel('Distance (μm)')
    return fig, ax

def setup_probe_graphics(ax):
    center_x_ax = 0.50
    arrow_style = dict(ha='center', va='center', fontsize=22, color='white', weight='bold', zorder=20)
    
    cantilever = Rectangle((center_x_ax - 0.5/2, 0.95), 0.5, 0.1,
                           transform=ax.transAxes, facecolor='black', edgecolor='black', linewidth=2, zorder=20)
    ax.add_patch(cantilever)
    
    rod = Rectangle((center_x_ax - 0.08/2, 0.6), 0.08, 0.95 - 0.6,
                    transform=ax.transAxes, facecolor='black', edgecolor='black', linewidth=2, zorder=20)
    ax.add_patch(rod)
    
    tip = Polygon([[center_x_ax - 0.0765/2, 0.6], [center_x_ax + 0.0765/2, 0.6], [center_x_ax, 0.5]],
                  closed=True, transform=ax.transAxes, facecolor='black', edgecolor='black', linewidth=2, zorder=20)
    ax.add_patch(tip)
    
    ax.text(0.5, 0.95, "↑", transform=ax.transAxes, **arrow_style)
    ax.text(0.5, 0.03, "↓", transform=ax.transAxes, **arrow_style)
    ax.text(0.03, 0.5, "←", transform=ax.transAxes, **arrow_style)
    ax.text(0.97, 0.5, "→", transform=ax.transAxes, **arrow_style)
    
    return cantilever, rod, tip, center_x_ax

def setup_buttons():
    button_objects = {}
    
    # 第1行：运动模式
    ax_pi = plt.axes([0.08, 0.38, 0.10, 0.05])
    ax_auto = plt.axes([0.20, 0.38, 0.10, 0.05])
    ax_clear = plt.axes([0.32, 0.38, 0.10, 0.05])
    ax_show_hyst = plt.axes([0.44, 0.38, 0.12, 0.05])
    
    button_objects['pi'] = Button(ax_pi, 'PI Mode')
    button_objects['auto'] = Button(ax_auto, 'Auto Scan')
    button_objects['clear'] = Button(ax_clear, 'Clear Trails')
    button_objects['show_hyst'] = Button(ax_show_hyst, 'Show Hysteresis')
    
    # 第2行：步长选择
    ax_step = plt.axes([0.08, 0.25, 0.30, 0.08])
    ax_step.set_title('Step Size')
    radio_step = RadioButtons(ax_step, ['1 μm', '5 μm', '50 μm', '200 μm'], active=1)
    
    # 第3行：样品重定位
    ax_save_ref = plt.axes([0.08, 0.18, 0.10, 0.05])
    ax_remove = plt.axes([0.20, 0.18, 0.12, 0.05])
    ax_relocate = plt.axes([0.34, 0.18, 0.10, 0.05])
    
    button_objects['save_ref'] = Button(ax_save_ref, 'Save Ref')
    button_objects['remove_sample'] = Button(ax_remove, 'Remove Sample')
    button_objects['relocate'] = Button(ax_relocate, 'Relocate')
    
    # 第4行：其他功能
    ax_zoom_in = plt.axes([0.08, 0.10, 0.08, 0.05])
    ax_zoom_out = plt.axes([0.18, 0.10, 0.08, 0.05])
    ax_coord = plt.axes([0.28, 0.10, 0.10, 0.05])
    ax_tilt = plt.axes([0.40, 0.10, 0.08, 0.05])
    ax_artifact = plt.axes([0.50, 0.10, 0.10, 0.05])
    ax_collect = plt.axes([0.62, 0.10, 0.12, 0.05])
    
    button_objects['zoom_in'] = Button(ax_zoom_in, 'Zoom In')
    button_objects['zoom_out'] = Button(ax_zoom_out, 'Zoom Out')
    button_objects['coord'] = Button(ax_coord, 'Tip Coord')
    button_objects['tilt'] = Button(ax_tilt, 'Tilt')
    button_objects['artifact'] = Button(ax_artifact, 'Artifact: ON')
    button_objects['data_collect'] = Button(ax_collect, 'Collect Data')
    
    # 方向控制（右侧）
    ax_up = plt.axes([0.70, 0.32, 0.08, 0.05])
    ax_down = plt.axes([0.70, 0.20, 0.08, 0.05])
    ax_left = plt.axes([0.60, 0.26, 0.08, 0.05])
    ax_right = plt.axes([0.80, 0.26, 0.08, 0.05])
    ax_stop = plt.axes([0.75, 0.10, 0.10, 0.05])
    
    button_objects['up'] = Button(ax_up, '↑')
    button_objects['down'] = Button(ax_down, '↓')
    button_objects['left'] = Button(ax_left, '←')
    button_objects['right'] = Button(ax_right, '→')
    button_objects['stop'] = Button(ax_stop, 'Stop')
    
    return button_objects, radio_step