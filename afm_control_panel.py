import matplotlib.pyplot as plt
from collections import deque
import json
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle
from pathlib import Path
import tkinter as tk
from tkinter import scrolledtext, simpledialog

from afm_animation import AFMAnimation
from afm_callbacks import AFMCallbacks
from afm_data import TrajectoryData
from afm_state import AFMState
from afm_ui import setup_dashboard, setup_figure, setup_probe_graphics
from afm_utils import create_stage_fov, get_defocus_metrics, get_tip_position, render_camera_frame, rotate_camera_frame, update_title
from hysteresis import NanoPositioner
from sample_generation import artifact_layer, height_um, sample as stage_surface_image, width_um

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SETTINGS_PATH = BASE_DIR / "afm_default_settings.json"
DOCK_LAYOUT_PATH = BASE_DIR / "afm_dock_layout.json"
DEFAULT_IMAGE_CONFIG = {
    "path": Path(r"c:\Users\fairu\Downloads\To delete\ChatGPT Image May 20, 2026, 02_20_13 PM.png"),
    "autoload": True,
}


class ActivityTerminal:
    def __init__(self, fig, manager, history_lines=400, initial_height=220, min_height=120, max_height=420):
        self.fig = fig
        self.manager = manager
        self.history_lines = history_lines
        self.height_px = initial_height
        self.min_height = min_height
        self.max_height = max_height
        self.lines = deque(maxlen=history_lines)
        self.terminal_frame = None
        self.text_widget = None
        self._drag_start_y = None
        self._drag_start_height = None
        self._build_terminal()

    def _build_terminal(self):
        if self.manager is None:
            return
        root = getattr(self.manager, "window", None)
        tk_widget_getter = getattr(self.fig.canvas, "get_tk_widget", None)
        if root is None or tk_widget_getter is None:
            return
        try:
            canvas_widget = tk_widget_getter()
            toolbar = getattr(self.manager, "toolbar", None)
            if canvas_widget is not None:
                try:
                    canvas_widget.pack_forget()
                except Exception:
                    pass
            if toolbar is not None:
                try:
                    toolbar.pack_forget()
                except Exception:
                    pass

            terminal_frame = tk.Frame(root, bg="#111827", height=self.height_px, highlightbackground="#24364d", highlightthickness=1)
            terminal_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, padx=0, pady=0)
            terminal_frame.pack_propagate(False)
            self.terminal_frame = terminal_frame

            resize_bar = tk.Frame(terminal_frame, height=10, bg="#22364d", cursor="sb_v_double_arrow")
            resize_bar.pack(side=tk.TOP, fill=tk.X)
            resize_bar.bind("<ButtonPress-1>", self._start_resize)
            resize_bar.bind("<B1-Motion>", self._on_resize_drag)
            resize_bar.bind("<ButtonRelease-1>", self._end_resize)

            header = tk.Label(
                terminal_frame,
                text="Activity Terminal",
                anchor="w",
                bg="#17212b",
                fg="#dbeafe",
                padx=10,
                pady=5,
                font=("Consolas", 10, "bold"),
            )
            header.pack(side=tk.TOP, fill=tk.X)

            text_widget = scrolledtext.ScrolledText(
                terminal_frame,
                height=10,
                wrap=tk.WORD,
                bg="#0f172a",
                fg="#d1fae5",
                insertbackground="#d1fae5",
                font=("Consolas", 9),
                relief=tk.FLAT,
                borderwidth=0,
                padx=10,
                pady=8,
            )
            text_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
            text_widget.configure(state=tk.DISABLED)
            self.text_widget = text_widget

            if toolbar is not None:
                try:
                    toolbar.pack(side=tk.TOP, fill=tk.X)
                except Exception:
                    pass
            if canvas_widget is not None:
                try:
                    canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
                except Exception:
                    pass
            root.update_idletasks()
        except Exception:
            self.terminal_frame = None
            self.text_widget = None

    def _start_resize(self, event):
        self._drag_start_y = event.y_root
        self._drag_start_height = self.height_px

    def _on_resize_drag(self, event):
        if self.terminal_frame is None or self._drag_start_y is None or self._drag_start_height is None:
            return
        delta = self._drag_start_y - event.y_root
        new_height = max(self.min_height, min(self.max_height, self._drag_start_height + delta))
        self.height_px = int(new_height)
        self.terminal_frame.configure(height=self.height_px)
        self.terminal_frame.update_idletasks()

    def _end_resize(self, event):
        self._drag_start_y = None
        self._drag_start_height = None

    def append(self, message):
        self.lines.append(str(message))
        if self.text_widget is None:
            return
        self.text_widget.configure(state=tk.NORMAL)
        self.text_widget.delete("1.0", tk.END)
        self.text_widget.insert(tk.END, "\n".join(self.lines))
        self.text_widget.see(tk.END)
        self.text_widget.configure(state=tk.DISABLED)


class ScrollableStatusDock:
    def __init__(self, fig, panel, text_artist):
        self.fig = fig
        self.panel = panel
        self.text_artist = text_artist
        self.lines = []
        self.scroll_offset = 0
        self.fig.canvas.mpl_connect("scroll_event", self.on_scroll)

    def _get_visible_line_count(self):
        panel_height = float(self.panel.bounds[3])
        return max(8, int((panel_height - 0.10) / 0.017))

    def _clamp_offset(self):
        visible = self._get_visible_line_count()
        max_offset = max(0, len(self.lines) - visible)
        self.scroll_offset = int(max(0, min(self.scroll_offset, max_offset)))

    def set_lines(self, lines):
        self.lines = [str(line) for line in lines]
        self._clamp_offset()
        self.render()

    def render(self):
        self._clamp_offset()
        visible = self._get_visible_line_count()
        start = self.scroll_offset
        end = start + visible
        visible_lines = self.lines[start:end]
        if start > 0:
            visible_lines[0] = "..." if len(visible_lines) == 1 else f"... {visible_lines[0]}"
        if end < len(self.lines) and visible_lines:
            visible_lines[-1] = f"{visible_lines[-1]} ..."
        self.text_artist.set_text("\n".join(visible_lines))

    def on_scroll(self, event):
        if event.inaxes != self.panel.ax or not self.lines:
            return
        step = -1 if event.button == "up" else 1
        old_offset = self.scroll_offset
        self.scroll_offset += step
        self._clamp_offset()
        if self.scroll_offset != old_offset:
            self.render()
            self.fig.canvas.draw_idle()


def load_default_settings():
    if not DEFAULT_SETTINGS_PATH.exists():
        return dict(DEFAULT_IMAGE_CONFIG)
    try:
        data = json.loads(DEFAULT_SETTINGS_PATH.read_text(encoding="utf-8"))
        path_str = data.get("path")
        autoload = bool(data.get("autoload", True))
        if path_str:
            return {
                "path": Path(path_str),
                "autoload": autoload,
                "width_um": data.get("width_um"),
                "height_um": data.get("height_um"),
                "scale_um_per_px": data.get("scale_um_per_px"),
            }
    except Exception:
        pass
    return dict(DEFAULT_IMAGE_CONFIG)


def save_default_settings(default_info, autoload=True):
    path = Path(default_info["path"])
    payload = {
        "path": str(path),
        "autoload": bool(autoload),
        "width_um": default_info.get("width_um"),
        "height_um": default_info.get("height_um"),
        "scale_um_per_px": default_info.get("scale_um_per_px"),
    }
    DEFAULT_SETTINGS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


runtime_default_config = load_default_settings()

state = AFMState(stage_surface_image, width_um, height_um)
state.default_image_path = str(runtime_default_config["path"])
state.default_image_width_um = runtime_default_config.get("width_um")
state.default_image_height_um = runtime_default_config.get("height_um")
if runtime_default_config.get("scale_um_per_px") is not None:
    state.default_scale_um_per_px = float(runtime_default_config["scale_um_per_px"])
stage = NanoPositioner(log_file="movement_log.csv")
data = TrajectoryData()

fig, ax = setup_figure()
fig.suptitle("AFM Hysteresis Simulation Control Panel", fontsize=14, fontweight="bold", color="#22364d")

manager = getattr(fig.canvas, "manager", None)
if manager is not None:
    try:
        manager.set_window_title("AFM Hysteresis Simulation Control Panel")
    except Exception:
        pass
    try:
        manager.window.wm_geometry("1600x900")
    except Exception:
        try:
            manager.resize(1600, 900)
        except Exception:
            pass

initial_fov, initial_outside_mask, initial_ix, initial_iy = create_stage_fov(
    state.surface_image,
    artifact_layer,
    state.show_artifact,
    state.x,
    state.y,
    state.fov_width,
    state.fov_height,
)
state.current_fov_raw = initial_fov.copy()
img = ax.imshow(
    rotate_camera_frame(
        render_camera_frame(
            initial_fov,
            state.camera_resolution,
            outside_mask=initial_outside_mask,
            focus_model=state.get_focus_model(),
        )[0],
        state.surface_tilt_angle,
    ),
    cmap="gray",
    extent=[initial_ix, initial_ix + state.fov_width, initial_iy + state.fov_height, initial_iy],
    origin="upper",
)

ideal_line, = ax.plot([], [], "g-", linewidth=2, alpha=0.7, label="Linear Path")
hyst_line, = ax.plot([], [], "b-", linewidth=2, alpha=0.7, label="PI Hysteresis Path")
ax.legend(loc="upper right")
scale_bar_black, = ax.plot([], [], color="black", linewidth=6.0, solid_capstyle="butt", zorder=25)
scale_bar_white, = ax.plot([], [], color="white", linewidth=6.0, solid_capstyle="butt", zorder=26)
scale_bar_text = ax.text(
    0,
    0,
    "",
    color="white",
    fontsize=9,
    ha="center",
    va="bottom",
    zorder=25,
    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
)
origin_marker, = ax.plot([], [], marker="+", markersize=14, markeredgewidth=2.0, color="#ffbf00", linestyle="None", zorder=27)
origin_text = ax.text(
    0,
    0,
    "",
    color="#ffdd57",
    fontsize=9,
    ha="left",
    va="bottom",
    zorder=27,
    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
)
tip_marker, = ax.plot([], [], marker="o", markersize=5, markeredgewidth=1.2, markerfacecolor="#8ef9f3", markeredgecolor="#0f766e", linestyle="None", zorder=28)
tip_text = ax.text(
    0,
    0,
    "",
    color="#8ef9f3",
    fontsize=9,
    ha="left",
    va="bottom",
    zorder=28,
    bbox=dict(boxstyle="round,pad=0.15", facecolor=(0, 0, 0, 0.35), edgecolor="none"),
)
scan_region = Rectangle(
    (0, 0),
    state.scan_region_size_um,
    state.scan_region_size_um,
    linewidth=1.6,
    edgecolor="#8ef9f3",
    facecolor="none",
    linestyle="--",
    zorder=28,
)
ax.add_patch(scan_region)

cantilever, rod, tip, center_x_ax = setup_probe_graphics(ax)
button_objects, radio_step, status_text, activity_text, dock_manager = setup_dashboard(fig, layout_path=DOCK_LAYOUT_PATH)
activity_terminal = ActivityTerminal(fig, manager)
relocation_panel = next((panel for panel in dock_manager.panels if panel.panel_id == "relocation"), None)
status_panel = next((panel for panel in dock_manager.panels if panel.panel_id == "status_activity"), None)
status_dock = None if status_panel is None else ScrollableStatusDock(fig, status_panel, status_text)
relocation_help_text = None if relocation_panel is None else relocation_panel.children_by_role["relocation_help"]["artist"]
relocation_tooltips = {
    "save_ref": "Capture Ref: save the current viewport as the reference image for later relocation.",
    "remove_sample": "Shift Sample: simulate moving the sample away from the saved reference position.",
    "auto_origin": "Auto Origin: unsupervised search for a distinctive feature in the current viewport and use it as origin.",
    "ml_origin": "Find Labeled: supervised search for the saved origin template that you labeled earlier.",
    "relocate": "Return To Ref: move the viewport back toward the saved reference location.",
    "show_hyst": "View Error: open the hysteresis/error plot from the recorded stage motion.",
    "data_collect": "Record Data: collect motion data for analysis or machine-learning training.",
}


def refresh_status_panel():
    pos_center_x, pos_center_y = callbacks.get_position_center()
    target_center_x, target_center_y = callbacks.get_target_center()
    if state.origin_defined:
        origin_line = f"{state.origin_label}: X={state.origin_x:7.1f}  Y={state.origin_y:7.1f}"
        pos_rel_line = f"Pos Rel: dX={pos_center_x - state.origin_x:+7.1f}  dY={pos_center_y - state.origin_y:+7.1f}"
        tgt_rel_line = f"Tgt Rel: dX={target_center_x - state.origin_x:+7.1f}  dY={target_center_y - state.origin_y:+7.1f}"
    else:
        origin_line = "Origin : not set"
        pos_rel_line = "Pos Rel: not set"
        tgt_rel_line = "Tgt Rel: not set"

    digital_zoom = state.get_digital_zoom_level()
    optical_mag = state.get_current_objective_magnification()
    focus_metrics = get_defocus_metrics(state.get_focus_model(), (state.camera_resolution[1], state.camera_resolution[0]))
    effective_camera_stage_um = state.get_effective_camera_stage_position_um()
    probe_gap_um = state.get_probe_sample_gap_um()
    dof_mode = "MAN" if state.manual_dof_camera_um is not None else "AUTO"

    lines = [
        f"Surface: {state.sample_source}",
        f"Image  : {Path(state.sample_path).name if state.sample_path else 'synthetic surface'}",
        f"Camera : {state.camera_mode}",
        f"Zoom   : {digital_zoom:7g}x",
        f"Obj    : {optical_mag:7g}x",
        f"Live   : {state.camera_resolution[0]} x {state.camera_resolution[1]} px",
        f"Ref HW : {state.camera_reference_resolution[0]} x {state.camera_reference_resolution[1]} px",
        f"Aux Cam: {state.sample_view_camera_resolution[0]} x {state.sample_view_camera_resolution[1]} px",
        f"Pos Ctr: X={pos_center_x:7.1f}  Y={pos_center_y:7.1f}",
        f"Tgt Ctr: X={target_center_x:7.1f}  Y={target_center_y:7.1f}",
        origin_line,
        pos_rel_line,
        tgt_rel_line,
        f"Step   : {state.current_step:7.1f} um",
        f"Smp Z  : {state.z_stage_position_um:+7.1f} um",
        f"Cam Z  : {effective_camera_stage_um:+7.1f} um",
        f"Gap    : {probe_gap_um:+7.1f} um",
        f"Focus  : {state.focus_z_um:+7.1f} um  DOF={focus_metrics['dof_camera_um']:5.2f} um {dof_mode}",
        f"Blur   : {focus_metrics['blur_diameter_um']:7.2f} um  {focus_metrics['blur_diameter_px']:6.2f} px",
        f"Smooth : {state.smooth_move_step:7.1f} um/frame",
        f"Tilt   : {state.surface_tilt_angle:7.1f} deg",
        f"Scan   : {'ON' if state.auto_scan_active else 'OFF'}",
        f"PI     : {'ON' if state.pi_mode else 'OFF'}",
        f"Pause  : {'YES' if state.paused else 'NO'}",
        f"HUD    : {'ON' if state.show_probe_hud else 'OFF'}",
        f"FOV    : {state.fov_width} x {state.fov_height} um",
    ]
    if status_dock is not None:
        status_dock.set_lines(lines)
    else:
        status_text.set_text("\n".join(lines))
    update_origin_overlay()
    update_tip_overlay()


def log_message(message):
    print(message)
    activity_terminal.append(message)
    if activity_text is not None:
        activity_text.set_text("See Activity Terminal below")
    refresh_status_panel()
    fig.canvas.draw_idle()


def update_origin_overlay():
    if not state.origin_defined:
        origin_marker.set_data([], [])
        origin_text.set_text("")


def update_relocation_hover_help(event):
    if relocation_help_text is None:
        return
    hovered_key = None
    for key, tooltip in relocation_tooltips.items():
        if event.inaxes == button_objects[key].ax:
            hovered_key = key
            relocation_help_text.set_text(tooltip)
            fig.canvas.draw_idle()
            break
    if hovered_key is None and relocation_help_text.get_text() != "Hover over a relocation button to see what it does.":
        relocation_help_text.set_text("Hover over a relocation button to see what it does.")
        fig.canvas.draw_idle()
        return

    ox = float(state.origin_x)
    oy = float(state.origin_y)
    in_view = state.x <= ox <= state.x + state.fov_width and state.y <= oy <= state.y + state.fov_height
    if in_view:
        origin_marker.set_data([ox], [oy])
        origin_text.set_position((ox + state.fov_width * 0.015, oy - state.fov_height * 0.02))
        origin_text.set_text(state.origin_label)
    else:
        origin_marker.set_data([], [])
        origin_text.set_text("")


def update_tip_overlay():
    tip_x, tip_y = get_tip_wrapper()
    hud_visible = bool(state.show_probe_hud)
    tip_marker.set_visible(hud_visible)
    tip_text.set_visible(hud_visible)
    scan_region.set_visible(hud_visible)

    if not hud_visible:
        tip_marker.set_data([], [])
        tip_text.set_text("")
        return

    half_scan = float(state.scan_region_size_um) / 2.0
    tip_marker.set_data([tip_x], [tip_y])
    tip_text.set_position((tip_x + state.fov_width * 0.015, tip_y - state.fov_height * 0.02))
    tip_text.set_text("Cantilever Tip")
    scan_region.set_xy((tip_x - half_scan, tip_y - half_scan))
    scan_region.set_width(state.scan_region_size_um)
    scan_region.set_height(state.scan_region_size_um)


def show_viewport_context_menu(event):
    if event.inaxes != ax or event.xdata is None or event.ydata is None:
        return
    if getattr(event, "button", None) != 3:
        return
    if manager is None or getattr(manager, "window", None) is None:
        return

    clicked_x = float(event.xdata)
    clicked_y = float(event.ydata)
    menu = tk.Menu(manager.window, tearoff=0)

    def set_origin_here():
        label = simpledialog.askstring(
            "Origin Label",
            "Enter origin label:",
            initialvalue=state.origin_label if state.origin_defined else "Origin",
            parent=manager.window,
        )
        callbacks.set_origin(clicked_x, clicked_y, label=label or "Origin")

    menu.add_command(label="Set Origin Here", command=set_origin_here)
    if state.origin_defined:
        menu.add_command(label="Clear Origin", command=callbacks.clear_origin)
    try:
        menu.tk_popup(int(event.guiEvent.x_root), int(event.guiEvent.y_root))
    finally:
        menu.grab_release()


def get_tip_wrapper():
    return get_tip_position(tip, ax)


def update_title_wrapper():
    tip_x, _ = get_tip_wrapper()
    update_title(ax, fig, state.pi_mode, state.target_x, tip_x)


callbacks = AFMCallbacks(
    state,
    stage,
    fig,
    ax,
    tip,
    cantilever,
    rod,
    center_x_ax,
    data,
    update_title_wrapper,
    get_tip_wrapper,
    button_objects,
    artifact_layer,
)
callbacks.img = img
callbacks.set_log_callback(log_message)
callbacks.set_status_callback(refresh_status_panel)
callbacks.set_persist_default_callback(lambda default_info: save_default_settings(default_info, autoload=True))
callbacks.update_probe_visuals()

button_objects["up"].on_clicked(callbacks.move_up)
button_objects["down"].on_clicked(callbacks.move_down)
button_objects["left"].on_clicked(callbacks.move_left)
button_objects["right"].on_clicked(callbacks.move_right)
button_objects["stop"].on_clicked(callbacks.toggle_pause)

button_objects["pi"].on_clicked(callbacks.toggle_pi)
button_objects["auto"].on_clicked(callbacks.start_auto_scan)
button_objects["clear"].on_clicked(callbacks.clear_trails)

button_objects["save_ref"].on_clicked(callbacks.save_reference)
button_objects["remove_sample"].on_clicked(callbacks.remove_sample)
button_objects["auto_origin"].on_clicked(callbacks.auto_origin_unsupervised)
button_objects["ml_origin"].on_clicked(callbacks.ml_find_origin)
button_objects["relocate"].on_clicked(callbacks.relocate)
button_objects["show_hyst"].on_clicked(callbacks.show_hysteresis_curve)
button_objects["data_collect"].on_clicked(callbacks.start_data_collection)

button_objects["zoom_in"].on_clicked(callbacks.zoom_in)
button_objects["zoom_out"].on_clicked(callbacks.zoom_out)
button_objects["z_down"].on_clicked(callbacks.move_z_down)
button_objects["z_up"].on_clicked(callbacks.move_z_up)
button_objects["dof_down"].on_clicked(callbacks.decrease_dof)
button_objects["dof_up"].on_clicked(callbacks.increase_dof)
button_objects["dof_auto"].on_clicked(callbacks.reset_dof_auto)
button_objects["focus_reset"].on_clicked(callbacks.reset_focus)
button_objects["load_default"].on_clicked(callbacks.load_default_image)
button_objects["load_image"].on_clicked(callbacks.load_sample_image)
button_objects["save_default"].on_clicked(callbacks.save_current_as_default)
button_objects["save_layout"].on_clicked(lambda event: log_message(f"Dock layout saved to {dock_manager.save_layout()}"))
button_objects["scale_bar"].label.set_text(f"Scale Bar: {int(state.scale_bar_total_um)} um")
button_objects["scale_bar"].on_clicked(callbacks.cycle_scale_bar_length)
button_objects["coord"].on_clicked(callbacks.show_tip_coord)
button_objects["hud"].label.set_text(f"HUD: {'ON' if state.show_probe_hud else 'OFF'}")
button_objects["hud"].on_clicked(callbacks.toggle_probe_hud)
button_objects["tilt"].on_clicked(callbacks.set_tilt)
fig.canvas.mpl_connect("button_press_event", callbacks.move_to_clicked_point)
fig.canvas.mpl_connect("button_press_event", show_viewport_context_menu)
fig.canvas.mpl_connect("motion_notify_event", update_relocation_hover_help)


def on_step_selected(label):
    step = int(label.split()[0])
    callbacks.set_step(step)


radio_step.on_clicked(on_step_selected)

animation = AFMAnimation(
    state,
    stage,
    data,
    ax,
    img,
    ideal_line,
    hyst_line,
    artifact_layer,
    fig,
    get_tip_wrapper,
    scale_bar_black,
    scale_bar_white,
    scale_bar_text,
    refresh_status_panel,
    callbacks.update_probe_visuals,
)
ani = FuncAnimation(fig, animation.update, interval=state.animation_interval_ms, cache_frame_data=False)

is_closing = False


def on_close(event):
    global is_closing
    if is_closing:
        return
    is_closing = True
    try:
        data.save()
        data.plot()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        plt.close("all")


fig.canvas.mpl_connect("close_event", on_close)

update_title_wrapper()
refresh_status_panel()
if dock_manager.load_layout():
    log_message(f"Loaded dock layout: {DOCK_LAYOUT_PATH.name}")
log_message("AFM control panel ready.")
log_message("Definition: sample = the surface image mounted on the stage.")
log_message("Position and Target now mean viewport center on the stage, not the top-left corner.")
log_message("Right-click in the Viewport to set a named origin for relative coordinates.")
log_message("Relocation sequence: 1 Save Reference -> 2 Remove Sample -> 3 Auto Origin -> 4 ML Find Origin -> 5 Relocate Sample -> 6 Plot Hysteresis -> 7 Collect Data.")
log_message("Auto Origin = unsupervised distinctive-pattern pick from the current viewport. ML Find Origin = supervised search using the saved origin template.")
log_message("Named regions: Viewport, FOV, Workflow Dock, Navigation Dock, Motion Dock, Status Dock, Relocation Dock, Utility Dock.")
log_message("Suggested order: load image -> save as default if needed -> move tip -> set origin or use Auto Origin -> save reference -> remove sample -> ML find origin -> relocate sample.")
if runtime_default_config.get("autoload"):
    callbacks.load_default_image()
plt.show()
