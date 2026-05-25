import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle
from matplotlib.widgets import Button, RadioButtons


PANEL_FACE = "#eaf1fb"
PANEL_EDGE = "#9cb6d3"
HEADER_FACE = "#d8e6f8"
TEXT_COLOR = "#24384f"
TITLE_COLOR = "#1e3550"
SUBTITLE_COLOR = "#53708f"
DEFAULT_LAYOUT_PATH = Path(__file__).resolve().parent / "afm_dock_layout.json"


class DockablePanel:
    def __init__(
        self,
        fig,
        panel_id,
        bounds,
        title,
        subtitle,
        header_frac=0.16,
        min_size=(0.14, 0.12),
        resize_grip_frac=0.16,
    ):
        self.fig = fig
        self.panel_id = panel_id
        self.title = title
        self.subtitle = subtitle
        self.bounds = list(bounds)
        self.home_bounds = list(bounds)
        self.header_frac = header_frac
        self.min_width, self.min_height = min_size
        self.resize_grip_frac = resize_grip_frac
        self.children = []
        self.children_by_role = {}
        self.layout_callback = None

        self.ax = fig.add_axes(bounds, zorder=1)
        self.ax.set_facecolor(PANEL_FACE)
        for spine in self.ax.spines.values():
            spine.set_edgecolor(PANEL_EDGE)
            spine.set_linewidth(1.2)
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        self.header = Rectangle(
            (0, 1.0 - self.header_frac),
            1.0,
            self.header_frac,
            transform=self.ax.transAxes,
            facecolor=HEADER_FACE,
            edgecolor="none",
            zorder=0,
        )
        self.ax.add_patch(self.header)
        self.ax.text(0.04, 0.98, title, fontsize=10, fontweight="bold", color=TITLE_COLOR, va="top")
        self.ax.text(0.04, 0.90, subtitle, fontsize=7.9, color=SUBTITLE_COLOR, va="top")
        self.ax.text(0.96, 0.98, "Drag", fontsize=8.4, color=SUBTITLE_COLOR, ha="right", va="top")
        self.ax.text(0.975, 0.03, "Resize", fontsize=7.8, color=SUBTITLE_COLOR, ha="right", va="bottom")

    def set_layout_callback(self, callback):
        self.layout_callback = callback
        self._apply_bounds()

    def _register_child(self, child):
        self.children.append(child)
        role = child.get("role")
        if role:
            self.children_by_role[role] = child

    def _to_absolute_bounds(self, rel_bounds):
        x0, y0, w, h = self.bounds
        return [
            x0 + rel_bounds[0] * w,
            y0 + rel_bounds[1] * h,
            rel_bounds[2] * w,
            rel_bounds[3] * h,
        ]

    def add_button(self, key, label, facecolor="#dde7f5", hovercolor="#c9daef", fontsize=8.7, role=None):
        axis = self.fig.add_axes(self._to_absolute_bounds([0.0, 0.0, 0.2, 0.2]), zorder=2)
        axis.set_facecolor("#eef3fa")
        button = Button(axis, label, color=facecolor, hovercolor=hovercolor)
        button.label.set_fontsize(fontsize)
        self._register_child(
            {
                "type": "axes",
                "kind": "button",
                "key": key,
                "role": role or key,
                "axes": axis,
                "widget": button,
                "rel_bounds": [0.0, 0.0, 0.2, 0.2],
            }
        )
        return key, button

    def add_radio(self, title, labels, active=0, fontsize=9, role=None):
        axis = self.fig.add_axes(self._to_absolute_bounds([0.0, 0.0, 0.2, 0.2]), zorder=2)
        axis.set_facecolor("#eef3fa")
        axis.set_title(title, fontsize=fontsize)
        radio = RadioButtons(axis, labels, active=active)
        self._register_child(
            {
                "type": "axes",
                "kind": "radio",
                "role": role or title.lower().replace(" ", "_"),
                "axes": axis,
                "widget": radio,
                "rel_bounds": [0.0, 0.0, 0.2, 0.2],
            }
        )
        return radio

    def add_text_block(self, rel_x, rel_y, text="", fontsize=8.4, family=None, linespacing=1.3, weight=None, role=None):
        text_artist = self.ax.text(
            rel_x,
            rel_y,
            text,
            transform=self.ax.transAxes,
            fontsize=fontsize,
            color=TEXT_COLOR,
            va="top",
            family=family,
            linespacing=linespacing,
            fontweight=weight,
        )
        self._register_child(
            {
                "type": "text",
                "role": role,
                "artist": text_artist,
                "rel_pos": [rel_x, rel_y],
            }
        )
        return text_artist

    def set_child_bounds(self, role, rel_bounds):
        child = self.children_by_role.get(role)
        if child and child["type"] == "axes":
            child["rel_bounds"] = list(rel_bounds)

    def set_text_position(self, role, rel_x, rel_y):
        child = self.children_by_role.get(role)
        if child and child["type"] == "text":
            child["rel_pos"] = [rel_x, rel_y]

    def _relayout_content(self):
        if self.layout_callback is not None:
            self.layout_callback(self)

    def _apply_bounds(self):
        self._relayout_content()
        self.ax.set_position(self.bounds)
        for child in self.children:
            if child["type"] == "axes":
                child["axes"].set_position(self._to_absolute_bounds(child["rel_bounds"]))
            elif child["type"] == "text":
                child["artist"].set_position(child["rel_pos"])

    def move_to(self, x0, y0):
        self.bounds[0] = x0
        self.bounds[1] = y0
        self._apply_bounds()

    def resize_to(self, width, height):
        self.bounds[2] = width
        self.bounds[3] = height
        self._apply_bounds()

    def clamp(self, margin=0.01):
        self.bounds[2] = min(max(self.bounds[2], self.min_width), 1.0 - 2 * margin)
        self.bounds[3] = min(max(self.bounds[3], self.min_height), 1.0 - 2 * margin)
        width = self.bounds[2]
        height = self.bounds[3]
        x0 = min(max(self.bounds[0], margin), 1.0 - width - margin)
        y0 = min(max(self.bounds[1], margin), 1.0 - height - margin)
        self.bounds[0] = x0
        self.bounds[1] = y0
        self._apply_bounds()

    def snap(self, margin=0.01, threshold=0.02):
        x0, y0, width, height = self.bounds
        candidates_x = [margin, self.home_bounds[0], 1.0 - width - margin]
        candidates_y = [margin, self.home_bounds[1], 1.0 - height - margin]

        best_x = min(candidates_x, key=lambda value: abs(value - x0))
        best_y = min(candidates_y, key=lambda value: abs(value - y0))
        if abs(best_x - x0) <= threshold:
            x0 = best_x
        if abs(best_y - y0) <= threshold:
            y0 = best_y

        self.move_to(x0, y0)
        self.clamp(margin=margin)

    def header_contains(self, event):
        if event.x is None or event.y is None:
            return False
        bbox = self.ax.get_window_extent()
        if not (bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1):
            return False
        header_height_px = max((bbox.y1 - bbox.y0) * self.header_frac, 18.0)
        return event.y >= bbox.y1 - header_height_px

    def resize_grip_contains(self, event):
        if event.x is None or event.y is None:
            return False
        bbox = self.ax.get_window_extent()
        if not (bbox.x0 <= event.x <= bbox.x1 and bbox.y0 <= event.y <= bbox.y1):
            return False
        grip_width_px = max((bbox.x1 - bbox.x0) * self.resize_grip_frac, 18.0)
        grip_height_px = max((bbox.y1 - bbox.y0) * self.resize_grip_frac, 18.0)
        return event.x >= bbox.x1 - grip_width_px and event.y <= bbox.y0 + grip_height_px

    def serialize(self):
        return {"bounds": [float(value) for value in self.bounds]}

    def apply_layout(self, layout_dict):
        bounds = layout_dict.get("bounds")
        if not bounds or len(bounds) != 4:
            return
        self.bounds = [float(value) for value in bounds]
        self.clamp()


class DockManager:
    def __init__(self, fig, panels, layout_path=None):
        self.fig = fig
        self.panels = panels
        self.layout_path = Path(layout_path) if layout_path else DEFAULT_LAYOUT_PATH
        self.active_panel = None
        self.active_mode = None
        self.drag_offset = (0.0, 0.0)
        self.resize_anchor = (0.0, 0.0)

        canvas = self.fig.canvas
        canvas.mpl_connect("button_press_event", self.on_press)
        canvas.mpl_connect("motion_notify_event", self.on_motion)
        canvas.mpl_connect("button_release_event", self.on_release)

    def _event_to_figure(self, event):
        return self.fig.transFigure.inverted().transform((event.x, event.y))

    def on_press(self, event):
        for panel in reversed(self.panels):
            if panel.resize_grip_contains(event):
                self.active_panel = panel
                self.active_mode = "resize"
                self.resize_anchor = (panel.bounds[0], panel.bounds[1])
                return
            if panel.header_contains(event):
                self.active_panel = panel
                self.active_mode = "drag"
                fig_x, fig_y = self._event_to_figure(event)
                self.drag_offset = (fig_x - panel.bounds[0], fig_y - panel.bounds[1])
                return

    def on_motion(self, event):
        if self.active_panel is None:
            return
        fig_x, fig_y = self._event_to_figure(event)
        if self.active_mode == "resize":
            anchor_x, anchor_y = self.resize_anchor
            self.active_panel.resize_to(fig_x - anchor_x, fig_y - anchor_y)
        else:
            new_x = fig_x - self.drag_offset[0]
            new_y = fig_y - self.drag_offset[1]
            self.active_panel.move_to(new_x, new_y)
        self.active_panel.clamp()
        self.fig.canvas.draw_idle()

    def on_release(self, event):
        if self.active_panel is None:
            return
        self.active_panel.snap()
        self.active_panel = None
        self.active_mode = None
        self.fig.canvas.draw_idle()

    def serialize_layout(self):
        return {panel.panel_id: panel.serialize() for panel in self.panels}

    def save_layout(self):
        self.layout_path.write_text(json.dumps(self.serialize_layout(), indent=2), encoding="utf-8")
        return self.layout_path

    def load_layout(self):
        if not self.layout_path.exists():
            return False
        payload = json.loads(self.layout_path.read_text(encoding="utf-8"))
        for panel in self.panels:
            if panel.panel_id in payload:
                panel.apply_layout(payload[panel.panel_id])
        self.fig.canvas.draw_idle()
        return True


def setup_figure():
    fig = plt.figure(figsize=(16, 9))
    fig.patch.set_facecolor("#f3f6fb")
    ax = fig.add_axes([0.05, 0.22, 0.60, 0.70])
    ax.set_aspect("equal")
    ax.set_anchor("NW")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Distance (um)")
    ax.set_ylabel("Distance (um)")
    ax.set_title("Viewport", loc="left", fontsize=10, fontweight="bold", color=TITLE_COLOR, pad=10)
    ax.set_title("FOV", loc="right", fontsize=9, color=SUBTITLE_COLOR, pad=10)
    return fig, ax


def setup_probe_graphics(ax):
    center_x_ax = 0.50
    arrow_style = dict(ha="center", va="center", fontsize=22, color="white", weight="bold", zorder=20)

    cantilever = Rectangle(
        (center_x_ax - 0.5 / 2, 0.95),
        0.5,
        0.1,
        transform=ax.transAxes,
        facecolor="black",
        edgecolor="black",
        linewidth=2,
        zorder=20,
    )
    ax.add_patch(cantilever)

    rod = Rectangle(
        (center_x_ax - 0.08 / 2, 0.6),
        0.08,
        0.95 - 0.6,
        transform=ax.transAxes,
        facecolor="black",
        edgecolor="black",
        linewidth=2,
        zorder=20,
    )
    ax.add_patch(rod)

    tip = Polygon(
        [[center_x_ax - 0.0765 / 2, 0.6], [center_x_ax + 0.0765 / 2, 0.6], [center_x_ax, 0.5]],
        closed=True,
        transform=ax.transAxes,
        facecolor="black",
        edgecolor="black",
        linewidth=2,
        zorder=20,
    )
    ax.add_patch(tip)

    ax.text(0.5, 0.95, "^", transform=ax.transAxes, **arrow_style)
    ax.text(0.5, 0.03, "v", transform=ax.transAxes, **arrow_style)
    ax.text(0.03, 0.5, "<", transform=ax.transAxes, **arrow_style)
    ax.text(0.97, 0.5, ">", transform=ax.transAxes, **arrow_style)

    return cantilever, rod, tip, center_x_ax


def _layout_workflow(panel):
    panel.set_text_position("workflow_body", 0.04, 0.78)


def _layout_navigation(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 1.15:
        panel.set_child_bounds("step_size", [0.05, 0.16, 0.34, 0.56])
        button_w = 0.17
        button_h = 0.18
        center_x = 0.65
        center_y = 0.35
        gap = 0.03
        panel.set_child_bounds("up", [center_x - button_w / 2, center_y + button_h + gap, button_w, button_h])
        panel.set_child_bounds("left", [center_x - button_w - gap, center_y, button_w, button_h])
        panel.set_child_bounds("down", [center_x - button_w / 2, center_y - button_h - gap, button_w, button_h])
        panel.set_child_bounds("right", [center_x + gap, center_y, button_w, button_h])
    else:
        panel.set_child_bounds("step_size", [0.07, 0.48, 0.86, 0.23])
        button_w = 0.26
        button_h = 0.15
        center_x = 0.50
        center_y = 0.18
        gap = 0.04
        panel.set_child_bounds("up", [center_x - button_w / 2, center_y + button_h + gap, button_w, button_h])
        panel.set_child_bounds("left", [center_x - button_w - gap / 2, center_y, button_w, button_h])
        panel.set_child_bounds("down", [center_x - button_w / 2, center_y - button_h - gap, button_w, button_h])
        panel.set_child_bounds("right", [center_x + gap / 2, center_y, button_w, button_h])


def _layout_motion(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 1.4:
        panel.set_child_bounds("pi", [0.05, 0.62, 0.41, 0.16])
        panel.set_child_bounds("auto", [0.54, 0.62, 0.27, 0.16])
        panel.set_child_bounds("stop", [0.05, 0.40, 0.41, 0.16])
        panel.set_child_bounds("focus_reset", [0.54, 0.40, 0.27, 0.16])
        panel.set_child_bounds("zoom_in", [0.05, 0.18, 0.17, 0.16])
        panel.set_child_bounds("zoom_out", [0.24, 0.18, 0.17, 0.16])
        panel.set_child_bounds("z_down", [0.54, 0.18, 0.12, 0.16])
        panel.set_child_bounds("z_up", [0.68, 0.18, 0.12, 0.16])
    else:
        panel.set_child_bounds("pi", [0.06, 0.71, 0.86, 0.09])
        panel.set_child_bounds("auto", [0.06, 0.59, 0.86, 0.09])
        panel.set_child_bounds("stop", [0.06, 0.47, 0.86, 0.09])
        panel.set_child_bounds("focus_reset", [0.06, 0.35, 0.86, 0.09])
        panel.set_child_bounds("zoom_in", [0.06, 0.21, 0.40, 0.09])
        panel.set_child_bounds("zoom_out", [0.52, 0.21, 0.40, 0.09])
        panel.set_child_bounds("z_down", [0.06, 0.08, 0.40, 0.09])
        panel.set_child_bounds("z_up", [0.52, 0.08, 0.40, 0.09])


def _layout_status(panel):
    panel.set_text_position("status_label", 0.04, 0.82)
    panel.set_text_position("status_text", 0.04, 0.75)


def _layout_relocation(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 3.0:
        button_w = 0.12
        button_h = 0.30
        gap = 0.02
        start_x = 0.03
        y = 0.30
        roles = ["save_ref", "remove_sample", "auto_origin", "ml_origin", "relocate", "show_hyst", "data_collect"]
        for index, role in enumerate(roles):
            panel.set_child_bounds(role, [start_x + index * (button_w + gap), y, button_w, button_h])
    elif width >= height * 1.1:
        panel.set_child_bounds("save_ref", [0.05, 0.70, 0.28, 0.12])
        panel.set_child_bounds("remove_sample", [0.36, 0.70, 0.28, 0.12])
        panel.set_child_bounds("auto_origin", [0.67, 0.70, 0.28, 0.12])
        panel.set_child_bounds("ml_origin", [0.05, 0.50, 0.43, 0.12])
        panel.set_child_bounds("relocate", [0.52, 0.50, 0.43, 0.12])
        panel.set_child_bounds("show_hyst", [0.05, 0.30, 0.43, 0.12])
        panel.set_child_bounds("data_collect", [0.52, 0.30, 0.43, 0.12])
    else:
        panel.set_child_bounds("save_ref", [0.08, 0.80, 0.84, 0.06])
        panel.set_child_bounds("remove_sample", [0.08, 0.70, 0.84, 0.06])
        panel.set_child_bounds("auto_origin", [0.08, 0.60, 0.84, 0.06])
        panel.set_child_bounds("ml_origin", [0.08, 0.50, 0.84, 0.06])
        panel.set_child_bounds("relocate", [0.08, 0.40, 0.84, 0.06])
        panel.set_child_bounds("show_hyst", [0.08, 0.30, 0.84, 0.06])
        panel.set_child_bounds("data_collect", [0.08, 0.20, 0.84, 0.06])
    panel.set_text_position("relocation_help", 0.04, 0.12)


def _layout_utility(panel):
    width, height = panel.bounds[2], panel.bounds[3]
    if width >= height * 1.45:
        placements = {
            "load_default": [0.07, 0.74, 0.38, 0.11],
            "load_image": [0.55, 0.74, 0.30, 0.11],
            "save_default": [0.07, 0.58, 0.78, 0.11],
            "save_layout": [0.07, 0.42, 0.78, 0.11],
            "scale_bar": [0.07, 0.26, 0.38, 0.11],
            "tilt": [0.55, 0.26, 0.30, 0.11],
            "coord": [0.07, 0.10, 0.38, 0.11],
            "clear": [0.55, 0.10, 0.30, 0.11],
        }
    else:
        placements = {
            "load_default": [0.07, 0.73, 0.40, 0.08],
            "load_image": [0.53, 0.73, 0.40, 0.08],
            "save_default": [0.07, 0.62, 0.86, 0.08],
            "save_layout": [0.07, 0.51, 0.86, 0.08],
            "scale_bar": [0.07, 0.40, 0.40, 0.08],
            "tilt": [0.53, 0.40, 0.40, 0.08],
            "coord": [0.07, 0.29, 0.40, 0.08],
            "clear": [0.07, 0.18, 0.86, 0.08],
        }
    for role, rel_bounds in placements.items():
        panel.set_child_bounds(role, rel_bounds)


def setup_dashboard(fig, layout_path=None):
    button_objects = {}

    workflow_panel = DockablePanel(fig, "workflow", [0.69, 0.77, 0.28, 0.17], "Workflow Dock", "Panel: workflow")
    workflow_panel.add_text_block(
        0.04,
        0.78,
        "1. Load a default or microscope image.\n"
        "2. Move with the navigation pad and choose step size.\n"
        "3. Run PI mode, auto scan, or pause motion if needed.\n"
        "4. Save Reference -> Remove Sample -> Auto Origin / ML Find Origin -> Relocate.\n"
        "5. Watch live parameters and activity below.",
        fontsize=8.35,
        linespacing=1.42,
        role="workflow_body",
    )
    workflow_panel.set_layout_callback(_layout_workflow)

    navigation_panel = DockablePanel(fig, "navigation", [0.69, 0.51, 0.28, 0.22], "Navigation Dock", "Panel: step-size + motion pad")
    radio_step = navigation_panel.add_radio("Step Size", ["1 um", "5 um", "50 um", "200 um"], active=1, role="step_size")
    for key, button in [
        navigation_panel.add_button("up", "Up"),
        navigation_panel.add_button("left", "Left"),
        navigation_panel.add_button("down", "Down"),
        navigation_panel.add_button("right", "Right"),
    ]:
        button_objects[key] = button
    navigation_panel.set_layout_callback(_layout_navigation)

    motion_panel = DockablePanel(fig, "motion_view", [0.69, 0.29, 0.28, 0.18], "Motion Dock", "Panel: scan, PI, zoom, focus")
    motion_buttons = [
        ("pi", "PI Compensation Mode", "#dde7f5", "#c9daef", 8.4),
        ("auto", "Auto Scan", "#dde7f5", "#c9daef", 8.7),
        ("stop", "Motion: ON", "#f3d9d9", "#eebfc0", 8.7),
        ("focus_reset", "Best Focus", "#e9f5dd", "#d7ebc3", 8.3),
        ("zoom_in", "Zoom +", "#dde7f5", "#c9daef", 8.7),
        ("zoom_out", "Zoom -", "#dde7f5", "#c9daef", 8.7),
        ("z_down", "Z -", "#fde8d6", "#f8d5b7", 8.7),
        ("z_up", "Z +", "#fde8d6", "#f8d5b7", 8.7),
    ]
    for key, label, facecolor, hovercolor, fontsize in motion_buttons:
        button_key, button = motion_panel.add_button(key, label, facecolor, hovercolor, fontsize, role=key)
        button_objects[button_key] = button
    motion_panel.set_layout_callback(_layout_motion)

    status_panel = DockablePanel(fig, "status_activity", [0.69, 0.04, 0.28, 0.22], "Status Dock", "Panel: live parameters", min_size=(0.22, 0.18))
    status_panel.add_text_block(0.04, 0.82, "Parameters", fontsize=8.7, weight="bold", role="status_label")
    status_text = status_panel.add_text_block(0.04, 0.75, "", fontsize=8.2, family="monospace", linespacing=1.3, role="status_text")
    status_panel.set_layout_callback(_layout_status)

    relocation_panel = DockablePanel(fig, "relocation", [0.05, 0.04, 0.44, 0.13], "Relocation Dock", "Panel: sequence, unsupervised + supervised origin", min_size=(0.40, 0.18))
    for key, button in [
        relocation_panel.add_button("save_ref", "1. Capture Ref", fontsize=9.1),
        relocation_panel.add_button("remove_sample", "2. Shift Sample", fontsize=9.1),
        relocation_panel.add_button("auto_origin", "3. Auto Origin", fontsize=9.0),
        relocation_panel.add_button("ml_origin", "4. Find Labeled", fontsize=9.0),
        relocation_panel.add_button("relocate", "5. Return To Ref", fontsize=9.1),
        relocation_panel.add_button("show_hyst", "6. View Error", fontsize=8.9),
        relocation_panel.add_button("data_collect", "7. Record Data", fontsize=8.9),
    ]:
        button_objects[key] = button
    relocation_panel.add_text_block(
        0.04,
        0.12,
        "Hover over a relocation button to see what it does.",
        fontsize=8.1,
        linespacing=1.2,
        role="relocation_help",
    )
    relocation_panel.set_layout_callback(_layout_relocation)

    utility_panel = DockablePanel(fig, "utility", [0.51, 0.04, 0.16, 0.22], "Utility Dock", "Panel: surface image, plots, data")
    utility_specs = [
        ("load_default", "Load Default", 8.4),
        ("load_image", "Load Image", 8.6),
        ("save_default", "Save As Default", 8.5),
        ("save_layout", "Save Dock Layout", 8.4),
        ("scale_bar", "Scale Bar: 200 um", 8.1),
        ("clear", "Clear Path", 8.6),
        ("coord", "Tip Position", 8.5),
        ("tilt", "Stage Tilt", 8.5),
    ]
    for key, label, fontsize in utility_specs:
        button_key, button = utility_panel.add_button(key, label, fontsize=fontsize, role=key)
        button_objects[button_key] = button
    utility_panel.set_layout_callback(_layout_utility)

    panels = [
        workflow_panel,
        navigation_panel,
        motion_panel,
        status_panel,
        relocation_panel,
        utility_panel,
    ]
    dock_manager = DockManager(fig, panels, layout_path=layout_path)
    return button_objects, radio_step, status_text, None, dock_manager
