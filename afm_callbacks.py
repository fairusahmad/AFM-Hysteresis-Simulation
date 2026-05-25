import random
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog

import cv2
import joblib
import matplotlib.pyplot as plt
import numpy as np

from afm_utils import create_stage_fov, render_camera_frame, rotate_camera_frame
from artefact_detector import ArtefactDetector
from image_matching import match_reference_template
from sample_generation import load_real_sample_image, load_real_sample_image_from_scale

BASE_DIR = Path(__file__).resolve().parent


class AFMCallbacks:
    SCALE_BAR_CHOICES_UM = (100.0, 200.0, 500.0)
    PROBE_TIP_REL_X = 0.50
    PROBE_TIP_REL_Y = 0.50
    PROBE_TIP_HEIGHT_REL = 0.10
    PROBE_TIP_BASE_WIDTH_REL = 0.0765
    PROBE_ROD_LENGTH_REL = 0.35
    PROBE_ROD_WIDTH_REL = 0.08
    PROBE_CANTILEVER_HEIGHT_REL = 0.10
    PROBE_CANTILEVER_WIDTH_REL = 0.50

    def __init__(
        self,
        state,
        stage,
        fig,
        ax,
        tip,
        cantilever,
        rod,
        center_x_ax,
        data,
        update_title_func,
        get_tip_func,
        buttons,
        artifact_layer,
    ):
        self.state = state
        self.stage = stage
        self.fig = fig
        self.ax = ax
        self.tip = tip
        self.cantilever = cantilever
        self.rod = rod
        self.center_x_ax = center_x_ax
        self.data = data
        self.update_title = update_title_func
        self.get_tip = get_tip_func
        self.buttons = buttons
        self.artifact_layer = artifact_layer
        self.img = None
        self.ai_mode = False
        self.ai_compensator = None
        self.busy_actions = set()
        self.log_callback = None
        self.status_callback = None
        self.persist_default_callback = None

        self.artefact_detector = None
        try:
            self.artefact_detector = ArtefactDetector()
            if getattr(self.artefact_detector, "model_loaded", False):
                self.log("Artefact detector ready")
            else:
                self.artefact_detector = None
                self.log("Artefact detector unavailable")
        except Exception as e:
            self.log(f"Failed to load artefact detector: {e}")

        inverse_model_path = self._resolve_project_path("inverse_model.pkl")
        try:
            self.ai_compensator = joblib.load(inverse_model_path)
            self.ai_mode = True
            self.log(f"AI inverse model loaded: {inverse_model_path}")
        except Exception as e:
            self.log(f"AI inverse model not available at {inverse_model_path}: {e}")

    def set_log_callback(self, callback):
        self.log_callback = callback

    def set_status_callback(self, callback):
        self.status_callback = callback

    def set_persist_default_callback(self, callback):
        self.persist_default_callback = callback

    def log(self, message):
        if self.log_callback is not None:
            self.log_callback(message)
        else:
            print(message)
        if self.status_callback is not None:
            self.status_callback()

    def _resolve_project_path(self, path_str):
        path = Path(path_str)
        if path.is_absolute():
            return path
        cwd_candidate = Path.cwd() / path
        if cwd_candidate.exists():
            return cwd_candidate
        return BASE_DIR / path

    def _begin_action(self, action_name, message):
        if action_name in self.busy_actions:
            self.log(message)
            return False
        self.busy_actions.add(action_name)
        return True

    def _end_action(self, action_name):
        self.busy_actions.discard(action_name)

    def _start_smooth_move(self, target_x, target_y):
        self.state.smooth_move_active = True
        self.state.smooth_move_target_x = target_x
        self.state.smooth_move_target_y = target_y

    def _get_zoom_level_index(self, zoom_level):
        levels = np.array(self.state.zoom_levels, dtype=float)
        return int(np.argmin(np.abs(levels - float(zoom_level))))

    def _begin_quantized_zoom(self, target_zoom_level):
        levels = list(self.state.zoom_levels)
        target_zoom_level = float(levels[self._get_zoom_level_index(target_zoom_level)])
        if self.state.zooming:
            self.log("Zoom already in progress")
            return
        if np.isclose(target_zoom_level, float(self.state.current_zoom_level)):
            self.log(f"Zoom already at {target_zoom_level:g}x")
            return

        self.state.zooming = True
        self.state.zoom_progress = 0
        self.state.zoom_direction = 1 if target_zoom_level > self.state.current_zoom_level else -1
        self.state.zoom_base_width = self.state.fov_width
        self.state.zoom_base_height = self.state.fov_height
        self.state.zoom_target_width, self.state.zoom_target_height = self.state.get_fov_for_zoom_level(target_zoom_level)
        self.state.target_zoom_level = target_zoom_level
        self.state.zoom_center_x = self.state.x + self.state.fov_width / 2.0
        self.state.zoom_center_y = self.state.y + self.state.fov_height / 2.0

    def update_probe_visuals(self):
        baseline_width_um = float(self.state.on_axis_fov_width_um)
        baseline_height_um = float(self.state.on_axis_fov_height_um)
        tip_x = float(self.state.x + self.state.fov_width * self.PROBE_TIP_REL_X)
        tip_y = float(self.state.y + self.state.fov_height * self.PROBE_TIP_REL_Y)
        z_scale = 1.0

        tip_height_um = baseline_height_um * self.PROBE_TIP_HEIGHT_REL * z_scale
        tip_base_width_um = baseline_width_um * self.PROBE_TIP_BASE_WIDTH_REL * z_scale
        rod_length_um = baseline_height_um * self.PROBE_ROD_LENGTH_REL * z_scale
        rod_width_um = baseline_width_um * self.PROBE_ROD_WIDTH_REL * z_scale
        cantilever_height_um = baseline_height_um * self.PROBE_CANTILEVER_HEIGHT_REL * z_scale
        cantilever_width_um = baseline_width_um * self.PROBE_CANTILEVER_WIDTH_REL * z_scale

        tip_base_y = tip_y + tip_height_um
        rod_y = tip_base_y
        cantilever_y = rod_y + rod_length_um

        self.cantilever.set_transform(self.ax.transData)
        self.rod.set_transform(self.ax.transData)
        self.tip.set_transform(self.ax.transData)

        self.cantilever.set_xy((tip_x - cantilever_width_um / 2.0, cantilever_y))
        self.cantilever.set_width(cantilever_width_um)
        self.cantilever.set_height(cantilever_height_um)

        self.rod.set_xy((tip_x - rod_width_um / 2.0, rod_y))
        self.rod.set_width(rod_width_um)
        self.rod.set_height(rod_length_um)

        self.tip.set_xy(
            [
                [tip_x - tip_base_width_um / 2.0, tip_base_y],
                [tip_x + tip_base_width_um / 2.0, tip_base_y],
                [tip_x, tip_y],
            ]
        )
        self.fig.canvas.draw_idle()

    def toggle_probe_hud(self, event):
        self.state.show_probe_hud = not self.state.show_probe_hud
        if "hud" in self.buttons:
            self.buttons["hud"].label.set_text(f"HUD: {'ON' if self.state.show_probe_hud else 'OFF'}")
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()

    def _clamp_z_stage(self, z_position_um):
        half_travel = float(self.state.z_stage_travel_um) / 2.0
        return float(np.clip(z_position_um, -half_travel, half_travel))

    def get_position_center(self):
        return (
            float(self.state.x + self.state.fov_width / 2.0),
            float(self.state.y + self.state.fov_height / 2.0),
        )

    def get_target_center(self):
        return (
            float(self.state.target_x + self.state.fov_width / 2.0),
            float(self.state.target_y + self.state.fov_height / 2.0),
        )

    def set_origin(self, x_um, y_um, label=None):
        self.state.origin_x = float(x_um)
        self.state.origin_y = float(y_um)
        if label:
            self.state.origin_label = str(label)
        elif not self.state.origin_label:
            self.state.origin_label = "Origin"
        self.state.origin_defined = True
        self._capture_origin_template(self.state.origin_x, self.state.origin_y)
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log(
            f"Origin set: {self.state.origin_label} at "
            f"X={self.state.origin_x:.1f} um, Y={self.state.origin_y:.1f} um"
        )

    def clear_origin(self):
        self.state.origin_defined = False
        self.state.origin_template = None
        if self.status_callback is not None:
            self.status_callback()
        self.fig.canvas.draw_idle()
        self.log("Origin cleared")

    def _capture_origin_template(self, abs_x, abs_y):
        if self.state.current_fov_raw is None:
            self.state.origin_template = None
            return
        rel_x = int(round(abs_x - self.state.x))
        rel_y = int(round(abs_y - self.state.y))
        half_size = int(self.state.origin_template_half_size)
        x0 = max(0, rel_x - half_size)
        x1 = min(self.state.current_fov_raw.shape[1], rel_x + half_size)
        y0 = max(0, rel_y - half_size)
        y1 = min(self.state.current_fov_raw.shape[0], rel_y + half_size)
        if x1 - x0 < 16 or y1 - y0 < 16:
            self.state.origin_template = None
            self.log("Origin template too small in current view; move origin away from the edge and set it again.")
            return
        self.state.origin_template = self.state.current_fov_raw[y0:y1, x0:x1].copy()

    def _load_sample_into_state(self, sample, width_um, height_um, sample_source, sample_path=None):
        self.state.surface_image = sample
        self.state.sample = self.state.surface_image
        self.state.width_um = float(width_um)
        self.state.height_um = float(height_um)
        self.state.sample_source = sample_source
        self.state.sample_path = str(sample_path) if sample_path else None
        if sample_path:
            self.state.default_image_path = str(sample_path)

        self.state.current_zoom_level = 1.0
        self.state.target_zoom_level = 1.0
        self.state.fov_width, self.state.fov_height = self.state.get_fov_for_zoom_level(self.state.current_zoom_level)
        self.state.x = (self.state.width_um - self.state.fov_width) / 2.0
        self.state.y = (self.state.height_um - self.state.fov_height) / 2.0
        self.state.target_x = self.state.x
        self.state.target_y = self.state.y
        self.state.stage_margin_um = max(2000.0, float(max(self.state.width_um, self.state.height_um)) * 1.5)
        self.state.sample_removed = False
        self.state.show_artifact = False
        self.state.ai_desired_history_x = [self.state.x, self.state.x]
        self.state.ai_desired_history_y = [self.state.y, self.state.y]
        self._clear_reference_data()

        if self.artifact_layer is not None:
            self.artifact_layer.reset_canvas(sample.shape[1], sample.shape[0])

        self.stage.reset(self.state.x, self.state.y)
        self._refresh_current_view()
        self.update_title()

    def load_default_image(self, event=None):
        if not self._begin_action("load_default_image", "Default image loading is already running"):
            return
        try:
            candidate_paths = []
            if self.state.default_image_path:
                candidate_paths.append(Path(self.state.default_image_path))
            candidate_paths.extend(
                [
                    self._resolve_project_path("afm_ideal_scan.png"),
                    self._resolve_project_path("../gambar/figure4_vision.png"),
                ]
            )
            image_path = next((path for path in candidate_paths if path.exists()), None)
            if image_path is None:
                self.log("No default image found. Use Load Image to choose one.")
                return

            raw = plt.imread(image_path)
            if raw.ndim == 3:
                raw = raw[..., 0]
            scale_bar_um = 500.0
            scale_bar_px = 140.0
            self.state.default_scale_um_per_px = scale_bar_um / scale_bar_px
            default_width_um = self.state.default_image_width_um
            default_height_um = self.state.default_image_height_um
            if default_width_um and default_height_um:
                width_um = float(default_width_um)
                height_um = float(default_height_um)
            else:
                width_um = float(raw.shape[1])
                height_um = float(raw.shape[0])
            sample, width_um, height_um = load_real_sample_image(
                str(image_path),
                width_um,
                height_um,
            )
            self._load_sample_into_state(sample, width_um, height_um, "default-image", image_path)
            self.log(f"Loaded default image: {image_path}")
            if default_width_um and default_height_um:
                self.log(
                    f"Loaded saved default calibration: {width_um / 1000.0:.3f} mm x "
                    f"{height_um / 1000.0:.3f} mm"
                )
            else:
                self.log(
                    f"Default image scale estimate: {self.state.default_scale_um_per_px:.3f} um/px "
                    f"from 500 um over ~140 px"
                )
        except Exception as e:
            self.log(f"Failed to load default image: {e}")
        finally:
            self._end_action("load_default_image")

    def _clear_reference_data(self):
        self.state.ref_template = None
        self.state.ref_artefacts = []
        self.state.ref_x = 0.0
        self.state.ref_y = 0.0

    def _refresh_current_view(self):
        fov, outside_mask, ix, iy = create_stage_fov(
            self.state.surface_image,
            self.artifact_layer,
            self.state.show_artifact,
            self.state.x,
            self.state.y,
            self.state.fov_width,
            self.state.fov_height,
        )
        self.state.current_fov_raw = fov.copy()
        display_fov, focus_metrics = render_camera_frame(
            fov,
            self.state.camera_resolution,
            outside_mask=outside_mask,
            focus_model=self.state.get_focus_model(),
        )
        self.state.last_blur_diameter_um = focus_metrics["blur_diameter_um"]
        self.state.last_blur_sigma_px = focus_metrics["sigma_px"]
        self.state.last_dof_camera_um = focus_metrics["dof_camera_um"]
        display_fov = rotate_camera_frame(display_fov, self.state.surface_tilt_angle)
        self.img.set_data(display_fov)
        self.img.set_extent([ix, ix + self.state.fov_width, iy + self.state.fov_height, iy])
        self.update_probe_visuals()
        self.fig.canvas.draw_idle()

    def _clamp_to_stage_margin(self, x, y):
        min_x = -self.state.stage_margin_um
        min_y = -self.state.stage_margin_um
        max_x = self.state.width_um + self.state.stage_margin_um - self.state.fov_width
        max_y = self.state.height_um + self.state.stage_margin_um - self.state.fov_height
        return float(np.clip(x, min_x, max_x)), float(np.clip(y, min_y, max_y))

    def move_to_clicked_point(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return
        if getattr(event, "button", None) != 1:
            return
        if self.state.zooming:
            self.log("Wait for zoom to finish before selecting a new scan area.")
            return

        clicked_x = float(event.xdata)
        clicked_y = float(event.ydata)
        target_x = clicked_x - self.state.fov_width / 2.0
        target_y = clicked_y - self.state.fov_height / 2.0
        target_x, target_y = self._clamp_to_stage_margin(target_x, target_y)
        self.state.auto_scan_active = False

        if self.state.pi_mode:
            self._start_smooth_move(target_x, target_y)
        else:
            self.state.target_x = target_x
            self.state.target_y = target_y

        self.log(
            f"AOI selected at X={clicked_x:.1f} um, Y={clicked_y:.1f} um -> "
            f"moving viewport center to X={target_x + self.state.fov_width / 2.0:.1f}, "
            f"Y={target_y + self.state.fov_height / 2.0:.1f}"
        )

    def _append_desired_history(self, axis, value):
        history_attr = f"ai_desired_history_{axis}"
        history = list(getattr(self.state, history_attr))
        history.append(float(value))
        setattr(self.state, history_attr, history[-2:])

    def _predict_compensated_axis(self, desired, axis):
        if not self.ai_mode or not self.ai_compensator:
            return float(desired)

        required_keys = {"model", "scaler_X", "scaler_y"}
        if not required_keys.issubset(self.ai_compensator):
            return float(desired)

        history = getattr(self.state, f"ai_desired_history_{axis}", [float(desired), float(desired)])
        prev_1 = float(history[-1])
        prev_2 = float(history[-2])
        velocity = float(desired - prev_1)
        direction = 0.0 if np.isclose(velocity, 0.0) else float(np.sign(velocity))

        features = np.array([[desired, prev_1, prev_2, direction, velocity]], dtype=float)
        scaler_X = self.ai_compensator["scaler_X"]
        scaler_y = self.ai_compensator["scaler_y"]
        model = self.ai_compensator["model"]

        features_scaled = scaler_X.transform(features)
        prediction_scaled = model.predict(features_scaled).reshape(-1, 1)
        command = float(scaler_y.inverse_transform(prediction_scaled).ravel()[0])

        self._append_desired_history(axis, desired)
        return command

    def load_sample_image(self, event):
        if not self._begin_action("load_sample_image", "Image loading is already running"):
            return
        root = tk.Tk()
        try:
            root.withdraw()
            image_path = filedialog.askopenfilename(
                title="Select Microscope Image",
                filetypes=[
                    ("Image files", "*.png;*.jpg;*.jpeg;*.tif;*.tiff;*.bmp"),
                    ("All files", "*.*"),
                ],
            )
            if not image_path:
                return

            use_scale_bar = messagebox.askyesno(
                "Scale Calibration",
                "Do you want to calibrate using the scale bar embedded in the image?\n\n"
                "Choose Yes to click the two ends of the original scale bar.\n"
                "Choose No to enter the full image width and height manually.",
            )

            if use_scale_bar:
                raw_image = plt.imread(image_path)
                if raw_image.ndim == 3:
                    raw_image = raw_image[..., 0]

                fig, ax = plt.subplots(figsize=(10, 7))
                ax.imshow(raw_image, cmap="gray")
                ax.set_title("Click the LEFT and RIGHT ends of the original scale bar, then close this window")
                points = plt.ginput(2, timeout=-1)
                plt.close(fig)

                if len(points) != 2:
                    self.log("Scale bar calibration cancelled")
                    return

                (x1, y1), (x2, y2) = points
                scale_length_px = float(np.hypot(x2 - x1, y2 - y1))
                scale_length_um = simpledialog.askfloat(
                    "Scale Bar Length",
                    "Enter the real scale bar length (um):",
                    initialvalue=500.0,
                    minvalue=0.001,
                )
                if scale_length_um is None:
                    return

                sample, width_um, height_um = load_real_sample_image_from_scale(
                    image_path,
                    scale_bar_length_um=scale_length_um,
                    scale_bar_length_px=scale_length_px,
                )
                self.log(
                    f"Scale calibration from image bar: {scale_length_um:.3f} um "
                    f"over {scale_length_px:.2f} px"
                )
            else:
                width_mm = simpledialog.askfloat("Sample Width", "Enter physical image width (mm):", initialvalue=2.0, minvalue=0.01)
                if width_mm is None:
                    return
                height_mm = simpledialog.askfloat("Sample Height", "Enter physical image height (mm):", initialvalue=2.0, minvalue=0.01)
                if height_mm is None:
                    return

                sample, width_um, height_um = load_real_sample_image(image_path, width_mm * 1000.0, height_mm * 1000.0)
                self.log(f"Manual calibration: {width_mm:.3f} mm x {height_mm:.3f} mm")

            self._load_sample_into_state(sample, width_um, height_um, "image", image_path)
            self.log(f"Loaded microscope image: {image_path}")
            self.log(f"Calibrated sample size: {self.state.width_um / 1000.0:.3f} mm x {self.state.height_um / 1000.0:.3f} mm")
            self.log("Internal scale: 1 pixel = 1 um after calibration resize")
        except Exception as e:
            self.log(f"Failed to load microscope image: {e}")
        finally:
            root.destroy()
            self._end_action("load_sample_image")

    def save_current_as_default(self, event=None):
        if not self.state.sample_path:
            self.log("Load an image first before saving it as the default.")
            return
        default_path = Path(self.state.sample_path)
        if not default_path.exists():
            self.log(f"Current image path no longer exists: {default_path}")
            return

        self.state.default_image_path = str(default_path)
        self.state.default_image_width_um = float(self.state.width_um)
        self.state.default_image_height_um = float(self.state.height_um)
        if self.persist_default_callback is not None:
            try:
                self.persist_default_callback(
                    {
                        "path": default_path,
                        "width_um": float(self.state.width_um),
                        "height_um": float(self.state.height_um),
                        "scale_um_per_px": float(self.state.default_scale_um_per_px),
                    }
                )
            except Exception as e:
                self.log(f"Failed to save default image setting: {e}")
                return
        self.log(
            f"Saved default image: {default_path} "
            f"with calibration {self.state.width_um / 1000.0:.3f} mm x {self.state.height_um / 1000.0:.3f} mm"
        )

    def _estimate_relocation_offset(self, reference_detections, current_detections):
        candidates = []
        for ref in reference_detections:
            for curr in current_detections:
                if curr["class_name"] != ref["class_name"]:
                    continue
                offset_x = curr["abs_coord"][0] - ref["abs_coord"][0]
                offset_y = curr["abs_coord"][1] - ref["abs_coord"][1]
                candidates.append(
                    {
                        "ref": ref,
                        "curr": curr,
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                    }
                )

        if not candidates:
            return None

        offsets = np.array([[c["offset_x"], c["offset_y"]] for c in candidates], dtype=float)
        median_offset = np.median(offsets, axis=0)

        supporters = []
        for candidate in candidates:
            residual = float(
                np.hypot(
                    candidate["offset_x"] - median_offset[0],
                    candidate["offset_y"] - median_offset[1],
                )
            )
            if residual <= 75.0:
                candidate["residual"] = residual
                supporters.append(candidate)

        if not supporters:
            supporters = candidates
            for candidate in supporters:
                candidate["residual"] = float(
                    np.hypot(
                        candidate["offset_x"] - median_offset[0],
                        candidate["offset_y"] - median_offset[1],
                    )
                )

        offset_x = float(np.mean([c["offset_x"] for c in supporters]))
        offset_y = float(np.mean([c["offset_y"] for c in supporters]))
        best_match = min(supporters, key=lambda item: item["residual"])

        return {
            "offset_x": offset_x,
            "offset_y": offset_y,
            "best_match": best_match,
            "support_count": len(supporters),
        }

    def set_step(self, step):
        self.state.current_step = step
        self.log(f"Step size set to {step} um")

    def move_up(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x, self.state.target_y - self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move up {self.state.current_step} um -> target center Y={new_target_y + self.state.fov_height / 2.0:.1f}")

    def move_down(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x, self.state.target_y + self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move down {self.state.current_step} um -> target center Y={new_target_y + self.state.fov_height / 2.0:.1f}")

    def move_left(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x - self.state.current_step, self.state.target_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move left {self.state.current_step} um -> target center X={new_target_x + self.state.fov_width / 2.0:.1f}")

    def move_right(self, event):
        new_target_x, new_target_y = self._clamp_to_stage_margin(self.state.target_x + self.state.current_step, self.state.target_y)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, new_target_y)
        else:
            self.state.target_x = new_target_x
            self.state.target_y = new_target_y
        self.log(f"Move right {self.state.current_step} um -> target center X={new_target_x + self.state.fov_width / 2.0:.1f}")

    def toggle_pause(self, event):
        self.state.paused = not self.state.paused
        self.buttons["stop"].label.set_text("Motion: OFF" if self.state.paused else "Motion: ON")
        if not self.state.paused:
            self.state.target_x, self.state.target_y = self.state.x, self.state.y
        self.update_title()

    def toggle_pi(self, event):
        self.state.pi_mode = not self.state.pi_mode
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x, self.stage.cmd_y = self.state.target_x, self.state.target_y
            self.buttons["pi"].label.set_text("PI Compensation: ON")
            self.log("PI mode activated")
        else:
            self.buttons["pi"].label.set_text("PI Compensation Mode")
            self.log("PI mode deactivated")
        self.update_title()

    def zoom_in(self, event):
        current_index = self._get_zoom_level_index(self.state.current_zoom_level)
        levels = list(self.state.zoom_levels)
        self._begin_quantized_zoom(levels[min(current_index + 1, len(levels) - 1)])

    def zoom_out(self, event):
        current_index = self._get_zoom_level_index(self.state.current_zoom_level)
        levels = list(self.state.zoom_levels)
        self._begin_quantized_zoom(levels[max(current_index - 1, 0)])

    def move_z_up(self, event):
        self.state.z_stage_position_um = self._clamp_z_stage(self.state.z_stage_position_um + self.state.z_stage_step_um)
        self._refresh_current_view()
        self.log(
            f"Sample Z stage moved to {self.state.z_stage_position_um:+.1f} um "
            f"(probe gap {self.state.get_probe_sample_gap_um():+.1f} um, focus offset {self.state.get_focus_offset_um():+.1f} um)"
        )

    def move_z_down(self, event):
        self.state.z_stage_position_um = self._clamp_z_stage(self.state.z_stage_position_um - self.state.z_stage_step_um)
        self._refresh_current_view()
        self.log(
            f"Sample Z stage moved to {self.state.z_stage_position_um:+.1f} um "
            f"(probe gap {self.state.get_probe_sample_gap_um():+.1f} um, focus offset {self.state.get_focus_offset_um():+.1f} um)"
        )

    def reset_focus(self, event):
        self.state.z_stage_position_um = float(
            self.state.get_effective_camera_stage_position_um() - self.state.focus_z_um
        )
        self._refresh_current_view()
        self.log(
            f"Sample Z stage returned to best focus at {self.state.z_stage_position_um:.1f} um "
            f"for camera/cantilever stage {self.state.get_effective_camera_stage_position_um():.1f} um"
        )

    def show_tip_coord(self, event):
        tip_x, tip_y = self.get_tip()
        if self.state.origin_defined:
            self.log(
                f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um "
                f"(relative to {self.state.origin_label}: "
                f"dX={tip_x - self.state.origin_x:+.2f} um, dY={tip_y - self.state.origin_y:+.2f} um)"
            )
        else:
            self.log(f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um")

    def cycle_scale_bar_length(self, event):
        choices = list(self.SCALE_BAR_CHOICES_UM)
        try:
            current_index = choices.index(float(self.state.scale_bar_total_um))
        except ValueError:
            current_index = 0
        next_value = choices[(current_index + 1) % len(choices)]
        self.state.scale_bar_total_um = float(next_value)
        if "scale_bar" in self.buttons:
            self.buttons["scale_bar"].label.set_text(f"Scale Bar: {int(next_value)} um")
        self.fig.canvas.draw_idle()
        self.log(f"Viewport scale bar length set to {int(next_value)} um")

    def auto_origin_unsupervised(self, event):
        if self.state.current_fov_raw is None or self.state.current_fov_raw.size == 0:
            self.log("No current viewport image available for unsupervised origin detection.")
            return

        fov = self.state.current_fov_raw
        gray = fov[..., 0] if fov.ndim == 3 else fov
        gray_u8 = np.clip(gray, 0, 255).astype(np.uint8)

        corners = cv2.goodFeaturesToTrack(
            gray_u8,
            maxCorners=40,
            qualityLevel=0.01,
            minDistance=20,
            blockSize=7,
        )

        best_point = None
        best_score = -np.inf
        patch_half = max(18, int(self.state.origin_template_half_size // 2))

        if corners is not None:
            for corner in corners.reshape(-1, 2):
                cx = int(round(float(corner[0])))
                cy = int(round(float(corner[1])))
                x0 = max(0, cx - patch_half)
                x1 = min(gray_u8.shape[1], cx + patch_half)
                y0 = max(0, cy - patch_half)
                y1 = min(gray_u8.shape[0], cy + patch_half)
                patch = gray_u8[y0:y1, x0:x1]
                if patch.shape[0] < 12 or patch.shape[1] < 12:
                    continue
                gy, gx = np.gradient(patch.astype(np.float32))
                score = float(np.std(patch) + 0.35 * np.mean(np.hypot(gx, gy)))
                if score > best_score:
                    best_score = score
                    best_point = (cx, cy)

        if best_point is None:
            orb = cv2.ORB_create(nfeatures=100)
            keypoints = orb.detect(gray_u8, None)
            if keypoints:
                strongest = max(keypoints, key=lambda kp: kp.response)
                best_point = (int(round(strongest.pt[0])), int(round(strongest.pt[1])))
                best_score = float(strongest.response)

        if best_point is None:
            self.log("Unsupervised origin detection could not find a distinctive pattern in the current viewport.")
            return

        abs_x = float(self.state.x + best_point[0])
        abs_y = float(self.state.y + best_point[1])
        self.set_origin(abs_x, abs_y, label="Auto Origin")
        self.log(
            f"Unsupervised origin selected at X={abs_x:.1f} um, Y={abs_y:.1f} um "
            f"(distinctiveness score {best_score:.3f})"
        )

    def ml_find_origin(self, event):
        if not self._begin_action("ml_find_origin", "ML origin search is already running"):
            return
        try:
            if self.state.origin_template is None:
                self.log("Set an origin in the viewport first so the supervised ML search has a labeled pattern.")
                return

            template = self.state.origin_template
            surface = self.state.surface_image
            if template.ndim == 3:
                template = template[..., 0]
            if surface.ndim == 3:
                surface = surface[..., 0]

            match = cv2.matchTemplate(surface.astype(np.float32), template.astype(np.float32), cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(match)
            if max_val < 0.40:
                self.log(f"ML origin search confidence too low: {max_val:.3f}")
                return

            tpl_h, tpl_w = template.shape[:2]
            origin_x = float(max_loc[0] + tpl_w / 2.0)
            origin_y = float(max_loc[1] + tpl_h / 2.0)
            self.state.origin_x = origin_x
            self.state.origin_y = origin_y
            self.state.origin_defined = True
            center_target_x, center_target_y = self._clamp_to_stage_margin(
                origin_x - self.state.fov_width / 2.0,
                origin_y - self.state.fov_height / 2.0,
            )
            if self.state.pi_mode:
                self._start_smooth_move(center_target_x, center_target_y)
            else:
                self.state.target_x = center_target_x
                self.state.target_y = center_target_y
            if self.status_callback is not None:
                self.status_callback()
            self.fig.canvas.draw_idle()
            self.log(
                f"ML origin recognized {self.state.origin_label} at "
                f"X={origin_x:.1f} um, Y={origin_y:.1f} um "
                f"(confidence {max_val:.3f})"
            )
        finally:
            self._end_action("ml_find_origin")

    def set_tilt(self, event):
        if self.state.tilting:
            self.log("Tilt adjustment already open")
            return
        self.state.tilting = True
        root = tk.Tk()
        try:
            root.withdraw()
            angle = simpledialog.askfloat(
                "Stage Surface Tilt",
                "Enter stage surface rotation angle (degrees):",
                initialvalue=float(self.state.surface_tilt_angle),
                minvalue=-180,
                maxvalue=180,
            )
            if angle is not None:
                self.state.surface_tilt_angle = float(angle)
                self._refresh_current_view()
                self.log(f"Stage surface tilt set to {self.state.surface_tilt_angle:.1f} degrees")
        finally:
            root.destroy()
            self.state.tilting = False
            self.update_title()

    def clear_trails(self, event):
        self.data.clear()
        self.log("Trails cleared")

    def start_auto_scan(self, event):
        if self.state.auto_scan_active:
            self.log("Auto scan is already running")
            return
        self.state.auto_scan_start_x = self.state.target_x
        self.state.auto_scan_end_x = self.state.target_x + 1000
        self.state.auto_scan_step = 0
        self.state.auto_scan_direction = 1
        self.state.auto_scan_active = True
        self.log(f"Auto scan: {self.state.auto_scan_start_x:.0f} -> {self.state.auto_scan_end_x:.0f}")

    def show_hysteresis_curve(self, event):
        if not self._begin_action("show_hysteresis", "Hysteresis plot is already opening"):
            return
        if self.state.pi_mode and len(self.stage.history_cmd) > 0:
            try:
                self.stage.plot_hysteresis("Hysteresis (Multi-Operator PI Model)")
            finally:
                self._end_action("show_hysteresis")
        else:
            self._end_action("show_hysteresis")
            self.log("Please enable PI mode and move the tip first")

    def start_data_collection(self, event):
        if not self._begin_action("data_collection", "Data collection is already running"):
            return
        self.log("=== Starting data collection ===")
        try:
            from data_collection import DataCollector

            collector = DataCollector(self.stage, self.state, self.get_tip)
            configs = [
                {"start_x": 200, "end_x": 600, "steps": 80, "speed_factor": 1, "label": "Range_400um_Slow"},
                {"start_x": 200, "end_x": 600, "steps": 80, "speed_factor": 2, "label": "Range_400um_Fast"},
                {"start_x": 200, "end_x": 1000, "steps": 120, "speed_factor": 1, "label": "Range_800um_Slow"},
                {"start_x": 200, "end_x": 1000, "steps": 120, "speed_factor": 2, "label": "Range_800um_Fast"},
                {"start_x": 200, "end_x": 1400, "steps": 160, "speed_factor": 1, "label": "Range_1200um_Slow"},
                {"start_x": 200, "end_x": 1400, "steps": 160, "speed_factor": 2, "label": "Range_1200um_Fast"},
            ]
            collector.collect_multi_configurations(configs, base_wait_time=0.05)
            filename = collector.save_all_to_csv()
            collector.plot_collected_data()
            self.log(f"Data collection completed. File saved: {filename}")
        except Exception as e:
            self.log(f"Data collection failed: {e}")
        finally:
            self._end_action("data_collection")

    def save_reference(self, event):
        """Save the current field of view as the relocation reference template."""
        if not self._begin_action("save_reference", "Reference capture is already running"):
            return
        try:
            if self.img is None:
                self.log("Cannot get current image")
                return

            fov = self.state.current_fov_raw if self.state.current_fov_raw is not None else self.img.get_array()
            self.state.ref_template = fov.copy()
            self.state.ref_artefacts = []
            self.state.ref_x = self.state.x
            self.state.ref_y = self.state.y
            self.state.ai_desired_history_x = [self.state.x, self.state.x]
            self.state.ai_desired_history_y = [self.state.y, self.state.y]
            self.log(f"Reference position saved: ({self.state.ref_x:.1f}, {self.state.ref_y:.1f})")
            self.log(f"Saved reference template size: {self.state.ref_template.shape[1]} x {self.state.ref_template.shape[0]} px")
        finally:
            self._end_action("save_reference")

    def remove_sample(self, event):
        """Simulate sample removal by shifting the visible field of view."""
        self.state.sample_removed = True
        dx = random.uniform(-500, 500)
        dy = random.uniform(-500, 500)
        new_x, new_y = self._clamp_to_stage_margin(self.state.x + dx, self.state.y + dy)
        self.state.x = new_x
        self.state.y = new_y
        self.state.target_x = new_x
        self.state.target_y = new_y
        self.log(f"Sample removal simulation: view shifted to ({new_x:.1f}, {new_y:.1f})")

    def relocate(self, event):
        """Relocate using real-image template matching, with AI command compensation when available."""
        if not self._begin_action("relocate", "Relocation is already running"):
            return
        try:
            if self.state.ref_template is None:
                self.log("Please save reference position first")
                return

            self.log("Starting image-based relocation...")

            match = match_reference_template(
                self.state.surface_image,
                self.state.ref_template,
                self.state.x,
                self.state.y,
                half_range=500,
            )
            if match is None:
                self.log("Reference template could not be matched in the search area")
                return

            if match["score"] < 0.35:
                self.log(f"Match confidence too low: {match['score']:.3f}")
                return

            desired_x = float(match["x"])
            desired_y = float(match["y"])
            self.log(f"Matched reference top-left at ({desired_x:.1f}, {desired_y:.1f})")
            self.log(f"Template match score: {match['score']:.3f}")

            if self.ai_compensator is not None and self.ai_mode:
                cmd_x = self._predict_compensated_axis(desired_x, "x")
                cmd_y = self._predict_compensated_axis(desired_y, "y")
                self.log("AI compensation applied to relocation command")
            else:
                cmd_x = float(desired_x)
                cmd_y = float(desired_y)

            cmd_x, cmd_y = self._clamp_to_stage_margin(cmd_x, cmd_y)

            self._start_smooth_move(cmd_x, cmd_y)
            self.state.sample_removed = False
            self.log(f"Relocation complete, moved to ({cmd_x:.1f}, {cmd_y:.1f})")
        finally:
            self._end_action("relocate")

    def toggle_ai_mode(self, event):
        """Toggle AI compensation mode."""
        self.ai_mode = not self.ai_mode
        self.log(f"AI compensation mode: {'ON' if self.ai_mode else 'OFF'}")
