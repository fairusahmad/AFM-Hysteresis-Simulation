import numpy as np


class AFMState:
    def __init__(self, sample, width_um, height_um):
        self.surface_image = sample
        self.sample = self.surface_image
        self.width_um = width_um
        self.height_um = height_um

        self.fov_width = 840
        self.fov_height = 630
        self.x = 200.0
        self.y = 400.0
        self.target_x = self.x
        self.target_y = self.y
        self.stage_margin_um = 2000.0
        self.max_zoom_out_scale = 4.0

        self.step_speed = 5
        self.move_step = 200
        self.current_step = 5
        self.animation_interval_ms = 30
        self.paused = False

        self.zooming = False
        self.zoom_progress = 0
        self.zoom_steps = 20
        self.zoom_direction = 0
        self.zoom_anchor_tip_x = None
        self.zoom_anchor_tip_y = None
        self.zoom_anchor_rel_x = None
        self.zoom_anchor_rel_y = None
        self.zoom_base_width = None
        self.zoom_base_height = None
        self.zoom_center_x = None
        self.zoom_center_y = None
        self.zoom_target_width = None
        self.zoom_target_height = None
        self.current_zoom_level = 1
        self.target_zoom_level = 1
        self.min_zoom_level = 1
        self.max_zoom_level = 10
        self.current_fov_raw = None

        self.surface_tilt_angle = 0.0
        self.probe_tilt_angle = 0.0
        self.tilting = False
        self.pi_mode = False

        self.auto_scan_active = False
        self.auto_scan_step = 0
        self.auto_scan_total_steps = 200
        self.auto_scan_start_x = 0.0
        self.auto_scan_end_x = 0.0
        self.auto_scan_direction = 1

        self.show_artifact = False
        self.sample_source = "synthetic-surface"
        self.sample_path = None
        self.default_image_path = None
        self.default_image_width_um = None
        self.default_image_height_um = None

        self.camera_resolution = (1024, 768)
        self.camera_reference_resolution = (2592, 1944)
        self.sample_view_camera_resolution = (4208, 3120)
        self.camera_mode = "Park FX40 On-Axis Optics"
        self.base_objective_magnification = 10.0
        self.objective_numerical_aperture = 0.21
        self.illumination_wavelength_um = 0.55
        self.sensor_pixel_size_um = 3.45
        self.on_axis_fov_width_um = 840.0
        self.on_axis_fov_height_um = 630.0
        self.sample_view_fov_width_mm = 172.0
        self.sample_view_fov_height_mm = 97.0
        self.z_stage_travel_um = 22000.0
        self.z_stage_position_um = 0.0
        self.focus_z_um = 0.0
        self.z_stage_step_um = 5.0
        self.afm_z_scanner_range_options_um = (15.0, 30.0)
        self.xy_scanner_range_options_um = ((100.0, 100.0), (50.0, 50.0), (5.0, 5.0))
        self.last_blur_diameter_um = 0.0
        self.last_blur_sigma_px = 0.0
        self.last_dof_camera_um = 0.0

        self.default_scale_um_per_px = 1.0
        self.scale_bar_total_um = 200.0
        self.scale_bar_segments = 2
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_label = "Origin"
        self.origin_defined = False
        self.origin_template = None
        self.origin_template_half_size = 48

        self.ref_artefacts = []
        self.ref_template = None
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.sample_removed = False
        self.ai_desired_history_x = [self.x, self.x]
        self.ai_desired_history_y = [self.y, self.y]

        self.smooth_move_active = False
        self.smooth_move_target_x = None
        self.smooth_move_target_y = None
        self.smooth_move_step = 20.0

    def get_optical_zoom_ratio(self):
        return float(np.clip(self.current_zoom_level, self.min_zoom_level, self.max_zoom_level))

    def get_digital_zoom_level(self):
        return int(np.clip(self.current_zoom_level, self.min_zoom_level, self.max_zoom_level))

    def get_current_objective_magnification(self):
        return float(self.base_objective_magnification) * self.get_optical_zoom_ratio()

    def get_fov_for_zoom_level(self, zoom_level):
        zoom_level = int(np.clip(int(round(zoom_level)), self.min_zoom_level, self.max_zoom_level))
        width = max(50, int(round(self.on_axis_fov_width_um / float(zoom_level))))
        height = max(50, int(round(self.on_axis_fov_height_um / float(zoom_level))))
        return width, height

    def get_focus_model(self):
        return {
            "z_position_um": float(self.z_stage_position_um),
            "focus_z_um": float(self.focus_z_um),
            "numerical_aperture": float(self.objective_numerical_aperture),
            "wavelength_um": float(self.illumination_wavelength_um),
            "sensor_pixel_size_um": float(self.sensor_pixel_size_um),
            "objective_magnification": float(self.get_current_objective_magnification()),
            "fov_width_um": float(self.fov_width),
            "fov_height_um": float(self.fov_height),
        }
