class AFMState:
    def __init__(self, sample, width_um, height_um):
        self.surface_image = sample
        self.sample = self.surface_image
        self.width_um = width_um
        self.height_um = height_um
        
        self.fov_width = 842
        self.fov_height = 631
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
        self.camera_resolution = (1024, 1024)
        self.camera_reference_resolution = (2592, 1944)
        self.camera_mode = "Fixed Live Camera"
        self.default_scale_um_per_px = 1.0
        self.scale_bar_total_um = 200.0
        self.scale_bar_segments = 2
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_label = "Origin"
        self.origin_defined = False
        self.origin_template = None
        self.origin_template_half_size = 48
        
        # 重定位相关（基于 artefact）
        self.ref_artefacts = []
        self.ref_template = None
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.sample_removed = False
        self.ai_desired_history_x = [self.x, self.x]
        self.ai_desired_history_y = [self.y, self.y]
        
        # 平滑移动相关
        self.smooth_move_active = False
        self.smooth_move_target_x = None
        self.smooth_move_target_y = None
        self.smooth_move_step = 20.0
