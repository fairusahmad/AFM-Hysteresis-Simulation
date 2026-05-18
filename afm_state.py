class AFMState:
    def __init__(self, sample, width_um, height_um):
        self.sample = sample
        self.width_um = width_um
        self.height_um = height_um
        
        self.fov_width = 842
        self.fov_height = 631
        self.x = 200.0
        self.y = 400.0
        self.target_x = self.x
        self.target_y = self.y
        
        self.step_speed = 5
        self.move_step = 200
        self.current_step = 5
        self.paused = False
        
        self.zooming = False
        self.zoom_progress = 0
        self.zoom_steps = 20
        self.zoom_direction = 0
        
        self.tilt_angle = 0
        self.tilting = False
        self.pi_mode = False
        
        self.auto_scan_active = False
        self.auto_scan_step = 0
        self.auto_scan_total_steps = 200
        self.auto_scan_start_x = 0.0
        self.auto_scan_end_x = 0.0
        self.auto_scan_direction = 1
        
        self.show_artifact = True
        
        # 重定位相关（基于 artefact）
        self.ref_artefacts = {}  # artefact_id -> (x, y)
        self.ref_x = 0.0
        self.ref_y = 0.0
        self.sample_removed = False
        
        # 平滑移动相关
        self.smooth_move_active = False
        self.smooth_move_target_x = None
        self.smooth_move_target_y = None
        self.smooth_move_step = 2.0