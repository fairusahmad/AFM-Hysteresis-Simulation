import numpy as np
from afm_utils import create_fov_image, update_title

class AFMAnimation:
    def __init__(self, state, stage, data, ax, img, ideal_line, hyst_line, 
                 artifact_layer, fig, get_tip_func):
        self.state = state
        self.stage = stage
        self.data = data
        self.ax = ax
        self.img = img
        self.ideal_line = ideal_line
        self.hyst_line = hyst_line
        self.artifact_layer = artifact_layer
        self.fig = fig
        self.get_tip = get_tip_func
    
    def update(self, frame):
        # 自动扫描
        if self.state.auto_scan_active and not self.state.paused and not self.state.zooming:
            frac = self.state.auto_scan_step / self.state.auto_scan_total_steps
            if self.state.auto_scan_direction == 1:
                self.state.target_x = self.state.auto_scan_start_x + (self.state.auto_scan_end_x - self.state.auto_scan_start_x) * frac
            else:
                self.state.target_x = self.state.auto_scan_end_x - (self.state.auto_scan_end_x - self.state.auto_scan_start_x) * frac
            self.state.auto_scan_step += 1
            if self.state.auto_scan_step > self.state.auto_scan_total_steps:
                if self.state.auto_scan_direction == 1:
                    self.state.auto_scan_direction = -1
                    self.state.auto_scan_step = 0
                else:
                    self.state.auto_scan_active = False
        
        # 平滑移动处理
        if self.state.smooth_move_active and not self.state.paused and not self.state.zooming:
            dx = self.state.smooth_move_target_x - self.state.target_x
            dy = self.state.smooth_move_target_y - self.state.target_y
            step = self.state.smooth_move_step
            
            if abs(dx) > step:
                self.state.target_x += step if dx > 0 else -step
            else:
                self.state.target_x = self.state.smooth_move_target_x
            
            if abs(dy) > step:
                self.state.target_y += step if dy > 0 else -step
            else:
                self.state.target_y = self.state.smooth_move_target_y
            
            if (self.state.target_x == self.state.smooth_move_target_x and 
                self.state.target_y == self.state.smooth_move_target_y):
                self.state.smooth_move_active = False
        
        # 位置更新
        if not self.state.paused and not self.state.zooming:
            if self.state.pi_mode:
                x_new, y_new = self.stage.move_to(self.state.target_x, self.state.target_y)
                self.state.x = x_new
                self.state.y = y_new
            else:
                if self.state.x < self.state.target_x:
                    self.state.x = min(self.state.x + self.state.step_speed, self.state.target_x)
                elif self.state.x > self.state.target_x:
                    self.state.x = max(self.state.x - self.state.step_speed, self.state.target_x)
                if self.state.y < self.state.target_y:
                    self.state.y = min(self.state.y + self.state.step_speed, self.state.target_y)
                elif self.state.y > self.state.target_y:
                    self.state.y = max(self.state.y - self.state.step_speed, self.state.target_y)
        
        # 缩放
        if self.state.zooming:
            self._handle_zoom()
            return self.img, self.ideal_line, self.hyst_line
        
        # 视野更新
        ix = int(np.clip(self.state.x, 0, self.state.width_um - self.state.fov_width))
        iy = int(np.clip(self.state.y, 0, self.state.height_um - self.state.fov_height))
        fov = self.state.sample[iy:iy+self.state.fov_height, ix:ix+self.state.fov_width].copy()
        
        if self.state.show_artifact and self.artifact_layer is not None:
            artifact_fov = self.artifact_layer.get_display()[iy:iy+self.state.fov_height, ix:ix+self.state.fov_width]
            fov = np.maximum(fov, artifact_fov)
        
        self.img.set_data(fov)
        self.img.set_extent([ix, ix+self.state.fov_width, iy+self.state.fov_height, iy])
        
        # 记录轨迹
        if not self.state.tilting:
            tip_x, tip_y = self.get_tip()
            if self.state.pi_mode:
                self.data.add_hyst(tip_x, tip_y, self.state.target_x, self.state.target_y)
            else:
                self.data.add_ideal(tip_x, tip_y, self.state.target_x, self.state.target_y)
        
        # 更新线条和标题
        self.ideal_line.set_data(self.data.ideal_tip_x, self.data.ideal_tip_y)
        self.hyst_line.set_data(self.data.hyst_tip_x, self.data.hyst_tip_y)
        
        tip_x, tip_y = self.get_tip()
        update_title(self.ax, self.fig, self.state.pi_mode, self.state.target_x, tip_x)
        
        return self.img, self.ideal_line, self.hyst_line
    
    def _handle_zoom(self):
        self.state.zoom_progress += 1
        alpha = self.state.zoom_progress / self.state.zoom_steps
        scale = 1 - 0.5 * alpha if self.state.zoom_direction == 1 else 1 + 0.5 * alpha
        
        tip_x, tip_y = self.get_tip()
        rel_x = (tip_x - self.state.x) / self.state.fov_width
        rel_y = (tip_y - self.state.y) / self.state.fov_height
        new_width = max(50, int(self.state.fov_width * scale))
        new_height = max(50, int(self.state.fov_height * scale))
        x_new = tip_x - rel_x * new_width
        y_new = tip_y - rel_y * new_height
        x_new = np.clip(x_new, 0, self.state.width_um - new_width)
        y_new = np.clip(y_new, 0, self.state.height_um - new_height)
        
        ix, iy = int(x_new), int(y_new)
        fov = self.state.sample[iy:iy+new_height, ix:ix+new_width]
        self.img.set_data(fov)
        self.img.set_extent([x_new, x_new+new_width, y_new+new_height, y_new])
        
        if self.state.zoom_progress >= self.state.zoom_steps:
            self.state.zooming = False
            self.state.zoom_progress = 0
            self.state.x, self.state.y = x_new, y_new
            self.state.fov_width, self.state.fov_height = new_width, new_height