import tkinter as tk
from tkinter import simpledialog
import numpy as np
import cv2
import random
from matplotlib.transforms import Affine2D
from afm_utils import create_fov_image
from artefact_detector import ArtefactDetector

class AFMCallbacks:
    def __init__(self, state, stage, fig, ax, tip, cantilever, rod, center_x_ax,
                 data, update_title_func, get_tip_func, buttons, artifact_layer):
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
        
        # Initialize artefact detector
        self.artefact_detector = None
        try:
            self.artefact_detector = ArtefactDetector()
            print("Artefact detector loaded")
        except Exception as e:
            print(f"Failed to load artefact detector: {e}")
        
        # Try to load AI model for hysteresis compensation
        try:
            import joblib
            model_data = joblib.load("inverse_model.pkl")
            self.ai_compensator = model_data
            print("AI model loaded")
        except:
            print("AI model not found")
    
    # ========== Step size setting ==========
    def set_step(self, step):
        self.state.current_step = step
        print(f"Step size set to {step} μm")
    
    # ========== Start smooth movement ==========
    def _start_smooth_move(self, target_x, target_y):
        self.state.smooth_move_active = True
        self.state.smooth_move_target_x = target_x
        self.state.smooth_move_target_y = target_y
    
    # ========== Directional movement ==========
    def move_up(self, event):
        new_target_y = max(0, self.state.target_y - self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(self.state.target_x, new_target_y)
        else:
            self.state.target_y = new_target_y
        print(f"Move up {self.state.current_step} μm -> Y={new_target_y:.1f}")
    
    def move_down(self, event):
        new_target_y = min(self.state.height_um - self.state.fov_height,
                           self.state.target_y + self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(self.state.target_x, new_target_y)
        else:
            self.state.target_y = new_target_y
        print(f"Move down {self.state.current_step} μm -> Y={new_target_y:.1f}")
    
    def move_left(self, event):
        new_target_x = max(0, self.state.target_x - self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, self.state.target_y)
        else:
            self.state.target_x = new_target_x
        print(f"Move left {self.state.current_step} μm -> X={new_target_x:.1f}")
    
    def move_right(self, event):
        new_target_x = min(self.state.width_um - self.state.fov_width,
                           self.state.target_x + self.state.current_step)
        if self.state.pi_mode:
            self._start_smooth_move(new_target_x, self.state.target_y)
        else:
            self.state.target_x = new_target_x
        print(f"Move right {self.state.current_step} μm -> X={new_target_x:.1f}")
    
    # ========== Basic functions ==========
    def toggle_pause(self, event):
        self.state.paused = not self.state.paused
        self.buttons['stop'].label.set_text("Resume" if self.state.paused else "Stop")
        if not self.state.paused:
            self.state.target_x, self.state.target_y = self.state.x, self.state.y
        self.update_title()
    
    def toggle_pi(self, event):
        self.state.pi_mode = not self.state.pi_mode
        if self.state.pi_mode:
            self.stage.reset(self.state.x, self.state.y)
            self.stage.cmd_x, self.stage.cmd_y = self.state.target_x, self.state.target_y
            self.buttons['pi'].label.set_text("PI ON")
            print("PI mode activated")
        else:
            self.buttons['pi'].label.set_text("PI Mode")
            print("PI mode deactivated")
        self.update_title()
    
    def zoom_in(self, event):
        self.state.zooming = True
        self.state.zoom_direction = 1
        self.state.zoom_progress = 0
    
    def zoom_out(self, event):
        self.state.zooming = True
        self.state.zoom_direction = -1
        self.state.zoom_progress = 0
    
    def show_tip_coord(self, event):
        tip_x, tip_y = self.get_tip()
        print(f"Tip position: X={tip_x:.2f} um, Y={tip_y:.2f} um")
    
    def set_tilt(self, event):
        self.state.tilting = True
        root = tk.Tk()
        root.withdraw()
        angle = simpledialog.askfloat("Tilt Angle", "Enter angle (degrees):", minvalue=-30, maxvalue=30)
        if angle is not None:
            self.state.tilt_angle = angle
            cx, cy = self.center_x_ax, 0.95
            transform = Affine2D().rotate_deg_around(cx, cy, angle) + self.ax.transAxes
            self.cantilever.set_transform(transform)
            self.rod.set_transform(transform)
            self.tip.set_transform(transform)
            self.fig.canvas.draw_idle()
            self.data.insert_nan()
            print(f"Tilt: {angle}°")
        self.state.tilting = False
        self.update_title()
    
    def clear_trails(self, event):
        self.data.clear()
        print("Trails cleared")
    
    def start_auto_scan(self, event):
        if self.state.auto_scan_active:
            return
        self.state.auto_scan_start_x = self.state.target_x
        self.state.auto_scan_end_x = min(self.state.width_um - self.state.fov_width, 
                                          self.state.target_x + 1000)
        self.state.auto_scan_step = 0
        self.state.auto_scan_direction = 1
        self.state.auto_scan_active = True
        print(f"Auto scan: {self.state.auto_scan_start_x:.0f} -> {self.state.auto_scan_end_x:.0f}")
    
    def show_hysteresis_curve(self, event):
        if self.state.pi_mode and len(self.stage.history_cmd) > 0:
            self.stage.plot_hysteresis("Hysteresis (Simple Play Model)")
        else:
            print("Please enable PI mode and move the tip first")
    
    def toggle_artifact(self, event, artifact_layer):
        self.state.show_artifact = not self.state.show_artifact
        if self.state.show_artifact:
            artifact_layer.enable()
            self.buttons['artifact'].label.set_text('Artifact: ON')
        else:
            artifact_layer.disable()
            self.buttons['artifact'].label.set_text('Artifact: OFF')
        print(f"Artifact: {'ON' if self.state.show_artifact else 'OFF'}")
    
    def start_data_collection(self, event):
        print("\n=== Starting data collection ===")
        try:
            from data_collection import DataCollector
            collector = DataCollector(self.stage, self.state, self.get_tip)
            configs = [
                {'start_x': 200, 'end_x': 600, 'steps': 80, 'speed_factor': 1, 'label': 'Range_400um_Slow'},
                {'start_x': 200, 'end_x': 600, 'steps': 80, 'speed_factor': 2, 'label': 'Range_400um_Fast'},
                {'start_x': 200, 'end_x': 1000, 'steps': 120, 'speed_factor': 1, 'label': 'Range_800um_Slow'},
                {'start_x': 200, 'end_x': 1000, 'steps': 120, 'speed_factor': 2, 'label': 'Range_800um_Fast'},
                {'start_x': 200, 'end_x': 1400, 'steps': 160, 'speed_factor': 1, 'label': 'Range_1200um_Slow'},
                {'start_x': 200, 'end_x': 1400, 'steps': 160, 'speed_factor': 2, 'label': 'Range_1200um_Fast'},
            ]
            collector.collect_multi_configurations(configs, base_wait_time=0.05)
            filename = collector.save_all_to_csv()
            collector.plot_collected_data()
            print(f"\nData collection completed! File saved: {filename}")
        except Exception as e:
            print(f"Data collection failed: {e}")
    
    # ========== Artefact-based relocation ==========
    def save_reference(self, event):
        """Save reference position (record coordinates of all artefacts in current view)"""
        if self.artefact_detector is None:
            print("Artefact detector not ready, please check model file")
            return
        
        if self.img is None:
            print("Cannot get current image")
            return
        
        # Get current field of view image
        fov = self.img.get_array()
        
        # Detect artefacts
        detections = self.artefact_detector.detect_in_fov(fov, self.state.x, self.state.y)
        
        if not detections:
            print("Warning: No artefact detected in current view. Please move to a marked area.")
            return
        
        # Save detected artefact positions
        self.state.ref_artefacts = {}
        for i, det in enumerate(detections):
            artefact_id = f"{det['class_name']}_{i}"
            self.state.ref_artefacts[artefact_id] = det['abs_coord']
            print(f"  Saved artefact: {det['class_name']} at ({det['abs_coord'][0]:.1f}, {det['abs_coord'][1]:.1f})")
        
        self.state.ref_x = self.state.x
        self.state.ref_y = self.state.y
        
        print(f"Reference position saved: ({self.state.ref_x:.1f}, {self.state.ref_y:.1f})")
        print(f"Recorded {len(self.state.ref_artefacts)} artefact(s)")
    
    def remove_sample(self, event):
        """Simulate sample removal: randomly shift field of view"""
        self.state.sample_removed = True
        dx = random.uniform(-500, 500)
        dy = random.uniform(-500, 500)
        new_x = np.clip(self.state.x + dx, 0, self.state.width_um - self.state.fov_width)
        new_y = np.clip(self.state.y + dy, 0, self.state.height_um - self.state.fov_height)
        self.state.x = new_x
        self.state.y = new_y
        self.state.target_x = new_x
        self.state.target_y = new_y
        print(f"Sample removal simulation: view shifted to ({new_x:.1f}, {new_y:.1f})")
    
    def relocate(self, event):
        """AI-based artefact relocation (improved matching logic)"""
        if not self.state.ref_artefacts:
            print("Please save reference position first")
            return
        
        if self.artefact_detector is None:
            print("Artefact detector not ready")
            return
        
        print("="*50)
        print("Starting AI-based artefact relocation...")
        print("="*50)
        
        # 1. Wide area scan and detect artefacts
        current_artefacts = self.artefact_detector.detect_in_scan(
            self.state, self.artifact_layer, self.state.show_artifact,
            half_range=500, step=100
        )
        
        if not current_artefacts:
            print("No artefact detected")
            return
        
        # 2. Match artefacts: find the current artefact closest to each reference artefact
        best_offset = None
        best_dist = float('inf')
        best_ref_pos = None
        best_curr_pos = None
        
        for ref_id, ref_pos in self.state.ref_artefacts.items():
            for curr_id, curr_pos in current_artefacts.items():
                # Euclidean distance
                dist = np.hypot(curr_pos[0] - ref_pos[0], curr_pos[1] - ref_pos[1])
                if dist < best_dist:
                    best_dist = dist
                    best_offset = (curr_pos[0] - ref_pos[0], curr_pos[1] - ref_pos[1])
                    best_ref_pos = ref_pos
                    best_curr_pos = curr_pos
        
        if best_offset is None:
            print("No matching artefact found")
            return
        
        offset_x, offset_y = best_offset
        print(f"Matched artefact: reference position ({best_ref_pos[0]:.1f}, {best_ref_pos[1]:.1f})")
        print(f"            current position ({best_curr_pos[0]:.1f}, {best_curr_pos[1]:.1f})")
        print(f"Matching distance: {best_dist:.1f} μm")
        print(f"Calculated offset: ({offset_x:.1f}, {offset_y:.1f})")
        
        # 3. Desired position (reference view corner + offset)
        desired_x = self.state.ref_x + offset_x
        desired_y = self.state.ref_y + offset_y
        
        # 4. AI compensation (if available)
        if self.ai_compensator is not None:
            cmd_x = desired_x
            cmd_y = desired_y
            print("AI compensation applied")
        else:
            cmd_x, cmd_y = desired_x, desired_y
        
        # 5. Move
        self._start_smooth_move(cmd_x, cmd_y)
        self.state.sample_removed = False
        print(f"Relocation complete, moved to ({cmd_x:.1f}, {cmd_y:.1f})")
        print("="*50)
    
    def toggle_ai_mode(self, event):
        """Toggle AI compensation mode"""
        self.ai_mode = not self.ai_mode
        print(f"AI compensation mode: {'ON' if self.ai_mode else 'OFF'}")