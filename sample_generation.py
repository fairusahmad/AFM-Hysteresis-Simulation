import numpy as np
import cv2

# =========================
# 1. 创建基础样品
# =========================

width_um = 2500
height_um = 2000

sample = np.zeros((height_um, width_um), dtype=np.float32)

grid_size = 8
spacing = 2

A_height = 300
A_width = 200

start_x = 1000
start_y = 1500
middle_y = start_y - A_height // 2

for y0 in range(start_y - A_height, start_y, grid_size + spacing):

    ratio = (start_y - y0) / A_height
    left_x = int(start_x + ratio * (A_width // 2))
    right_x = int(start_x + A_width - ratio * (A_width // 2))

    for x0 in range(start_x, start_x + A_width, grid_size + spacing):

        if abs(x0 - left_x) < grid_size:
            cv2.rectangle(sample, (x0, y0),
                          (x0 + grid_size, y0 + grid_size),
                          255, -1)

        if abs(x0 - right_x) < grid_size:
            cv2.rectangle(sample, (x0, y0),
                          (x0 + grid_size, y0 + grid_size),
                          255, -1)

        if abs(y0 - middle_y) < grid_size:
            cv2.rectangle(sample, (x0, y0),
                          (x0 + grid_size, y0 + grid_size),
                          255, -1)

# 背景

bg_square = 8
bg_spacing = 2

for y0 in range(0, height_um, bg_square + bg_spacing):
    for x0 in range(0, width_um, bg_square + bg_spacing):
        if sample[y0, x0] == 0:
            cv2.rectangle(sample,
                          (x0, y0),
                          (x0 + bg_square, y0 + bg_square),
                          180, -1)

noise = np.random.normal(0, 10, sample.shape)
sample += noise
sample = cv2.normalize(sample, None, 0, 255, cv2.NORM_MINMAX)
sample = sample.astype(np.uint8)

# =========================
# 2. Artifact 图层（带唯一 ID）
# =========================

class ArtifactLayer:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.layer = np.zeros((height, width), dtype=np.uint8)
        self.enabled = True
        self.artefact_list = []  # 存储 (id, type, x, y, size)
        
    def add_circle(self, center_x, center_y, radius, intensity=255, label=None, artefact_id=None):
        cv2.circle(self.layer, (center_x, center_y), radius, intensity, -1)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.layer, label, (center_x - 15, center_y - radius - 5), 
                        font, 0.5, intensity, 1)
        if artefact_id is not None:
            self.artefact_list.append({
                'id': artefact_id, 'type': 'circle', 'x': center_x, 'y': center_y, 'size': radius*2
            })
        
    def add_square(self, center_x, center_y, size, intensity=255, label=None, artefact_id=None):
        half = size // 2
        cv2.rectangle(self.layer, 
                     (center_x - half, center_y - half),
                     (center_x + half, center_y + half),
                     intensity, -1)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.layer, label, (center_x - 15, center_y - half - 5), 
                        font, 0.5, intensity, 1)
        if artefact_id is not None:
            self.artefact_list.append({
                'id': artefact_id, 'type': 'square', 'x': center_x, 'y': center_y, 'size': size
            })
        
    def add_cross(self, center_x, center_y, size, intensity=255, label=None, artefact_id=None):
        half = size // 2
        cv2.line(self.layer, (center_x - half, center_y), (center_x + half, center_y), intensity, 3)
        cv2.line(self.layer, (center_x, center_y - half), (center_x, center_y + half), intensity, 3)
        if label:
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(self.layer, label, (center_x - 20, center_y - half - 5), 
                        font, 0.5, intensity, 1)
        if artefact_id is not None:
            self.artefact_list.append({
                'id': artefact_id, 'type': 'cross', 'x': center_x, 'y': center_y, 'size': size
            })
    
    def add_fiducial_marker(self, center_x, center_y, size=40, intensity=255, artefact_id=None):
        cv2.circle(self.layer, (center_x, center_y), size//2, intensity, 2)
        cv2.circle(self.layer, (center_x, center_y), size//4, intensity, -1)
        cv2.line(self.layer, (center_x - size//2, center_y), (center_x + size//2, center_y), intensity, 2)
        cv2.line(self.layer, (center_x, center_y - size//2), (center_x, center_y + size//2), intensity, 2)
        if artefact_id is not None:
            self.artefact_list.append({
                'id': artefact_id, 'type': 'fiducial', 'x': center_x, 'y': center_y, 'size': size
            })
    
    def add_scale_bar(self, x, y, length_um, pixel_per_um=1, intensity=255):
        length_px = int(length_um * pixel_per_um)
        cv2.line(self.layer, (x, y), (x + length_px, y), intensity, 3)
        cv2.line(self.layer, (x, y - 5), (x, y + 5), intensity, 2)
        cv2.line(self.layer, (x + length_px, y - 5), (x + length_px, y + 5), intensity, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(self.layer, f"{length_um} um", (x + length_px//2 - 20, y - 8), font, 0.5, intensity, 1)
    
    def add_grid(self, start_x, start_y, spacing, size, intensity=150):
        for x in range(start_x, self.width, spacing):
            cv2.line(self.layer, (x, start_y), (x, start_y + size), intensity, 1)
        for y in range(start_y, self.height, spacing):
            cv2.line(self.layer, (start_x, y), (start_x + size, y), intensity, 1)
            
    def clear(self):
        self.layer.fill(0)
        self.artefact_list = []
        
    def enable(self):
        self.enabled = True
        
    def disable(self):
        self.enabled = False
        
    def get_display(self):
        return self.layer if self.enabled else np.zeros_like(self.layer)

# =========================
# 3. 创建默认痕迹（带唯一 ID）
# =========================
artifact_layer = ArtifactLayer(width_um, height_um)

# 原点标记 (ID 0)
artifact_layer.add_cross(100, 100, 60, 220, label="Origin", artefact_id=0)

# 基准标记 (ID 1, 2)
artifact_layer.add_fiducial_marker(500, 500, 50, 210, artefact_id=1)
artifact_layer.add_fiducial_marker(2000, 1500, 50, 210, artefact_id=2)

# 标记点 (ID 3, 4, 5)
artifact_layer.add_circle(800, 800, 25, 200, label="A1", artefact_id=3)
artifact_layer.add_square(1200, 1200, 40, 200, label="B2", artefact_id=4)
artifact_layer.add_cross(1800, 600, 55, 210, label="C3", artefact_id=5)

# 比例尺（无 ID）
artifact_layer.add_scale_bar(100, 1850, 200, 1, 200)

# 随机颗粒（无 ID）
for _ in range(40):
    x = np.random.randint(0, width_um)
    y = np.random.randint(0, height_um)
    r = np.random.randint(3, 10)
    artifact_layer.add_circle(x, y, r, np.random.randint(150, 230))