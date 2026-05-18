"""
image_matching.py - 大范围扫描与模板匹配工具
"""

import numpy as np
import cv2
from afm_utils import create_fov_image

def scan_large_area(state, artifact_layer, show_artifact, half_range=500, step=100):
    """
    在当前位置周围扫描，拼接成大图
    返回 (big_image, top_left_x, top_left_y) 大图对应的实际坐标范围
    """
    start_x = max(0, state.x - half_range)
    end_x = min(state.width_um - state.fov_width, state.x + half_range)
    start_y = max(0, state.y - half_range)
    end_y = min(state.height_um - state.fov_height, state.y + half_range)
    
    nx = int((end_x - start_x) / step) + 1
    ny = int((end_y - start_y) / step) + 1
    
    tiles = []
    positions = []  # 每个图块左上角的实际坐标
    for i in range(nx):
        row_tiles = []
        row_positions = []
        for j in range(ny):
            cx = start_x + i * step
            cy = start_y + j * step
            fov, _, _ = create_fov_image(
                state.sample, artifact_layer, show_artifact,
                cx, cy, state.fov_width, state.fov_height
            )
            row_tiles.append(fov)
            row_positions.append((cx, cy))
        if row_tiles:
            row_img = np.hstack(row_tiles)
            tiles.append(row_img)
            positions.append(row_positions)
    
    if not tiles:
        return None, None, None
    
    big_img = np.vstack(tiles)
    # 大图左上角实际坐标 = (start_x, start_y)
    return big_img, start_x, start_y

def find_template(big_image, template, method=cv2.TM_CCOEFF_NORMED):
    """
    在大图中匹配模板，返回匹配点在大图中的像素坐标，以及匹配度
    """
    if big_image is None or template is None:
        return None, None, 0
    # 确保图像为 uint8 灰度
    if big_image.dtype != np.uint8:
        big_image = cv2.normalize(big_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if template.dtype != np.uint8:
        template = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    result = cv2.matchTemplate(big_image, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    # max_loc 是匹配区域左上角在大图中的像素坐标
    return max_loc[0], max_loc[1], max_val