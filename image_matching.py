"""
image_matching.py - Large-area scan and real-image relocation matching tools
"""

import cv2
import numpy as np


def find_template(search_image, template, method=cv2.TM_CCOEFF_NORMED):
    """Return template match top-left coordinates and score inside a search image."""
    if search_image is None or template is None:
        return None, None, 0.0

    if search_image.dtype != np.uint8:
        search_image = cv2.normalize(search_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if template.dtype != np.uint8:
        template = cv2.normalize(template, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    result = cv2.matchTemplate(search_image, template, method)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    return int(max_loc[0]), int(max_loc[1]), float(max_val)


def match_reference_template(sample, template, center_x, center_y, half_range=600):
    """
    Search around the current position for a previously saved reference FOV.
    Returns the best matching FOV top-left coordinate and score.
    """
    if sample is None or template is None:
        return None

    sample_h, sample_w = sample.shape[:2]
    template_h, template_w = template.shape[:2]
    if template_h >= sample_h or template_w >= sample_w:
        return None

    start_x = int(max(0, center_x - half_range))
    start_y = int(max(0, center_y - half_range))
    end_x = int(min(sample_w - template_w, center_x + half_range))
    end_y = int(min(sample_h - template_h, center_y + half_range))

    if end_x < start_x or end_y < start_y:
        return None

    search_image = sample[start_y : end_y + template_h, start_x : end_x + template_w]
    match_x, match_y, score = find_template(search_image, template)
    if match_x is None or match_y is None:
        return None

    return {
        "x": start_x + match_x,
        "y": start_y + match_y,
        "score": score,
        "search_origin": (start_x, start_y),
    }
