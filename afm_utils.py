import cv2
import numpy as np


_RADIAL_MASK_CACHE = {}


def get_tip_position(tip, ax):
    vertices = tip.get_xy()
    tip_vertex_axes = vertices[2]
    tip_transform = tip.get_transform()
    tip_vertex_display = tip_transform.transform(tip_vertex_axes)
    tip_vertex_data = ax.transData.inverted().transform(tip_vertex_display)
    return tip_vertex_data[0], tip_vertex_data[1]


def update_title(ax, fig, pi_mode, target_x, tip_x):
    if pi_mode:
        error = target_x - tip_x
        ax.set_title(f"Hysteresis Error: {error:+.1f} um", fontsize=10)
    else:
        ax.set_title("Linear Mode", fontsize=10)
    fig.canvas.draw_idle()


def get_scale_bar_geometry(x_origin, y_origin, fov_width, fov_height, total_length_um, segments=2):
    total_length_um = float(max(total_length_um, 1.0))
    segments = max(int(segments), 1)
    segment_length_um = total_length_um / segments

    margin_x = fov_width * 0.06
    margin_y = fov_height * 0.08
    x_end = x_origin + fov_width - margin_x
    x_start = x_end - total_length_um
    y_bar = y_origin + fov_height - margin_y

    segment_bounds = []
    for index in range(segments):
        seg_start = x_start + index * segment_length_um
        seg_end = seg_start + segment_length_um
        segment_bounds.append((seg_start, seg_end))

    text_x = (x_start + x_end) / 2.0
    text_y = y_bar - fov_height * 0.035
    return {
        "segments": segment_bounds,
        "text_pos": (text_x, text_y),
        "label": f"{int(round(total_length_um))} um",
        "y": y_bar,
    }


def create_stage_fov(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height):
    x0 = int(round(x))
    y0 = int(round(y))
    width = max(int(round(fov_width)), 1)
    height = max(int(round(fov_height)), 1)

    fov = np.zeros((height, width), dtype=sample.dtype)
    outside_mask = np.ones((height, width), dtype=bool)

    sample_h, sample_w = sample.shape[:2]
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(sample_w, x0 + width)
    src_y1 = min(sample_h, y0 + height)

    if src_x1 > src_x0 and src_y1 > src_y0:
        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)
        fov[dst_y0:dst_y1, dst_x0:dst_x1] = sample[src_y0:src_y1, src_x0:src_x1]
        outside_mask[dst_y0:dst_y1, dst_x0:dst_x1] = False

        if show_artifact and artifact_layer is not None:
            artifact = artifact_layer.get_display()[src_y0:src_y1, src_x0:src_x1]
            fov[dst_y0:dst_y1, dst_x0:dst_x1] = np.maximum(
                fov[dst_y0:dst_y1, dst_x0:dst_x1],
                artifact,
            )

    return fov, outside_mask, x0, y0


def create_fov_image(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height):
    fov, _, ix, iy = create_stage_fov(sample, artifact_layer, show_artifact, x, y, fov_width, fov_height)
    return fov, ix, iy


def render_camera_frame(fov, camera_resolution=None, outside_mask=None, outside_color=(155, 24, 24)):
    if fov.size == 0:
        return fov

    if camera_resolution is not None:
        target_w, target_h = camera_resolution
        if target_w > 0 and target_h > 0 and (fov.shape[1] != target_w or fov.shape[0] != target_h):
            interpolation = cv2.INTER_AREA if (target_w < fov.shape[1] or target_h < fov.shape[0]) else cv2.INTER_CUBIC
            fov = cv2.resize(fov, (int(target_w), int(target_h)), interpolation=interpolation)
            if outside_mask is not None:
                outside_mask = cv2.resize(
                    outside_mask.astype(np.uint8),
                    (int(target_w), int(target_h)),
                    interpolation=cv2.INTER_NEAREST,
                ) > 0

    blurred = apply_radial_focus_blur(fov)
    return _apply_outside_color(blurred, outside_mask, outside_color)


def rotate_camera_frame(frame, angle_deg, fill_color=(155, 24, 24)):
    angle_deg = float(angle_deg)
    if frame.size == 0 or np.isclose(angle_deg, 0.0):
        return frame

    height, width = frame.shape[:2]
    center = (width / 2.0, height / 2.0)
    rotation = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    border_value = fill_color if frame.ndim == 3 else fill_color[0]
    return cv2.warpAffine(
        frame,
        rotation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )


def _apply_outside_color(fov, outside_mask, outside_color):
    if fov.ndim == 2:
        display = np.repeat(fov[..., None], 3, axis=2)
    else:
        display = fov.copy()

    if outside_mask is not None and np.any(outside_mask):
        display[outside_mask] = np.array(outside_color, dtype=display.dtype)
    return display


def apply_radial_focus_blur(fov, focus_radius_ratio=0.28, feather_ratio=0.32, blur_kernel=25):
    if fov.size == 0:
        return fov

    if blur_kernel % 2 == 0:
        blur_kernel += 1

    height, width = fov.shape[:2]
    cache_key = (height, width, float(focus_radius_ratio), float(feather_ratio))
    blur_weight = _RADIAL_MASK_CACHE.get(cache_key)
    if blur_weight is None:
        yy, xx = np.indices((height, width))
        cx = (width - 1) / 2.0
        cy = (height - 1) / 2.0
        radius_scale = max(min(width, height) / 2.0, 1.0)
        radius = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / radius_scale

        focus_radius = max(0.05, focus_radius_ratio)
        sigma = max(0.08, feather_ratio)
        radial_distance = np.clip(radius - focus_radius, 0.0, None)
        blur_weight = 1.0 - np.exp(-(radial_distance ** 2) / (2.0 * sigma ** 2))
        blur_weight = np.clip(blur_weight * 0.55, 0.0, 0.55).astype(np.float32)
        _RADIAL_MASK_CACHE[cache_key] = blur_weight

    blurred = cv2.GaussianBlur(fov, (blur_kernel, blur_kernel), sigmaX=0.0, sigmaY=0.0)
    source = fov.astype(np.float32)
    softened = blurred.astype(np.float32)

    if fov.ndim == 2:
        blended = source * (1.0 - blur_weight) + softened * blur_weight
    else:
        weight = blur_weight[..., None]
        blended = source * (1.0 - weight) + softened * weight

    return np.clip(blended, 0, 255).astype(fov.dtype)
