"""在原图上绘制观测角点（圆点）和重投影角点（青色十字）。"""

import cv2
import numpy as np


COLOR_BGR = {"blue": (255, 160, 40), "red": (40, 90, 255), "neutral": (200, 200, 200), "purple": (220, 70, 220)}


def draw_detections(frame, detections, title, projections=None):
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (20, 20, 20), -1)
    cv2.putText(image, title, (15, 29), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    for index, detection in enumerate(detections):
        points = np.rint(detection["corners_px"]).astype(np.int32)
        color = COLOR_BGR[detection["color"]]
        cv2.polylines(image, [points], True, color, 2, cv2.LINE_AA)
        for number, point in enumerate(points):
            xy = tuple(map(int, point))
            cv2.circle(image, xy, 3, (0, 255, 0), -1, cv2.LINE_AA)
            cv2.putText(image, str(number), (xy[0] + 4, xy[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)
        label = f"{index}: {detection['color']}/{detection['pattern']} {detection['confidence']:.2f}"
        x = max(2, min(image.shape[1] - 260, int(points[:, 0].min())))
        y = max(60, min(image.shape[0] - 40, int(points[:, 1].min()) - 36))
        label_bottom = y + 20 if "reprojection_error_px" in detection else y + 4
        cv2.rectangle(image, (x - 2, y - 15), (min(image.shape[1] - 1, x + 270), label_bottom), (12, 12, 12), -1)
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if "reprojection_error_px" in detection:
            error = detection["reprojection_error_px"]
            if error is None:
                detail = "PnP failed"
            else:
                z = np.mean(np.asarray(detection["corners_camera_m"])[:, 2])
                detail = f"RMS={error:.2f}px Z={z:.2f}m"
            cv2.putText(image, detail, (x, y + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if projections is not None and projections[index] is not None:
            for point in np.rint(projections[index]).astype(np.int32):
                cv2.drawMarker(image, tuple(map(int, point)), (255, 255, 0), cv2.MARKER_CROSS, 9, 1, cv2.LINE_AA)
    return image
