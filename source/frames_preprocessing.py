from pathlib import Path

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from ultralytics.utils import ops

from utils.util import set_yolo_model
from utils.roi_polygon_selector import ROIPolygonSelector


def segment_frame(segmentation: Results, image: np.array,
                  left_shoulder: np.array, right_shoulder: np.array) -> np.array:
    mask = segmentation.masks.data.numpy()[0]
    y_nz, x_nz = np.nonzero(mask)  # координаты ненулевых пикселей
    mask_nz = mask[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]  # маска ненулевых пикселей
    segmented = cv2.bitwise_and(  # применяем маску к исходному изображению
        image, image, mask=cv2.resize(mask_nz, image.shape[:-1][::-1]).astype('uint8'))
    cv2.imwrite('test.png', segmented)
    return cv2.cvtColor(segmented[:max(left_shoulder[1], right_shoulder[1]),
                        min(left_shoulder[0], right_shoulder[0]):max(left_shoulder[0], right_shoulder[0])],
                        cv2.COLOR_BGR2GRAY)


def main(data_path: Path):
    yolo_seg: YOLO = set_yolo_model(yolo_model='s', yolo_class='seg', task='segment')
    yolo_pose: YOLO = set_yolo_model(yolo_model='s', yolo_class='pose', task='pose')
    for directory in data_path.glob('*'):
        if not directory.is_dir():
            continue
        for image in directory.glob('*.png'):
            posed: Results = yolo_pose.predict(image, classes=[0], verbose=False)[0]
            for pose_det in posed:
                x1, y1, x2, y2 = pose_det.boxes.numpy().xyxy[0].astype(int)
                human_frame = pose_det.orig_img[y1:y2, x1:x2, :]
                segmented: Results = yolo_seg.predict(human_frame, classes=[0], verbose=False)[0]
                cv2.imwrite('test3.png', segmented.plot())
                segmented_frame = segment_frame(segmented, human_frame)


if __name__ == '__main__':
    main((Path.cwd().parents[0] / 'resources' / 'data' / 'raw'))
