from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from tqdm import tqdm

from utils.util import set_yolo_model
from utils.util import log_trace


class Preprocessor:

    def __init__(self):
        """
        Предобработка изображений для обучения.
        """
        self.yolo_seg: YOLO = set_yolo_model(yolo_model='s', yolo_class='seg', task='segment')
        self.yolo_pose: YOLO = set_yolo_model(yolo_model='s', yolo_class='pose', task='pose')
        self.yolo_seg.predict(np.random.random((100, 100, 3)), verbose=False)  # тестово запускаем
        self.yolo_pose.predict(np.random.random((100, 100, 3)), verbose=False)

    def segment_crop_frame(self, pose_detection: Results, conf_kpts: float = 0.5) -> np.array:
        """
        Сегментация и обрезка изображения для вычленения верхней части тела человека.
        :param pose_detection: Результат детекции позы человека.
        :param conf_kpts: Conf для ключевых точек.
        :return: Кропнутое и загрейскейленное изображение верхней части тела человека.
        """

        def get_segmented(segmentation: Results, frame: np.array) -> np.array:
            """
            Применение сегментационной маски к изображению.
            :return: Сегментированное изображение.
            """
            mask = segmentation.masks.data.numpy()[0]  # берем маску
            y_nz, x_nz = np.nonzero(mask)  # координаты ненулевых пикселей
            mask_nz = mask[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]  # маска ненулевых пикселей
            return cv2.bitwise_and(  # применяем маску к исходному изображению
                frame, frame, mask=cv2.resize(mask_nz, frame.shape[:-1][::-1]).astype('uint8'))

        def crop_segmented(segmented_frame: np.array, keypoints: np.array, tl_coords) -> np.array:
            """
            Обрезание сегментированного изображения для вычленения верхней части тела человека.
            :param segmented_frame: Сегментированное изображение.
            :param keypoints: Ключевые точки тела человека.
            :param tl_coords: Верхние левые координаты ббокса человека.
            :return: Кропнутое изображение.
            """
            upper_keypoints = keypoints[:7]  # берем верхние ключевые точки
            # фильтруем их, чтобы они были не нулевыми + чтобы conf был больше порогового
            filtered_keypoints = (
                upper_keypoints[
                    (upper_keypoints[:, 0] > 0) | (upper_keypoints[:, 1] > 0) | (upper_keypoints[:, 2] > conf_kpts)
                    ][:, :-1].astype(int))
            if filtered_keypoints.shape[0] < 2:
                return None
            # берем крайние левую и правую точки (это могут быть как лицевые точки, так и плечи)
            leftmost, rightmost = np.array(
                [min(filtered_keypoints[:, 0]), max(filtered_keypoints[:, 0])]) - tl_coords[0]  # + в относительные
            lowest = max(filtered_keypoints[:, 1]) - tl_coords[1]  # нижняя точка (также не обязательно плечи)
            return segmented_frame[:lowest, leftmost:rightmost]  # обрезаем

        x1, y1, x2, y2 = pose_detection.boxes.numpy().xyxy[0].astype(int)  # берем координаты ббокса
        human_frame = pose_detection.orig_img[y1:y2, x1:x2, :]  # вычленяем человека из кадра
        seg_result: Results = self.yolo_seg.predict(
            human_frame, classes=[0], verbose=False)[0]  # применяем модель для сегментации
        if seg_result.masks is None:
            return None
        segmented = get_segmented(seg_result, human_frame)  # сегментируем кадр
        cropped = crop_segmented(  # обрезаем сегментированный кадр по верхней части тела человека
            segmented, pose_detection.keypoints.data.numpy()[0], np.array([x1, y1]))
        if cropped is None:
            return None
        return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    @log_trace
    def preprocess_data(self, data_path: Path) -> None:
        """
        Предобработка данных для обучения:
        - находим людей в кадре;
        - кропаем людей, вычленяя верхнюю часть тела;
        - переводим в грейскейлевый формат;
        - сохраняем обработанные изображения.
        :param data_path: Путь до директории с данными в формате '../data/1/*.png', '../data/2/*.png', ...
        :return: None.
        """
        save_path: Path = data_path.parent / 'preprocessed'  # путь для сохранения обработанных изображений
        for directory in tqdm(  # проходимся по директориям
                data_path.glob('*'), desc='Preprocessing directories',
                total=len(list(data_path.glob('*'))), colour='green'):
            if not directory.is_dir():  # если это не директория, скипаем
                continue
            # создаем директорию для сохранения
            (save_dir := save_path / directory.name).mkdir(parents=True, exist_ok=True)
            for image in tqdm(  # берем только пнг-шки
                    directory.glob('*.png'), desc='Preprocessing images',
                    total=len(list(directory.glob('*.png'))), colour='blue'):
                posed: Results = self.yolo_pose.predict(  # применяем модель позы
                    image, classes=[0], verbose=False)[0]
                for i, pose_det in enumerate(posed):  # проходимся по найденным людям
                    preprocessed_frame = self.segment_crop_frame(pose_det)  # предобрабатываем изображение
                    if preprocessed_frame is None:
                        continue
                    cv2.imwrite(  # сохраняем его
                        (save_dir / (image.stem + f'_{i}' + image.suffix)).as_posix(), preprocessed_frame)


if __name__ == '__main__':
    p = Preprocessor()
    p.preprocess_data((Path.cwd().parents[0] / 'resources' / 'data' / 'raw'))
