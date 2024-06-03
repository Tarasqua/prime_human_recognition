from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results
from tqdm import tqdm

from utils.util import set_yolo_model
from ultralytics.utils.ops import scale_image
from utils.util import log_trace


class Preprocessor:

    def __init__(self, yolo_seg_model: str = 's', yolo_pose_model: str = 's'):
        """
        Предобработка изображений для обучения.
        :param yolo_seg_model: Версия модели для сегментации.
        :param yolo_pose_model: Версия модели для позы.
        """
        self.yolo_seg: YOLO = set_yolo_model(yolo_model=yolo_seg_model, yolo_class='seg', task='segment')
        self.yolo_pose: YOLO = set_yolo_model(yolo_model=yolo_pose_model, yolo_class='pose', task='pose')
        self.yolo_seg.predict(np.random.random((100, 100, 3)), verbose=False)  # тестово запускаем
        self.yolo_pose.predict(np.random.random((100, 100, 3)), verbose=False)

    @staticmethod
    def align_frame(image: np.array, alignment_points: np.array, clockwise: bool = True) -> Tuple[np.array, np.array]:
        """
        Поворачивает изображение на заданный угол, не обрезая края изображения.
        :param image: Исходное изображение для поворота.
        :param alignment_points: Опорные точки, по которым будет произведен поворот, вида [[x1, y1], [x2, y2]].
        :param clockwise: Осуществлять поворот по часовой стрелке или нет.
        :return:
        """

        def rotate_image(angle) -> Tuple[np.array, np.array]:
            """
            Поворачивает изображение на заданный угол, не обрезая края изображения.
            :param angle: Угол поворота в радианах.
            :return: Изображение и матрица поворота.
            """
            (h, w) = image.shape[:2]
            center = (w / 2, h / 2)
            # новые размеры изображения
            new_w = int(h * (sin_a := abs(np.sin(angle))) + w * (cos_a := abs(np.cos(angle))))
            new_h = int(h * cos_a + w * sin_a)
            rot_mat = cv2.getRotationMatrix2D(center, np.degrees(angle), 1.0)  # матрица поворота
            # корректируем матрицу поворота с учетом смещения
            rot_mat[0, 2] += (new_w / 2) - center[0]
            rot_mat[1, 2] += (new_h / 2) - center[1]
            # применяем матрицу поворота с новыми размерами изображения
            rotated = cv2.warpAffine(image, rot_mat, (new_w, new_h))
            return rotated, rot_mat

        direction_vector = alignment_points[1] - alignment_points[0]  # считаем направляющую
        # находим угол поворота в радианах
        rotate_angle = (np.arctan2(direction_vector[1], direction_vector[0]) - np.pi) * (-1 if not clockwise else 1)
        return rotate_image(rotate_angle)

    def get_aligned_human(self, detection: Results, kpts_conf: float = 0.5
                          ) -> Tuple[Tuple[int, int, int, int], np.array, np.array]:
        """
        Поворачивает изображение и вычленяет человека из кадра.
        :param detection: Результат детекции позы человека.
        :param kpts_conf: Conf для ключевых точек.
        :return: Повернутые координаты ббокса и изображение человека.
        """

        def rotate_bbox(bbox_points: np.array, rotation_matrix: np.array) -> Tuple[int, int, int, int]:
            """
            Поворот координат ключевых точек.
            :param bbox_points: Координаты ббокса.
            :param rotation_matrix: Матрица поворота.
            :return: Повернутые координаты ключевых точек.
            """
            # преобразуем точки в однородные координаты, так как rot_mat имеем размерность (2, 3)
            points_homogeneous = np.hstack([bbox_points, np.ones(shape=(len(bbox_points), 1))])
            rotated_points = rotation_matrix.dot(points_homogeneous.T).T  # применяем матрицу поворота
            x1_new, y1_new, x2_new, y2_new = (  # получаем новые координаты
                int(rotated_points[:, 0].min()), int(rotated_points[:, 1].min()),
                int(rotated_points[:, 0].max()), int(rotated_points[:, 1].max()))
            # убеждаемся, что новые координаты находятся в пределах границ изображения
            h, w = rotated_frame.shape[:2]
            x1_new, y1_new = max(0, x1_new), max(0, y1_new)
            x2_new, y2_new = min(w, x2_new), min(h, y2_new)
            return x1_new, y1_new, x2_new, y2_new

        x1, y1, x2, y2 = detection.boxes.numpy().xyxy[0].astype(int)  # берем координаты ббокса
        if (kpts := detection.keypoints) is not None:  # если ключевые точки есть
            align_points = kpts.data.numpy()[0][1:3]  # точки, соответствующие глазам человека
            align_filtered = align_points[  # фильтруем, чтобы были не нулевые + чтобы conf был больше порогового
                                 (align_points[:, 0] > 0) | (align_points[:, 1] > 0) | (align_points[:, 2] > kpts_conf)
                                 ][:, :-1].astype(int)
            if align_filtered.shape[0] == 2:  # если с точками все ок
                # поворачиваем изображение и берем матрицу поворота
                rotated_frame, rot_mat = self.align_frame(detection.orig_img, align_points)
                # поворачиваем также ключевые точки
                x1, y1, x2, y2 = rotate_bbox(np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]]), rot_mat)
                # кропаем повернутое изображение по новым координатам, чтобы взять только человека
                return (x1, y1, x2, y2), rotated_frame[y1:y2, x1:x2], rot_mat
        return (x1, y1, x2, y2), detection.orig_img[y1:y2, x1:x2, :], np.array([])  # иначе же просто выдаем человека

    @staticmethod
    def get_segmented(segmentation: Results, frame: np.array) -> np.array:
        """
        Применение сегментационной маски к изображению с учетом ненулевых пикселей
        и раздувания выходного изображения.
        :param segmentation: Результат сегментации человека.
        :param frame: Изображение с человеком.
        :return: Сегментированное изображение.
        """
        if (seg_data := segmentation.masks) is None:  # убеждаемся, что сетка нашли что-либо
            return None
        masks = np.moveaxis(seg_data.data.numpy(), 0, -1)  # приводим к виду (H, W, N)
        masks = scale_image(masks, segmentation.masks.orig_shape)  # приводим маску к изначальному размеру кадра
        masks = np.moveaxis(masks, -1, 0)  # обратно к виду (N, H, W)
        return cv2.bitwise_and(  # применяем маску к исходному изображению
            frame, frame, mask=cv2.resize(masks[0], frame.shape[:-1][::-1]).astype('uint8'))

    @staticmethod
    def crop_segmented(segmented_frame: np.array, keypoints: np.array,
                       tl_coords: Tuple[int, int], rot_mat: np.array,
                       margin: bool = True, kpts_conf: float = 0.5) -> np.array:
        """
        Обрезание сегментированного изображения для вычленения верхней части тела человека.
        :param segmented_frame: Сегментированное изображение.
        :param keypoints: Ключевые точки тела человека.
        :param tl_coords: Верхние левые координаты ббокса человека.
        :param rot_mat: Матрица поворота.
        :param margin: Добавлять ли отступы слева и справа от ббокса (5% от ширины ббокса).
        :param kpts_conf: Conf для ключевых точек.
        :return: Кропнутое изображение.
        """

        def rotate_keypoints(points: np.array) -> np.array:
            """
            Поворот координат ключевых точек.
            :param points: Координаты ключевых точек
            :return: Повернутые координаты точек.
            """
            keypoints_homogeneous = np.hstack([points, np.ones(shape=(len(points), 1))])
            rotated_keypoints_homogeneous = rot_mat.dot(keypoints_homogeneous.T).T
            return rotated_keypoints_homogeneous[:, :2].astype(int)

        def get_borders(kpts: np.array) -> Tuple[int, int, int]:
            """
            Вычисление граничных точек верхней части тела.
            :param kpts: Координаты ключевых точек верхней части тела.
            :return: Граничные точки: левая, правая, нижняя.
            """
            leftmost, rightmost = np.array(  # берем крайние левую и правую точки (как лицевые точки, так и плечи)
                [min(kpts[:, 0]), max(kpts[:, 0])]) - tl_coords[0]  # + в относительные
            lowest = max(rotated_keypoints[:, 1]) - tl_coords[1]  # нижняя точка (также не обязательно плечи)
            # также проверяем, чтобы точки лежали внутри изображения
            h, w = segmented_frame.shape[:2]
            if margin:
                leftmost, rightmost = leftmost - (m := int(w * 0.05)), rightmost + m
            return (leftmost if leftmost >= 0 else 0,
                    rightmost if rightmost <= w else w,
                    lowest if lowest <= h else h)

        # берем верхние ключевые точки и фильтруем их, чтобы они были не нулевыми + чтобы conf был больше порогового
        filtered_keypoints = (
            (upper_keypoints := keypoints[:7])[
                (upper_keypoints[:, 0] > 0) | (upper_keypoints[:, 1] > 0) | (upper_keypoints[:, 2] > kpts_conf)
                ][:, :-1].astype(int))
        # если точек хватает + имел место поворот + все точки повернулись
        if (filtered_keypoints.shape[0] >= 2 and rot_mat.shape[0] != 0 and
                (rotated_keypoints := rotate_keypoints(filtered_keypoints)).shape[0] != 0):
            x1, x2, y2 = get_borders(rotated_keypoints)  # берем границы
            return segmented_frame[:y2, x1:x2]  # обрезаем
        # иначе же просто берем верхнюю треть человека
        return segmented_frame[:int(tl_coords[1] / 3), :, :]

    def segment_crop_frame(self, pose_detection: Results, margin: bool = True, kpts_conf: float = 0.5) -> np.array:
        """
        Выравнивание по глазам, сегментация, обрезка изображения для вычленения верхней части тела человека,
        а также перевод в грейскейл.
        :param pose_detection: Результат детекции позы человека.
        :param margin: Добавлять ли отступы слева и справа от ббокса (5% от ширины ббокса).
        :param kpts_conf: Conf для ключевых точек.
        :return: Кропнутое и загрейскейленное изображение верхней части тела человека.
        """
        # выравниваем изображение по глазам человека
        (x1, y1, x2, y2), human_frame, rot_mat = self.get_aligned_human(pose_detection, kpts_conf=kpts_conf)
        seg_result: Results = self.yolo_seg.predict(  # применяем модель для сегментации
            human_frame, classes=[0], verbose=False)[0]
        if (segmented := self.get_segmented(seg_result, human_frame)) is not None:  # сегментируем кадр
            cropped = self.crop_segmented(  # обрезаем сегментированный кадр по верхней части тела человека
                segmented, pose_detection.keypoints.data.numpy()[0], (x1, y1), rot_mat,
                margin=margin, kpts_conf=kpts_conf)
            # дополнительно кропаем черную маску вокруг, если она есть
            y_nz, x_nz = np.nonzero(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY))  # координаты ненулевых пикселей
            if any([y_nz.shape[0], x_nz.shape[0]]):
                cropped = cropped[np.min(y_nz):np.max(y_nz), np.min(x_nz):np.max(x_nz)]  # применяем к кропнутому кадру
            return cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        return None

    @log_trace
    def preprocess_data(self, data_path: Path, margin: bool = True, kpts_conf: float = 0.5) -> None:
        """
        Предобработка данных для обучения:
        - находим людей в кадре;
        - кропаем людей, вычленяя верхнюю часть тела;
        - переводим в грейскейлевый формат;
        - сохраняем обработанные изображения.
        :param data_path: Путь до директории с данными в формате '../data/1/*.png', '../data/2/*.png', ...
        :param margin: Добавлять ли отступы слева и справа от ббокса (5% от ширины ббокса).
        :param kpts_conf: Conf для ключевых точек.
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
                    if (preprocessed_frame := self.segment_crop_frame(  # предобрабатываем изображение
                            pose_det, margin=margin, kpts_conf=kpts_conf)) is None:
                        continue
                    cv2.imwrite(  # сохраняем его
                        (save_dir / (image.stem + f'_{i}' + image.suffix)).as_posix(), preprocessed_frame)


if __name__ == '__main__':
    p = Preprocessor('m', 'm')
    p.preprocess_data((Path.cwd().parents[0] / 'resources' / 'data' / 'raw_kitchen'))
