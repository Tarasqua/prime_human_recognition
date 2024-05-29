import os
from pathlib import Path
from time import perf_counter

from ultralytics import YOLO
from loguru import logger


def log_trace(func):
    """
    Декоратор, логирующий вызовы функции: вызов, время выполнения и ошибки.
    :param func: Функция, которая будет задекорирована.
    :return: Задекорированая функция.
    """
    def wrapper(*args, **kwargs):
        logger.info(f'Calling {func.__name__} function '
                    f'with args: {args}, kwargs: {kwargs}')
        start_time: float = perf_counter()
        try:
            original_result = func(*args, **kwargs)
        except Exception as e:
            logger.exception(f'Function {func.__name__} raised {e}')
            raise
        else:
            logger.success(f'Function {func.__name__} successfully finished '
                           f'in {round(perf_counter() - start_time, 2)} seconds')
        return original_result
    return wrapper


@log_trace
def set_yolo_model(yolo_model: str, yolo_class: str, task: str = 'detect') -> YOLO:
    """
    Выполняет проверку путей и наличие модели:
    Если директория отсутствует, создает ее, а также скачивает в нее необходимую модель.
    :param yolo_model: n (nano), m (medium), etc.
    :param yolo_class: seg, pose, boxes.
    :param task: detect, segment, classify, pose.
    :return: Объект YOLO-pose.
    """
    yolo_class = f'-{yolo_class}' if yolo_class != 'boxes' else ''
    yolo_models_path = Path.cwd().parents[0] / 'resources' / 'models' / 'yolo_models'
    if not os.path.exists(yolo_models_path):
        Path(yolo_models_path).mkdir(parents=True, exist_ok=True)
    model_path = os.path.join(yolo_models_path, f'yolov8{yolo_model}{yolo_class}')
    if not os.path.exists(f'{model_path}.onnx'):
        YOLO(model_path).export(format='onnx')
    return YOLO(f'{model_path}.onnx', task=task, verbose=False)

