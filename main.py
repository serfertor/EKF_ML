from ultralytics import YOLOv10
import cv2 as cv
import numpy as np


class ML:
    translation = {'-': 'text', 'AUX': "Вспомогательный", 'AVR': "АВР", 'FR': "ФР",
                   'FU': "Предохранитель плавкий/Быстродействующий", 'FV': "Разрядник вентильный/Трубчатый",
                   'HL': "Лампа накаливания сигнальная", 'ITU': "Расцепитель независимый",
                   'K': "Катушка электромеханического устройства", 'KM': "Контакт контактора", 'M': "Электродвигатель",
                   'MX': "???MX", 'OPS': "ОПС", 'PA': "Амперметр", 'PM': "Мультиметр", 'PV': "Вольтметр",
                   'Q': "Выключатель",'QD': "Выключатель дифференциальный",
                   'QF': "Выключатель-предохранитель", 'QFD': "Выключатель автоматический дифференциальный",
                   'QFU': "Предохранитель-разъединитель", 'QS': "Выключатель низковольтный однополюсный",
                   'QW': "Выключатель нагрузки", 'R': "Реле", 'S': "Рукоятка", 'TT': "Трансформатор", 'Timer': "Таймер",
                   'U-': "Минимальное напряжение", 'WH': "Счетчик активной энергии", 'XS': "Гнездо",
                   'YZIP': "УЗИП", 'switch': "Переключатель"}

    def __init__(self, weights="./bestTimur.pt"):
        self.model = YOLOv10(weights)

    def detect(self, picture_path):
        result = self.model.predict(source=picture_path, conf=0.75)
        return self.result_structured(result[0])

    def detect_show(self, picture_path):
        result = self.model.predict(source=picture_path, conf=0.75, show=True, show_labels=True, show_boxes=True,
                                    save=True, iou=1.0)
        return self.result_structured(result[0])

    def validation(self):
        self.model.val(data="./EKF.v1i.yolov9/data.yaml")

    def result_structured(self, res):
        print(res.names)
        classes_names = res.names
        classes = res.boxes.cls.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.int32)

        grouped_objects = {}

        for class_id, box in zip(classes, boxes):
            class_name = self.translation[classes_names[int(class_id)]]
            if class_name not in grouped_objects:
                grouped_objects[class_name] = []
            grouped_objects[class_name].append(box)
        print(grouped_objects)
        return


if __name__ == '__main__':
    ml = ML()
    ml.detect_show('DATAstart/images/205-page-00001.jpg')
    cv.waitKey(0)
