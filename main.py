from ultralytics import YOLOv10
import cv2 as cv
import numpy as np
import easyocr


class ML:

    translation = {'-': 'text', 'AUX': "Вспомогательный", 'AVR': "АВР", 'FR': "ФР",
                   'FU': "Предохранитель плавкий/Быстродействующий", 'FV': "Разрядник вентильный/Трубчатый",
                   'HL': "Лампа накаливания сигнальная", 'ITU': "Расцепитель независимый",
                   'K': "Катушка электромеханического устройства", 'KM': "Контакт контактора", 'M': "Электродвигатель",
                   'MX': "MX", 'OPS': "ОПС", 'PA': "Амперметр", 'PM': "Мультиметр", 'PV': "Вольтметр",
                   'Q': "Выключатель высоковольтный", 'QD': "Выключатель дифференциальный",
                   'QF': "Выключатель-предохранитель", 'QFD': "Выключатель автоматический дифференциальный",
                   'QFU': "Предохранитель-разъединитель", 'QS': "Разъединитель",
                   'QW': "Выключатель нагрузки", 'R': "Реле", 'S': "Рукоятка", 'TT': "Трансформатор", 'Timer': "Таймер",
                   'U-': "Минимальное напряжение", 'WH': "Счетчик активной энергии", 'XS': "Гнездо",
                   'YZIP': "УЗИП", 'switch': "Переключатель"}
    text_bool = {'Вспомогательный': 0,
                 'АВР': 0, 'ФР': 0, 'Предохранитель плавкий/Быстродействующий': 1, 'Разрядник вентильный/Трубчатый': 1,
                 'Лампа накаливания сигнальная': 1, 'Расцепитель независимый': 1,
                 'Катушка электромеханического устройства': 1,
                 'Контакт контактора': 1, 'Электродвигатель': 1, 'MX': 1, 'ОПС': 1, 'Амперметр': 0, 'Мультиметр': 1,
                 'Вольтметр': 0,
                 'Выключатель высоковольтный': 1, 'Выключатель дифференциальный': 1, 'Выключатель-предохранитель': 1,
                 'Выключатель автоматический дифференциальный': 1, 'Предохранитель-разъединитель': 1,
                 'Разъединитель': 1,
                 'Выключатель нагрузки': 1, 'Реле': 1, 'Рукоятка': 0, 'Трансформатор': 1, 'Таймер': 1,
                 'Минимальное напряжение': 0,
                 'Счетчик активной энергии': 1, 'Гнездо': 0, 'УЗИП': 1, 'Переключатель': 1}
    def __init__(self, weights="./bestTimur.pt"):
        self.picture_path = None
        self.grouped_objects = {}
        self.model = YOLOv10(weights)
        self.reader = easyocr.Reader(['en', 'ru'])
        self.result = []

    def detect(self, picture_path):
        result = self.model.predict(source=picture_path, conf=0.75)
        self.picture_path = picture_path
        return self.result_structured(result[0])

    def detect_show(self, picture_path):
        result = self.model.predict(source=picture_path, conf=0.75, show=True, show_labels=True, show_boxes=True,
                                    save=True, iou=1.0)
        self.picture_path = picture_path
        return self.result_structured(result[0])

    def validation(self):
        self.model.val(data="./EKF.v1i.yolov9/data.yaml")

    def result_structured(self, res):
        classes_names = res.names
        classes = res.boxes.cls.cpu().numpy()
        boxes = res.boxes.xyxy.cpu().numpy().astype(np.int32)

        for class_id, box in zip(classes, boxes):
            class_name = self.translation[classes_names[int(class_id)]]
            if class_name not in self.grouped_objects:
                self.grouped_objects[class_name] = []
            self.grouped_objects[class_name].append(box)
        print(self.grouped_objects)
        self.find_nearest()
        self.use_ocr()
        return self.grouped_objects

    def find_nearest(self):
        copy_coordinates = self.grouped_objects.copy()
        text_coordinates = copy_coordinates.pop("text")
        self.grouped_objects = []
        for k, v in copy_coordinates.items():
            if self.text_bool[k]:
                for item in v:
                    min = -1
                    min_text_coord = None
                    for i in text_coordinates:
                        sum = abs(item[0] - i[0] + item[1] - i[1] + item[2] - i[2] + item[3] - i[3])
                        if sum < min or min == -1:
                            min = sum
                            min_text_coord = i
                    self.grouped_objects.append([k, min_text_coord])
            else:
                self.result.append(k)
                continue

    def use_ocr(self):
        img = cv.imread(self.picture_path)
        for i in self.grouped_objects:
            print(self.reader.readtext(img[i[1][1]:i[1][3], i[1][0]:i[1][2]], detail=0))
            self.result.append(i[0])
        print(self.result)


if __name__ == '__main__':

    ml = ML()
    print({v: k for k, v in ML.translation.items()})
    ml.detect_show('DATAstart/images/205-page-00001.jpg')
    cv.waitKey(0)




