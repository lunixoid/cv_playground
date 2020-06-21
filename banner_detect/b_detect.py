__author__ = 'romankalashnikov'
# -*- coding: UTF-8 -*-

import cv2
import sys
import os
import numpy as np
import uuid
import shutil
import argparse

version = "1.0.1"
open

class Detector:
    """" Поиск и сохранение баннеров на изображениях в указанной папке """
    templates = ((240, 400),        #1280
                 (300, 300),        #1200
                 (300, 250),        #1100
                 (728, 90),         #1636
                 (160, 600),        #1560
                 (120, 600),        #1440
                 (468, 60),         #1056
                 (336, 280),        #1232
                 (250, 250),        #1000
                 (200, 200),        #800
                 (200, 300))        #1000
    use_pattern = True     # Сохранять изображение как есть или использовать паттерн
    border_size = 2        # Толщина новой рамки
    eps_pix = 7            # Допуск разброса рамки в пикселях
    def __init__(self):
        # Создаем искомые шаблоны нужных форматов
        blank_image = np.zeros((400, 240, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_400_240 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((300, 300, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_300_300 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((250, 300, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_250_300 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((90, 728, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_90_728 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((600, 160, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_600_160 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((600, 120, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_600_120 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((60, 468, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_60_468 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((280, 336, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_280_336 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((250, 250, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_250_250 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((200, 200, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_200_200 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

        blank_image = np.zeros((300, 200, 3), np.uint8)
        constant = cv2.copyMakeBorder(blank_image, 2, 2, 2, 2, cv2.BORDER_CONSTANT, value=WHITE)
        gray_img = cv2.cvtColor(constant, cv2.COLOR_BGR2GRAY)
        cnts, h = cv2.findContours(gray_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_KCOS)
        self.templ_300_200 = sorted(cnts, key=cv2.contourArea, reverse=True)[1]

    def size_to_template(self, w = int, h = int):
        """
        Приводим размер к стандартному виду
        :param w: Ширина
        :param h: Высота
        :return: Ширина и Высота из эталонного набора
        """
        for temp in self.templates:
            if (temp[0]-self.eps_pix < w < temp[0]+self.eps_pix) and (temp[1]-self.eps_pix < h < temp[1]+self.eps_pix):
                return temp[0], temp[1]

    def save_banner(self, path=str, img=None, cnt=None):
        """
        Сохраняет в файл с рандомным название желаемую область изображения
        :param path: Путь до папки, куда необходимо сохранить
        :param img: исходное изображение
        :param cnt: контур в формате openCV
        :return:
        """
        x, y, width, height = cv2.boundingRect(cnt)
        # x += 1
        # y += 1
        if self.use_pattern:
            w, h = self.size_to_template(width, height)
        else:
            w, h = width, height
        roi = img[y:y+h, x:x+w]
        crop = roi[self.border_size:h-self.border_size,
                   self.border_size:w-self.border_size]

        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        Z = gray_roi.reshape((-1, 3))
        # convert to np.float32
        Z = np.float32(Z)
        # define criteria, number of clusters(K) and apply kmeans()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 1
        ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        val = center[0]
        gray_scale = val[0]+val[1]+val[2]
        templ_gray = 128+128+128
        if gray_scale > templ_gray:
            val = (0, 0, 0)
        else:
            val = (255, 255, 255)

        constant = cv2.copyMakeBorder(crop, 1, 1, 1, 1, cv2.BORDER_REPLICATE)
        constant = cv2.copyMakeBorder(constant, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=val)
        # cv2.imshow("i", constant)
        # cv2.waitKey()
        cv2.imwrite(path + str(uuid.uuid4()) + ".png", constant)

    def is_size_in_range(self, w=int, h=int, wD=int, hD=int):
        """
        Проверяет возможность вхождения фигуры в эталон
        :param w: Исходная ширина
        :param h: Исходная высота
        :param wD: Эталонная ширина
        :param hD: Эталонная высота
        :return: True/False
        """
        if (wD-self.eps_pix < w < wD+self.eps_pix) and (hD-self.eps_pix < h < hD+self.eps_pix):
            return True
        else:
            return False

    def work_in_dir(self, dir=str):
        """
        Обрабатывает все доступные файлы в указанной папке (пропускает скрытые файлы)
        TODO: проверять является ли файл изображением
        :param dir: путь до папки с изображениями
        :return:
        """
        # Проверяем доступность папки
        if not os.path.exists(dir):
                print 'ERROR directory or file does not exist:' + dir
        else:
            print 'In directory %s:' % dir
            # Удаляем временную папку (если есть) и создаем заного
            if os.path.exists(dir + '/tmp'):
                shutil.rmtree(dir + '/tmp')
            if os.path.exists(dir + '/unresolved'):
                shutil.rmtree(dir + '/unresolved')
            os.mkdir(dir + '/unresolved')
            os.mkdir(dir + '/tmp')

            e1 = cv2.getTickCount()     # Последнее значение времени (для подсчета времени работы)
            for image in os.listdir(dir):
                # Проверяем является ли объект папкой или скрытым файлом (linux/osx)
                if not os.path.isfile(dir + "/" + image) or image[0] == ".":
                    continue
                # Читаем изображение
                img = cv2.imread(param+'/'+image)
                # img = cv2.bilateralFilter(img, 10, 10, 10)    # Сглаживание
                # Переводим в оттенки серого
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                # Применяем порог серого, тем самым уменьшаем вариации цветов
                ret, thresh = cv2.threshold(gray_img, 70, 140, cv2.THRESH_TRUNC)    # work
                # cv2.imshow("Show image", thresh)
                # cv2.waitKey()
                # Производим поиск сторон объектов методом Canny с апертурой и минимальной детализацией
                # (внтурни он сам очищает от шумов методом Sobel и сглаживает)
                edges = cv2.Canny(thresh, 15, 200, 1)   # work
                # edges = cv2.Canny(edges, 1, 50, 1)
                # cv2.imshow("Show image", edges)
                # cv2.waitKey()
                # Ищем контура в получившемся изображении по алгоритму RETR_TREE со сглаживанием CHAIN_APPROX_TC89_L1
                contours, h = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)    # work
                # contours, h = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)    # work

                matchM = 1000   # Минимальное значение совпадения (чем меньше, тем лучше)
                match0 = 1000   # Текущее значение совпадения
                prevM = 1000    # Индкес предыдущего, проверяемого, контура
                seqM = 0        # Индекс контура с наилучшим совпадением
                cnt = 0         # Счетчик итераций

                # Обходим все контура
                for contour in contours:
                    # Считаем периметр контура (контур считается разомкнутым)
                    perimeter = np.fabs(cv2.arcLength(contour, False))
                    # area = np.fabs(cv2.contourArea(contour))  # Площадь контура
                    # Получаем количество углов в контуре с упрощением формы на 20%
                    approx = cv2.approxPolyDP(contour, perimeter*0.2, False)
                    # Получаем геометрические размеры контура
                    x, y, width, height = cv2.boundingRect(contour)
                    if len(approx) < 1 or len(approx) > 5:  # Защита от явно неправильных контуров
                        continue
                    # Проверяем подходит ли контур по шаблону
                    # если подходит, то сравниваем с соответствущим эталоном self.templ_width_height
                    if self.is_size_in_range(width, height, 240, 400):
                        match0 = cv2.matchShapes(contour, self.templ_400_240, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 300, 300):
                        match0 = cv2.matchShapes(contour, self.templ_300_300, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 300, 250):
                        match0 = cv2.matchShapes(contour, self.templ_250_300, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 728, 90):
                        match0 = cv2.matchShapes(contour, self.templ_90_728, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 160, 600):
                        match0 = cv2.matchShapes(contour, self.templ_600_160, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 120, 600):
                        match0 = cv2.matchShapes(contour, self.templ_600_120, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 468, 60):
                        match0 = cv2.matchShapes(contour, self.templ_60_468, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 336, 280):
                        match0 = cv2.matchShapes(contour, self.templ_280_336, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 250, 250):
                        match0 = cv2.matchShapes(contour, self.templ_250_250, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 200, 200):
                        match0 = cv2.matchShapes(contour, self.templ_200_200, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    elif self.is_size_in_range(width, height, 200, 300):
                        match0 = cv2.matchShapes(contour, self.templ_300_200, cv2.cv.CV_CONTOURS_MATCH_I3, 0.0)
                    # Сверяем значение совпадения и если оно оптимальнее предыдущего, сохраняем
                    if match0 < matchM:
                        prevM = seqM
                        matchM = match0
                        seqM = cnt
                        # cv2.drawContours(img, [contour], -1, [0,255,0], 3)
                    cnt += 1
                # Если в изображении были контура и найдено хотя бы одно совпадение,
                # то сохраняем предполагаемый баннер в файл
                if len(contours) > 0 and match0 != 1000:
                    self.save_banner(dir + "/tmp/", img, contours[seqM])
                    # self.save_banner(dir + "/tmp/", img, contours[prevM])
                else:
                    shutil.copy(param+'/'+image, param+'/unresolved/'+image)
                # cv2.imshow("Show image", img)
                # cv2.waitKey()
            # Считаем время, за которое выполнена работа с папкой и выводим в консоль
            e2 = cv2.getTickCount() # Оконечное значение времени выполнения
            print "\nTime elapsed: %fs\n" % ((e2 - e1)/cv2.getTickFrequency())

def createParser():
    parser = argparse.ArgumentParser(
        prog='banner_detect',
        description="""Программа проходит все изображения в папке, вырезает баннеры
        и сохраняет их в подкаталог tmp
        """,
        epilog="""(c) Март 2015. Автор не несет ответственность за использование открытых библиотек""",
        add_help=False
    )
    parent_group = parser.add_argument_group(title='Параметры')
    parent_group.add_argument('--help', '-h', action='help', help='Справка')
    parent_group.add_argument('--version', '-v',
            action='version',
            help = 'Вывести номер версии',
            version='%(prog)s {}'.format (version))
    parent_group.add_argument('-p', help='Нужно ли использовать шаблонные размеры при сохранении',
                           default='True', type=bool, required=False, metavar='True/False')
    parent_group.add_argument('-b', help='Размер рамки (px)', default=2, type=int, required=False,
                           metavar='INT')
    parent_group.add_argument('-f', help='Перечень директорий', type=str, required=True,
                              nargs='+', metavar='DIR')
    parent_group.add_argument('-e', help='Максимальное отличие размеров от '
                                         'шаблона при обходе возможных баннеров (px)',
                              default=7, type=int, required=False, metavar='INT')
    return parser


if __name__ == "__main__":
    BLACK = [0, 0, 0]
    WHITE = [255, 255, 255]
    pars = createParser()
    namespace = pars.parse_args(sys.argv[1:])

    worker = Detector()
    worker.use_pattern = namespace.p
    worker.border_size = namespace.b
    worker.eps_pix = namespace.e

    for param in namespace.f:
        # try:
            worker.work_in_dir(param)
        # except: