# coding=utf-8
__author__ = 'romankalashnikov'
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import argparse
import sys

version = "1.0.1"

def get_colony_roi(img):
    col_sum = []
    row_sum = []
    colonies_x = []
    colonies_y = []
    colonies_broder = 1
    # Считаем сумму значений строк
    for elem in img:
        row_sum.append(elem.sum())
    # Считаем сумму значений столбцов
    t_edges = img.transpose()
    for elem in t_edges:
        col_sum.append(elem.sum())
    # Обходим значения получившихся гистрограмм, если это начало или конец области
    # то сохраняем координату
    for i in range(0, len(row_sum)-colonies_broder, 1):
        if row_sum[i] == 0 and row_sum[i+colonies_broder] > 0:
            colonies_y.append([i, 0])
        elif row_sum[i] > 0 and row_sum[i+colonies_broder] == 0:
            colonies_y.append([i, 1])
    for i in range(0, len(col_sum)-colonies_broder, 1):
        if col_sum[i] == 0 and col_sum[i+colonies_broder] > 0:
            colonies_x.append([i, 0])
        elif col_sum[i] > 0 and col_sum[i+colonies_broder] == 0:
            colonies_x.append([i, 1])
    dots = []
    # Создаем пары координат: [начало, конец] ненулевых областей
    for x in range(0, len(colonies_x), 2):
        for y in range(0, len(colonies_y), 2):
            dots.append([[colonies_x[x][0], colonies_y[y][0]], [colonies_x[x+1][0], colonies_y[y+1][0]]])
    colonies = []
    # Выделяем колонии
    for dot in dots:
            dot1 = (dot[0][0], dot[0][1])
            # dot2 = (dot[1][0], dot[1][1])
            roi_w = dot[1][0] - dot[0][0]
            roi_h = dot[1][1] - dot[0][1]
            colony_roi = img[dot1[1]:dot1[1]+roi_h, dot1[0]:dot1[0]+roi_w]
            if colony_roi.sum() != 0:
                colonies.append(colony_roi)
    return colonies, dots


def get_kmeans_threshold(gray_img, max_t=150, min_t=145):
    Z = gray_img
    # Конвертируем в np.float32
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS, 10, 1.0)
    K = 1
    # Считаем распространенность цветов
    ret, label, center = cv2.kmeans(Z, K, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # Берем среднее значение всех яркостей + 15
    gray_scale = center[0].sum()/len(center[0]) + 15
    # Сверяем с порогом
    if gray_scale > max_t:
        gray_scale = max_t
    elif min_t < min_t:
        gray_scale = min_t
    return gray_scale


def learn(dir=str):
    border = 3
    samples = np.empty((0, 100))
    responses = []
    for image_str in os.listdir(dir):
        print image_str
        if not os.path.isfile(dir + "/" + image_str) or image_str[0] == ".":
                    continue

        img = cv2.imread(dir+'/'+image_str)
        w_original, h_original = img.shape[:2]
        crop = img[border:w_original-border,
                   border:h_original-border]
        gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Первичное применение порога
        thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 209, -5)
        # Ручной подбор параметров
        cv2.imshow('tresh', thresh)
        key = cv2.waitKey()
        C = 209
        d = -5
        while key != 27:
            if key == 63234:
                C -= 2
            elif key == 63235:
                C += 2
            elif key == 63232:
                d += 5
            elif key == 63233:
                d -=5
            else:
                print key
            print "C=" + str(C) + " d=" + str(d)
            thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, C, d)
            cv2.imshow('tresh', thresh)
            key = cv2.waitKey(0)
        # Усредняем области данных
        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        # Находим стороны
        edges = cv2.Canny(thresh, 50, 100, 3)
        # Находим контура
        bacts, hier = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # cv2.drawContours(img_contours, contours, -1, [0, 255, 0], 1)    # DEBUG
        keys = [i for i in range(48, 58)]
        for bact in bacts:
            area = cv2.contourArea(bact)
            if area > 10:
                # Рисуем контур
                [x, y, w, h] = cv2.boundingRect(bact)
                roi_tmp = morph[y:y+h, x:x+w]
                roismall = cv2.resize(roi_tmp, (10, 10))
                cv2.drawContours(crop, [bact], -1, [0, 0, 255], 1)
                cv2.imshow('training', crop)
                key = cv2.waitKey(0)
                if key == 27:
                    break
                elif key == 13:
                    continue
                elif key in keys:
                    # Если пользователь опознал бактерию, сохраняем присовенный
                    # идентификатор и область контура
                    responses.append(int(chr(key)))
                    sample = roismall.reshape((1, 100))
                    samples = np.append(samples, sample, 0)
    # Сохраняем обучающие выборки в файл
    responses = np.array(responses, np.float32)
    responses = responses.reshape((responses.size, 1))
    np.savetxt('generalsamples.data', samples)
    np.savetxt('generalresponses.data', responses)


def find(dir=str):
    border = 3
    col_num = 0
    agalactiae = np.loadtxt('spectrums/agalactiae.data')
    aureus = np.loadtxt('spectrums/aureus.data')
    coli = np.loadtxt('spectrums/coli.data')
    epidermidis = np.loadtxt('spectrums/epidermidis.data')
    oxytoca = np.loadtxt('spectrums/oxytoca.data')
    saprophyticus = np.loadtxt('spectrums/saprophyticus.data')
    # Загружаем обучающие выборки
    samples = np.loadtxt('generalsamples.data', np.float32)
    responses = np.loadtxt('generalresponses.data', np.float32)
    responses = responses.reshape((responses.size, 1))
    # Обучаем модель Knn
    model = cv2.KNearest()
    model.train(samples, responses)
    for image_str in os.listdir(dir):
        print image_str
        if not os.path.isfile(dir + "/" + image_str) or image_str[0] == ".":
                    continue
        e1 = cv2.getTickCount()

        img = cv2.imread(dir+'/'+image_str)

        # img = cv2.bilateralFilter(img, -1, 30, 30)
        w_original, h_original = img.shape[:2]
        crop = img[border:w_original-border,
                   border:h_original-border]
        img_contours = np.zeros((w_original, h_original, 3), np.uint8)
        gray_img = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        gray_scale = get_kmeans_threshold(img, 255)
        print gray_scale
        # ret, thresh = cv2.threshold(gray_img, gray_scale, 255, cv2.THRESH_BINARY)

        thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 209, -5)
        # Ручной подбор параметров
        cv2.imshow('tresh', thresh)
        key = cv2.waitKey()
        C = 209
        d = -5
        while key != 27:
            if key == 63234:
                C -= 2
            elif key == 63235:
                C += 2
            elif key == 63232:
                d += 5
            elif key == 63233:
                d -=5
            else:
                print key
            print "C=" + str(C) + " d=" + str(d)
            thresh = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, C, d)
            cv2.imshow('tresh', thresh)
            key = cv2.waitKey(0)

        kernel = np.ones((2, 2), np.uint8)
        morph = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

        edges = cv2.Canny(morph, 50, 100, 3)
        bacts, hier = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
        cv2.drawContours(img_contours, bacts, -1, [0, 255, 0], 1)    # DEBUG
        # Выделяем колонии на изображении
        colonies, dots = get_colony_roi(edges)
        for dot in dots:
            dot1 = (dot[0][0], dot[0][1])
            dot2 = (dot[1][0], dot[1][1])
            roi_w = dot[1][0] - dot[0][0]
            roi_h = dot[1][1] - dot[0][1]
            colony_roi = edges[dot1[1]:dot1[1]+roi_h, dot1[0]:dot1[0]+roi_w]
            # Если колония не пуста
            if colony_roi.sum() > 1000:
                # Вырезаем области из серого изображения
                gray_roi = gray_img[dot1[1]:dot1[1]+roi_h, dot1[0]:dot1[0]+roi_w]
                # Считаем фурье
                dft = cv2.dft(np.float32(gray_roi), flags=cv2.DFT_COMPLEX_OUTPUT)
                dft_shift = np.fft.fftshift(dft)
                magnitude_spectrum = 20*np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))
                magnitude_spectrum.transpose()
                magn_simple = np.arange(len(magnitude_spectrum), dtype=np.float32)
                # Переводим спектр в 1d
                for i in range(0, len(magnitude_spectrum), 1):
                    magn_simple[i] = (magnitude_spectrum[i].sum())/float(len(magnitude_spectrum))
                # Если он пуст, то пропускаем область
                if magn_simple.sum() < 255:
                    continue
                # Нормализуем значения спектра к 255
                magn_simple = magn_simple/float(np.max(magn_simple)) * 255
                # Препроцессинг области колонии
                thresh_roi = cv2.adaptiveThreshold(gray_roi, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, C, d)
                #kernel = np.ones((2, 2), np.uint8)
                #morph_roi = cv2.morphologyEx(thresh_roi, cv2.MORPH_OPEN, kernel)
                edges_roi = cv2.Canny(thresh_roi, 50, 100, 3)
                # Находим бактерий
                bacts, hier = cv2.findContours(edges_roi, cv2.RETR_TREE, cv2.CHAIN_APPROX_TC89_L1)
                # Если это малочисленная колония, то пропускаем
                if len(bacts) < 6:
                    continue
                #cv2.imwrite(str(col_num) + image_str, magnitude_spectrum)
                """plt.subplot(121), plt.plot(magnitude_spectrum),
                plt.title('Magnitude Spectrum')
                plt.subplot(122), plt.imshow(gray_roi, cmap='gray')
                plt.title('Gray roi'), plt.xticks([]), plt.yticks([])
                plt.savefig(str(col_num) + "spectrum_" + image_str)
                plt.close()
                plt.close()"""
                col_num+=1
                bact_list = np.zeros(5, dtype=np.float32)
                # Проверяем каждую бактерию
                for bact in bacts:
                    area = cv2.contourArea(bact)
                    if area > 10:
                        [x, y, w, h] = cv2.boundingRect(bact)
                        tmp_roi = edges_roi[y:y+h, x:x+w]
                        bact_roi = cv2.resize(tmp_roi, (10, 10))
                        bact_roi = bact_roi.reshape((1, 100))
                        bact_roi = np.float32(bact_roi)
                        # Сверяем с эталоном по методу Knn
                        retval, results, neigh_resp, dists = model.find_nearest(bact_roi, k=1)
                        # Заполняем счетчик
                        bact_list[results[0][0]-1] += 1
                # print bact_list
                bact_num_sum = bact_list.sum()
                if bact_num_sum == 0:
                    continue
                # Выводим результаты
                print 'Sticks -> {1}%'.format(0, bact_list[0]/bact_num_sum*100)
                print 'Cocci  -> {1}%'.format(0, bact_list[1]/bact_num_sum*100)
                if bact_list[0] > bact_list[1]:
                    cv2.rectangle(img_contours, dot1, dot2, [0,0,255])
                else:
                    cv2.rectangle(img_contours, dot1, dot2, [0,255,255])
                if len(magn_simple) > 105:
                    variants = []
                    bord = 50
                    half = len(magn_simple)/2
                    magn_roi = magn_simple[half-bord:half+bord]
                    half = len(agalactiae)/2
                    agalactiae_roi = agalactiae[half-bord:half+bord]
                    compare_res = np.intersect1d(magn_roi, agalactiae_roi)
                    output = np.correlate(magn_roi, agalactiae_roi, 'full')
                    variants.append(np.max(output))
                    print 'agalactiae compare: ' + str(len(compare_res))

                    half = len(aureus)/2
                    aureus_roi = aureus[half-bord:half+bord]
                    compare_res = np.intersect1d(magn_roi, aureus_roi)
                    output = np.correlate(magn_roi, aureus_roi, 'full')
                    variants.append(np.max(output))
                    print 'aureus compare: ' + str(len(compare_res))

                    half = len(coli)/2
                    coli_roi = coli[half-bord:half+bord]
                    compare_res = np.intersect1d(magn_roi, coli_roi)
                    output = np.correlate(magn_roi, coli_roi, 'full')
                    variants.append(np.max(output))
                    print 'coli compare: ' + str(len(compare_res))

                    half = len(epidermidis)/2
                    epidermidis_roi = epidermidis[half-bord:half+bord]
                    compare_res = np.intersect1d(magn_roi, epidermidis_roi)
                    output = np.correlate(magn_roi, epidermidis_roi, 'full')
                    variants.append(np.max(output))
                    print 'epidermidis compare: ' + str(len(compare_res))

                    half = len(oxytoca)/2
                    oxytoca_roi = oxytoca[half-bord:half+bord]
                    compare_res = np.intersect1d(magn_roi, oxytoca_roi)
                    output = np.correlate(magn_roi, oxytoca_roi, 'full')
                    variants.append(np.max(output))
                    print 'oxytoca compare: ' + str(len(compare_res))

                    half = len(saprophyticus)/2
                    saprophyticus_roi = saprophyticus[half-bord:half+bord]
                    compare_res = np.intersect1d(magn_roi, saprophyticus_roi)
                    output = np.correlate(magn_roi, saprophyticus_roi, 'full')
                    variants.append(np.max(output))
                    print 'saprophyticus compare: ' + str(len(compare_res))
                    max = variants[0]
                    index = 0
                    for i in range(0, len(variants), 1):
                        if variants[i] > max:
                            max = variants[i]
                            index = i
                    # print index"""
                """plt.subplot(121), plt.plot(magnitude_spectrum),
                plt.title('Magnitude Spectrum')
                plt.subplot(122), plt.imshow(gray_roi, cmap='gray')
                plt.title('Gray roi'), plt.xticks([]), plt.yticks([])
                plt.show()"""

                """cv2.imshow('test', gray_roi)
                key = cv2.waitKey(0)
                if key == 27:
                    continue
                elif key == 13:
                    np.savetxt(image_str+'.data', magn_simple)"""

        e2 = cv2.getTickCount()     # Оконечное значение времени выполнения
        print "\nTime elapsed: %fs\n" % ((e2 - e1)/cv2.getTickFrequency())
        # cv2.drawContours(img_contours, contours, -1, [0, 255, 0], 1)

        #cv2.imshow("img", img)
        # cv2.imshow('gray', gray_img)
        #cv2.imshow('tresh', thresh)
        # cv2.imshow('morph', morph)
        # cv2.imshow('edges', edges)
        #cv2.imshow('contours', img_contours)
        # cv2.imshow('hist', histbase)
        # cv2.imwrite(dir+'/'+'tresh'+image_str, thresh)
        # cv2.imwrite(image_str, img_contours)
        #cv2.waitKey()


def createParser():
    parser = argparse.ArgumentParser(
        prog='baсteria_finder',
        description="""Программа обходит папку и производит поиск колоний бактерий на каждом изображении, выделяет их и
        идентифицирует:
        красная рамка - палочки
        желтая рамка - кокки
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
    parent_group.add_argument('-l', help='Запуск обучающего алгоритма', default=0, type=int, required=False,
                       metavar='INT')
    parent_group.add_argument('-f', help='Перечень директорий через пробел', type=str, required=True,
                              nargs='+', metavar='DIR')
    return parser


if __name__ == "__main__":
    pars = createParser()
    namespace = pars.parse_args(sys.argv[1:])
    if namespace.l > 0:
        learn('learn_dir')
    for param in namespace.f:
        find(param)