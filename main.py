import os

import cv2 as cv
import matplotlib.pyplot as plt


def execute_matcher_orb(show_img):
    orb = cv.ORB.create()
    for directory in ['50km', 'lombada', 'pare']:
        input_img = cv.cvtColor(cv.imread('input/' + directory + '.jpg'), cv.COLOR_BGR2RGB)
        input_img_gray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
        kp1, des1 = orb.detectAndCompute(input_img_gray, None)

        for img_name in os.listdir('dataset/' + directory):
            data_img = cv.cvtColor(cv.imread('dataset/' + directory + '/' + img_name, cv.IMREAD_COLOR),
                                   cv.COLOR_BGR2RGB)
            data_img_gray = cv.cvtColor(data_img, cv.COLOR_BGR2GRAY)
            kp2, des2 = orb.detectAndCompute(data_img_gray, None)

            bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

            matches = bf.match(des1, des2)
            # matches = sorted(matches, key=lambda x: x.distance)

            result = cv.drawMatches(input_img, kp1, data_img, kp2, matches, None)

            plt.title('ORB')
            plt.suptitle(img_name)
            plt.imshow(result)
            if show_img:
                plt.show()
            save_img('orb', directory, img_name)


def execute_matcher_sift(show_img):
    sift = cv.SIFT.create()
    for directory in ['50km', 'lombada', 'pare']:
        input_img = cv.cvtColor(cv.imread('input/' + directory + '.jpg'), cv.COLOR_BGR2RGB)
        input_img_gray = cv.cvtColor(input_img, cv.COLOR_BGR2GRAY)
        kp1, des1 = sift.detectAndCompute(input_img_gray, None)

        for img_name in os.listdir('dataset/' + directory):
            data_img = cv.cvtColor(cv.imread('dataset/' + directory + '/' + img_name, cv.IMREAD_COLOR),
                                   cv.COLOR_BGR2RGB)
            data_img_gray = cv.cvtColor(data_img, cv.COLOR_BGR2GRAY)
            kp2, des2 = sift.detectAndCompute(data_img_gray, None)

            bf = cv.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Seleciona os pontos que possuem uma distancia razoavel
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])

            result = cv.drawMatchesKnn(input_img, kp1, data_img, kp2, good_matches, None,
                                       flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

            plt.title('SIFT')
            plt.suptitle(img_name)
            plt.imshow(result)
            if show_img:
                plt.show()
            save_img('sift', directory, img_name)


def save_img(dir_to_save, directory, img_name):
    path = 'output'

    if not os.path.isdir(path):
        os.mkdir(path)

    path += '/' + dir_to_save

    if not os.path.isdir(path):
        os.mkdir(path)

    path += '/' + directory

    if not os.path.isdir(path):
        os.mkdir(path)

    plt.savefig(path + '/' + img_name, format='jpg')


def delete_imgs():
    path = 'output'
    if os.path.exists(path):
        for first_dir in os.listdir(path):
            for second_dir in os.listdir(path + '/' + first_dir):
                for file in os.listdir(path + '/' + first_dir + '/' + second_dir):
                    os.remove(path + '/' + first_dir + '/' + second_dir + '/' + file)


if __name__ == '__main__':
    delete_imgs()
    show_each_img = False
    execute_matcher_sift(show_each_img)
    execute_matcher_orb(show_each_img)
