import numpy as np
import cv2
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import czifile

kernel = np.array([[0, 0, 1, 0, 0],
                   [0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0],
                   [0, 0, 1, 0, 0]], dtype=np.uint8)


def e_d(image, it):
    image = cv2.erode(image, kernel, iterations=it)
    image = cv2.dilate(image, kernel, iterations=it)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=1)
    return image


path = r"D://Example/name"
#img_files = [file for file in os.listdir(path)]


def segment_index(index: int, dir):
    segment_file(dir[index])


def segment_file(img_file: str):
    img_path = path + "\\" + img_file
    img = czifile.imread(img_path)
    img = img[0,:,:,:]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    th = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    th = e_d(th.copy(), 1)

    cnt, hie = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cntImg = th.copy()
    for contour in cnt:
        x, y, w, h = cv2.boundingRect(contour)

        if w > img.shape[1] / 2:
            continue

        else:

            cntImg = cv2.drawContours(cntImg, [cv2.convexHull(contour)], -1, 255, -1)

    cntImg = e_d(cntImg, 2)

    cnt2, hie2 = cv2.findContours(cntImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    finalContours = []
    for contour in cnt2:
        (x, y), radius = cv2.minEnclosingCircle(contour)
        center = (int(x), int(y))
        radius = int(radius)

        circleImg = np.zeros(img.shape, dtype=np.uint8)
        circleImg = cv2.circle(circleImg, center, radius, 255, -1)

        contourImg = np.zeros(img.shape, dtype=np.uint8)
        contourImg = cv2.drawContours(contourImg, [contour], -1, 255, -1)

        union_inter = cv2.bitwise_xor(circleImg, contourImg)

        ratio = np.sum(union_inter == 255) / np.sum(circleImg == 255)

        if ratio > 0.55:
            finalContours.append(contour)

    finalContours = np.asarray(finalContours)

    bacteriaImg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    data = []
    #cent = []

    for bacteria in finalContours:
        M = cv2.moments(bacteria)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        # bacteriaImg = cv2.circle(bacteriaImg, (cx, cy), 5, (0, 0, 255), -1)
        cv2.circle(bacteriaImg, (cx, cy), 5, (0, 0, 255), -1)
        #ellipse = cv2.fitEllipse(bacteria)
        (x, y), (MA, ma), ellipse = cv2.fitEllipse(bacteria)
        #cv2.ellipse(bacteriaImg, ellipse, (0, 255, 0), 2)[2]
        #cent.append([x, y])
        data.append([MA, ma])

    # cent_df = pd.DataFrame(cent, columns=('x', 'y'))
    data_df = pd.DataFrame(data, columns=('MA', 'ma'))
    data_df.to_csv("D://Example/res/allDetectedObjects.csv", sep=";", index=False)

    data_df = data_df.drop(data_df[data_df['ma'] > 100].index)
    data_df = data_df.drop(data_df[data_df['MA'] < 10].index)
    data_df.to_csv("D://Example/res/clear.csv", sep=";", index=False)

    # print(data_df)
    data_df = data_df / 100

    clasterNum = 3
    k_means = KMeans(init="k-means++", n_clusters=clasterNum, n_init=12)
    k_means.fit(data_df)
    labels = k_means.labels_

    data_df['clust'] = labels
    data_df.to_csv("D://Example/res/clear_norm_clust.csv", sep=";", index=False)

    # print(data_df)
    # print(data_df.groupby('clust')['clust'].count())


    #plt.scatter(data_df['ma'], data_df['MA'], c=k_means.labels_.astype(float))
    #plt.show()

#for i in range(len(img_files)):
#    segment_index(i)
