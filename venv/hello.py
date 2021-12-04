import os
import zipfile
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import pandas as pd
import czifile
from sklearn.cluster import KMeans

SIMAGE_UPLOAD_FOLDER = 'D:\\Example\\simage'
RIMAGE_FOLDER = 'D:\\Example\\rimage'
SDATA_UPLOAD_FOLDER = 'D:\\Example\\sdata'
RDATA_FOLDER = 'D:\\Example\\rdata'

#SIMAGE_UPLOAD_FOLDER = '/home/julia/data/simage'
#RIMAGE_FOLDER = '/home/julia/data/rimage'
#SDATA_UPLOAD_FOLDER = '/home/julia/data/sdf'
#RDATA_FOLDER = '/home/julia/data/rdf'

IMAGE_ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'czi'])
DATA_ALLOWED_EXTENSIONS = set(['csv', 'xlsx'])

app = Flask(__name__)
app.config['SIMAGE_UPLOAD_FOLDER'] = SIMAGE_UPLOAD_FOLDER
app.config['SDATA_UPLOAD_FOLDER'] = SDATA_UPLOAD_FOLDER
app.config['RDATA_FOLDER'] = RDATA_FOLDER
app.config['RIMAGE_FOLDER'] = RIMAGE_FOLDER


def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in IMAGE_ALLOWED_EXTENSIONS


def allowed_data(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in DATA_ALLOWED_EXTENSIONS


core = np.array([[0, 0, 1, 0, 0],
                 [0, 1, 1, 1, 0],
                 [1, 1, 1, 1, 1],
                 [0, 1, 1, 1, 0],
                 [0, 0, 1, 0, 0]], dtype=np.uint8)


def stat_culc():
    data = pd.DataFrame()
    data = pd.read_csv(RDATA_FOLDER + "/clear.csv", sep=";")

    result = pd.DataFrame(index=['min_MA', 'min_ma', 'max_MA', 'max_ma', 'mean_MA', 'mean_ma', 'DX_MA', 'DX_ma',
                                 'symmetry_MA', 'symmetry_ma', 'general_sample_size'])

    result['data'] = result.index
    result = result.reset_index(drop=True)

    result["result"] = data['MA'].min(), \
                       data['ma'].min(), \
                       data['MA'].max(), \
                       data['ma'].max(), \
                       data['MA'].mean(), \
                       data['ma'].mean(), \
                       data['MA'].std(), \
                       data['ma'].std(), \
                       data['MA'].skew(), \
                       data['ma'].skew(), \
                       len(data.index)
    result.to_csv(RDATA_FOLDER + '/stat_culc.csv', sep=';', index=False)


def segment_index(dir, index: int):
    segment_file(dir[index])


def e_d(image, it):
    image = cv2.erode(image, core, iterations=it)
    image = cv2.dilate(image, core, iterations=it)
    image = cv2.morphologyEx(image, cv2.MORPH_OPEN, core, iterations=1)
    image = cv2.morphologyEx(image, cv2.MORPH_ERODE, core, iterations=1)
    return image


def read_image(path, file):
    if file.rsplit('.', 1)[1] == "czi":
        img = czifile.imread(path)
        return img[0, :, :, :]
    else:
        return cv2.imread(path)


def segment_file(img_file: str):
    img_path = os.path.join(SIMAGE_UPLOAD_FOLDER, img_file)
    img = read_image(img_path, img_file)
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

    for bacteria in finalContours:
        M = cv2.moments(bacteria)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])

        cv2.circle(bacteriaImg, (cx, cy), 5, (0, 0, 255), -1)
        ellipse = cv2.fitEllipse(bacteria)
        cv2.ellipse(bacteriaImg, ellipse, (0, 255, 0), 2)[2]
        (x, y), (MA, ma), ellipse = cv2.fitEllipse(bacteria)
        data.append([MA, ma])

    cv2.imwrite(RIMAGE_FOLDER + "/" + img_file.rsplit('.', 1)[0] + ".jpeg", bacteriaImg)

    data_df = pd.DataFrame(data, columns=('MA', 'ma'))
    data_df_m = data_df
    data_df_m.to_csv(RDATA_FOLDER + "/allDetectedObjects.csv", sep=";", index=False)

    data_df = data_df.drop(data_df[data_df['ma'] > 100].index)
    data_df = data_df.drop(data_df[data_df['MA'] < 10].index)
    data_df_m = data_df
    data_df_m.to_csv(RDATA_FOLDER + "/clear.csv", sep=";", index=False)

    data_df = data_df / 100

    clasterNum = 3
    k_means = KMeans(init="k-means++", n_clusters=clasterNum, n_init=12)
    k_means.fit(data_df)
    labels = k_means.labels_

    data_df_m = data_df
    data_df_m['clust'] = labels
    data_df_m.to_csv(RDATA_FOLDER + "/clear_norm_clust.csv", sep=";", index=False)
    stat_culc()


@app.route('/')
def main_page():
    return render_template("main_page.html")


def clear_folder(folder):
    files = [file for file in os.listdir(folder)]
    for file in files:
        os.remove(os.path.join(folder, file))


@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    clear_folder(SIMAGE_UPLOAD_FOLDER)
    clear_folder(RIMAGE_FOLDER)
    clear_folder(RDATA_FOLDER)
    if request.method == 'POST' and 'files' in request.files:
        for file in request.files.getlist('files'):
            if file and allowed_image(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['SIMAGE_UPLOAD_FOLDER'], filename))
        return redirect(url_for('image_functions'))
    return render_template("upload_image.html")


@app.route('/upload_dataframe', methods=['GET', 'POST'])
def upload_dataframe():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_data(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['SDATA_UPLOAD_FOLDER'], filename))
            return redirect(url_for('data_functions'))
    return render_template("upload_dataframe.html")


@app.route('/image_functions')
def image_functions():
    return render_template("image_functions.html")


@app.route('/download_files')
def download_files():
    return render_template("download_files.html")


@app.route('/data_functions')
def data_functions():
    return render_template("data_functions.html")


@app.route('/full_analis')
def full_analis():
    return redirect(url_for('analisys'))


@app.route('/preprocess_image')
def preprocess_image():
    return "preprocess_image"


@app.route('/final_results')
def final_results():
    return "final_results"


@app.route('/analis')
def analisys():
    img_files = [file for file in os.listdir(SIMAGE_UPLOAD_FOLDER)]
    for i in range(len(img_files)):
        segment_index(img_files, i)
    return render_template("download_files.html")


@app.route('/csv_only')
def csv_only():
    old_wd = os.getcwd()
    os.chdir(RDATA_FOLDER)
    res_files = [file for file in os.listdir(RDATA_FOLDER)]
    for file in res_files:
        zipFile = zipfile.ZipFile('data.zip', 'a', zipfile.ZIP_DEFLATED)
        zipFile.write(os.path.join(RDATA_FOLDER, file))
        zipFile.close()
        if os.path.exists(os.path.join(RDATA_FOLDER, file)):
            os.remove(os.path.join(RDATA_FOLDER, file))
    os.chdir(old_wd)
    return download(RDATA_FOLDER, 'data.zip')


@app.route('/image_only')
def image_only():
    old_wd = os.getcwd()
    os.chdir(RIMAGE_FOLDER)
    res_files = [file for file in os.listdir(RIMAGE_FOLDER)]
    for file in res_files:
        zipFile = zipfile.ZipFile('images.zip', 'a', zipfile.ZIP_DEFLATED)
        zipFile.write(os.path.join(RIMAGE_FOLDER, file))
        zipFile.close()
        if os.path.exists(os.path.join(RIMAGE_FOLDER, file)):
            os.remove(os.path.join(RIMAGE_FOLDER, file))
    os.chdir(old_wd)
    return download(RIMAGE_FOLDER, 'images.zip')


@app.route('/download/<filename>')
def download(dir, filename):
    return send_from_directory(dir, filename)
