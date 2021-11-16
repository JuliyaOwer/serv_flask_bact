import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from analis import *


SIMAGE_UPLOAD_FOLDER = 'D:\\Example\\simage\\'
RIMAGE_FOLDER = ''
SDATA_UPLOAD_FOLDER = 'D:\\Example\\sdata\\'
RDATA_FOLDER = ''


IMAGE_ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'czi'])
DATA_ALLOWED_EXTENSIONS = set(['csv', 'xlsx'])


app = Flask(__name__)
app.config['SIMAGE_UPLOAD_FOLDER'] = SIMAGE_UPLOAD_FOLDER
app.config['SDATA_UPLOAD_FOLDER'] = SDATA_UPLOAD_FOLDER

def allowed_image(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in IMAGE_ALLOWED_EXTENSIONS

def allowed_data(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in DATA_ALLOWED_EXTENSIONS


@app.route('/', methods=['GET', 'POST'])
def main_page():
    return render_template("func_page.html")

"""
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)
"""


################################
#Переход по страничкам с главной
################################

@app.route('/upload_file', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
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
    return render_template("functions.html")

@app.route('/data_functions')
def data_functions():######### ПОМЕНЯТЬ ОТОБРАЖАЕМЫЙ ШАБЛОН ################
    return render_template("functions.html")


@app.route('/full_analis')
def full_analis():
    return redirect(url_for('analisys'))

@app.route('/preprocess_image')
def preprocess_image():
    return "preprocess_iamge"

@app.route('/final_results')
def final_results():
    return "final_results"

#########################


@app.route('/analis')
def analisys():
    return "full_analis"
