import os
from flask import Flask, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from analis import *


UPLOAD_FOLDER = 'D:\\Example\\name\\'
ALLOWED_EXTENSIONS = set(['jpg', 'jpeg', 'czi'])


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('preprocessed_file')) #redirect(url_for('uploaded_file', filename=filename))
    return render_template("func_page.html")

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'],
                               filename)

@app.route('/processed')
def preprocessed_file():
    return '''
    <!DOCTYPE html>
    <head>
        <meta charset="UTF-8">
        <title>Анализ изображений</title>
    </head>
    <body>
        <p style="text-align: center">
        <button> <a href=click>Данные о бактериях</a></button>
    </p>
    </body>
    </html>
    '''

@app.route('/click')
def on_click():
    return redirect(url_for('analisys'))

@app.route('/analisys')
def analisys():
    img_files = [file for file in os.listdir('D://Example/name')]
    for i in range(len(img_files)): segment_index(i, img_files)
    return '''
    <!DOCTYPE html>
    <head>
        <meta charset="UTF-8">
        <title>Анализ изображений</title>
    </head>
    <body>
        <p style="text-align: center">
            <div> Данные анализа успешно сохранены.</div>
        </p>
    </body>
    </html>
    '''