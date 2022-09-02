import os
from urllib import response
from flask import Flask, flash, request, redirect, url_for
from werkzeug.utils import secure_filename
from SpeechEmotionRecognition import SER


UPLOAD_FOLDER = '.\\uploads'
ALLOWED_EXTENSIONS = {'wav','mp3'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def allowed_file(filename):
    return '.' in filename and \
    filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            upname = os.path.join(app.config['UPLOAD_FOLDER'],filename)
            file.save(upname)
            ser = SER('.\model.pkl')
            ser.load_model()
            em = ser.find_emotion(upname)
            return em[0]

            
            
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''