from flask import Flask,  request, redirect, url_for, render_template
import os
from urllib import response
from werkzeug.utils import secure_filename
import logging
import librosa
import numpy as np
from data import ser

UPLOAD_FOLDER = '.\\static\\audio'
ALLOWED_EXTENSIONS = {'wav'}
FILE_SIZE = 50


app = Flask(__name__, template_folder='template')
logging.basicConfig(level=logging.DEBUG)
@app.route('/',methods=['GET','POST'])
def handle_file_upload():
    if request.method == 'POST':
        if 'audiofile' in request.files:
            file  = request.files['audiofile']
            filename = file.filename
            file_path = os.path.join(UPLOAD_FOLDER,filename)
            file.save(file_path)
            s = ser(file_path)
            result = {}
            result['wave_path'] = s.get_waveshow()
            result['audio_path'] = file_path
            result['data'] = s.get_data()
            result['cnn'] = s.cnn_prediction()
            result['res_wave'] = s.split_and_process()
            result['svm'] = s.svm_prediction()
            result['mlp'] = s.mlp_prediction()
          
            app.logger.info(result)
        return render_template('result2.html',value=result)

    if request.method == 'GET':
       return render_template('index.html')
    
@app.route('/result',methods=['GET'])
def handle_result_page():
    if request.method == 'GET':
        return render_template('result2.html')

@app.route('/team',methods=['GET'])
def handle_team():
    if request.method == 'GET':
        return render_template('team.html')
if __name__ == "__main__":
    app.run(debug=True,host="localhost",port=80)
