from flask import Flask,  request, redirect, url_for, render_template
import os
from urllib import response
from werkzeug.utils import secure_filename
import logging
import librosa
import numpy as np
from data import ser

UPLOAD_FOLDER = '.\\static\\audio'
ALLOWED_EXTENSIONS = {'wav','mp3'}
FILE_SIZE = 50

def feature_extraction(file_path):
    data, sample_rate = librosa.load(file_path)
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result=np.hstack((result, zcr)) # stacking horizontally
    

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft)) # stacking horizontally
    

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc)) # stacking horizontally
   

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms)) # stacking horizontally
    

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel)) # stacking horizontally
    return result
    


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
            features = feature_extraction(file_path)
            app.logger.info(features)
            s = ser(file_path)
            result = {}
            result['wave_path'] = s.get_waveshow()
            result['svm'] = s.svm_prediction()
            result['rf'] = s.rf_prediction()
            result['mlp'] = s.mlp_prediction()
            result['data'] = s.get_data()
            result['audio_path'] =  file_path[1:]
            app.logger.info(result)
        return render_template('result.html',value=result)
    if request.method == 'GET':
       return render_template('index.html')
@app.route('/result',methods=['GET'])
def handle_result_page():
    if request.method == 'GET':
        return render_template('result.html')

@app.route('/team',methods=['GET'])
def handle_team():
    if request.method == 'GET':
        return render_template('team.html')
if __name__ == "__main__":
    app.run(debug=True)