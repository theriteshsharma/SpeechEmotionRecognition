import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
UPLOAD_FOLDER = '.\\uploads'

class ser:
    def __init__(self,path):
        self.file_path = path
        self.file_name = os.path.basename(self.file_path).split('/')[-1]
        self.audio,self.sample_rate = librosa.load(self.file_path)
        self.time_duration = (1/self.sample_rate)*len(self.audio)

    def get_waveshow(self):
        plt.figure(figsize=(14,5))
        librosa.display.waveshow(self.audio,sr=self.sample_rate)
        plt.savefig(os.path.join(UPLOAD_FOLDER,self.file_name.split('.')[0]))
    
    def extract_feature(self):
        data , sample_rate  = self.audio,self.sample_rate
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
    def svm_prediction(self):
        features = self.extract_feature()
        

        filename = 'models\\svm_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        scaler = pickle.load(open('models\\std_scaler.sav','rb'))
        features = scaler.transform([features])
        result = {}
        result['prediction'] = loaded_model.predict(features)
        result['classes'] = loaded_model.classes_
        result['probability'] = loaded_model.predict_proba(features)
        return result
s = ser('uploads\\03-01-05-02-02-01-12.wav')
s.get_waveshow()
print(s.svm_prediction())