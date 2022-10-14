import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
UPLOAD_FOLDER = '.\\static\\img\\generated'

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
        return os.path.join(UPLOAD_FOLDER,self.file_name.split('.')[0])[1:]+'.png'
        
    
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
        result['prediction'] = loaded_model.predict(features)[0]
        temp = {}
        probability = loaded_model.predict_proba(features)
        classes = list(loaded_model.classes_)
        dict = {}
        for idx,each in enumerate(classes):
            dict[each] = format(probability[0][idx],'.4f')

        probClass = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        
        result["probability"] = probClass
        return result
    def rf_prediction(self):
        features = self.extract_feature()
    
        filename = 'models\\rf_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        scaler = pickle.load(open('models\\std_scaler.sav','rb'))
        features = scaler.transform([features])
        result = {}
        result['prediction'] = loaded_model.predict(features)[0]
        temp = {}
        probability = loaded_model.predict_proba(features)
        classes = list(loaded_model.classes_)
        dict = {}
        for idx,each in enumerate(classes):
            dict[each] = format(probability[0][idx],'.4f')

        probClass = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        
        result["probability"] = probClass
        return result
    def mlp_prediction(self):
        features = self.extract_feature()
    
        filename = 'models\\mlp_model.sav'
        loaded_model = pickle.load(open(filename, 'rb'))
        scaler = pickle.load(open('models\\std_scaler.sav','rb'))
        features = scaler.transform([features])
        result = {}
        result['prediction'] = loaded_model.predict(features)[0]
        temp = {}
        probability = loaded_model.predict_proba(features)
        classes = list(loaded_model.classes_)
        dict = {}
        for idx,each in enumerate(classes):
            dict[each] = format(probability[0][idx],'.4f')

        probClass = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        
        result["probability"] = probClass
        return result
    def get_data(self):
        result  = {}
        result['sample_rate'] = "{:.3f}".format(self.sample_rate)
        result['time_duration_per_sample'] =  "{:.9f}".format(1/self.sample_rate)
        result['total_duration'] =  "{:.3f}".format((1/self.sample_rate)*len(self.audio))
        return result

# s = ser('uploads\\03-01-05-02-02-01-12.wav')
# s.get_waveshow()
# print(s.svm_prediction())