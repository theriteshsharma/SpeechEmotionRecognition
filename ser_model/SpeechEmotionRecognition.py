
import librosa
import soundfile
import os, glob, pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

class SER:
    modelPath    = ""
    fileName = ''
    features = ''
    model = ''
    emotions = {
        '01':'neutral',
        '02':'calm',
        '03':'happy',
        '04':'sad',
        '05':'angry',
        '06':'fearful',
        '07':'disgust',
        '08':'surprised'
    }

    observed_emotions = ['calm', 'happy', 'fearful', 'disgust']
    
    def __init__(self,modelPath):
       self.modelPath = modelPath

    def load_model(self):
        with open(self.modelPath, 'rb') as pickle_file:
            self.model = pickle.load(pickle_file)
      

    def extract_feature(self,file_name, mfcc, chroma, mel):
        with soundfile.SoundFile(file_name) as sound_file:
            X = sound_file.read(dtype="float32")
            sample_rate=sound_file.samplerate
            if chroma:
                stft=np.abs(librosa.stft(X))
            result=np.array([])
            if mfcc:
                mfccs=np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
                result=np.hstack((result, mfccs))
            if chroma:
                chroma=np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
                result=np.hstack((result, chroma))
            if mel:
                mel=np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
                result=np.hstack((result, mel))
            self.features =  result

    def find_emotion(self,fileName=''):
        if fileName != '':
            self.extract_feature(fileName,mfcc=True,chroma=True,mel=True)
            
            emotion = self.model.predict(self.features.reshape(1,-1))
            return emotion
   









