import librosa
import librosa.display
import os
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

from pydub import AudioSegment
from pydub.silence import detect_nonsilent

import keras
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization
from keras.utils import np_utils, to_categorical
from keras.callbacks import ModelCheckpoint

import matplotlib
matplotlib.use('agg')


UPLOAD_FOLDER = '.\\static\\img\\generated'

class ser:
    def __init__(self,path):
        self.file_path = path
        self.file_name = os.path.basename(self.file_path).split('/')[-1]
        self.audio,self.sample_rate = librosa.load(self.file_path,  res_type='kaiser_fast') 
        self.time_duration = (1/self.sample_rate)*len(self.audio)
        self.scaler = pickle.load(open('models\\new_models\\scaler.pkl','rb'))
        self.encoder = pickle.load(open('models\\new_models\\encoder.pkl','rb'))

    def get_waveshow(self):
        plt.figure(figsize=(14,5))
        librosa.display.waveshow(self.audio,sr=self.sample_rate)
        plt.savefig(os.path.join(UPLOAD_FOLDER,self.file_name.split('.')[0]))
        return os.path.join(UPLOAD_FOLDER,self.file_name.split('.')[0])[1:]+'.png'
    
    def extract_features(self):
        result = np.array([])
        #mfccs = librosa.feature.mfcc(y=data, sr=22050, n_mfcc=42) #42 mfcc so we get frames of ~60 ms
        mfccs = librosa.feature.mfcc(y=self.audio, sr=22050, n_mfcc=58)
        mfccs_processed = np.mean(mfccs.T,axis=0)
        result = np.array(mfccs_processed)
        return result

    
    def get_data(self):
        result  = {}
        result['sample_rate'] = "{:.3f}".format(self.sample_rate)
        result['time_duration_per_sample'] =  "{:.9f}".format(1/self.sample_rate)
        result['total_duration'] =  "{:.3f}".format((1/self.sample_rate)*len(self.audio))
        return result
    
    def match_target_amplitude(sound, target_dBFS):
        change_in_dBFS = target_dBFS - sound.dBFS
        return sound.apply_gain(change_in_dBFS)
    

    def split_and_process(self):
        #Convert wav to audio_segment
        audio_segment = AudioSegment.from_wav(self.file_path,"wav")

        #normalize audio_segment to -20dBFS 
        normalized_sound = audio_segment
        print("length of audio_segment={} seconds".format(len(normalized_sound)/1000))

        #Print detected non-silent chunks, which in our case would be spoken words.
        nonsilent_data = detect_nonsilent(normalized_sound, min_silence_len=1000, silence_thresh=audio_segment.dBFS - 16, seek_step=1)

        #convert ms to seconds
        print("start,Stop")
        print(nonsilent_data)
        result = []
        for i,chunk in enumerate(nonsilent_data):
            chunk_name = f'static\\audio\\genchunk\\chunk{i}.wav'
            chunk_audio = audio_segment[chunk[0]:chunk[1]]
            chunk_audio.export(chunk_name,format="wav")
            temp = ser(chunk_name)
            result.append(([chunk/1000 for chunk in chunk], temp.cnn_prediction()))

        
        plt.figure(figsize=(14,5))
        plot_array = normalized_sound.get_array_of_samples()
        plot_array = np.array(plot_array)
        frame_rate = normalized_sound.frame_rate

        duration = len(plot_array)/frame_rate

        time = np.linspace(
                0, # start
                len(plot_array) / frame_rate,
                num = len(plot_array)
            )

        emot = {
            'happy':'hotpink',
            'disgust': 'yellow',
            'calm':'lawngreen',
            'fear':'mediumblue',
            'neutral':'ghostwhite',
            'sad':'lightslategray',
            'surprise':'dodgerblue',
            'angry' : 'red',
        }

        print(time,plot_array)
        plt.plot(time,plot_array)
        for i,x in enumerate(nonsilent_data):
            plt.axvspan(x[0]/1000,x[1]/1000,alpha=0.2,color=emot[result[i][1]],label=result[i][1])
        plt.legend()
        print('test')
        plt.savefig(os.path.join(UPLOAD_FOLDER,self.file_name.split('.')[0])+"_res")

        return os.path.join(UPLOAD_FOLDER,self.file_name.split('.')[0])+"_res.png"
        
    def cnn_prediction(self):
        
        feature = self.extract_features()
       
        feature = self.scaler.transform(feature.reshape(1,-1))
        #feature = np.expand_dims(feature,axis=(0,1))
        #feature = feature.reshape(1,58,1)
        print(feature.shape)
        
        model = keras.models.load_model('models\\cnn')
        res = model.predict(feature)
        
        res = self.encoder.inverse_transform(res)
        return res[0][0]
    
    def svm_prediction(self):
        feature = self.extract_features()
        feature = self.scaler.transform(feature.reshape(1,-1))
        svm = pickle.load(open('models\\new_models\\svm.pkl','rb'))
        classes = list(svm.classes_)
        
        rslt = svm.predict_proba(feature)
        dict = {}
        for idx,each in enumerate(classes):
            dict[each] = format(rslt[0][idx],'.4f')
        rslt = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        print(rslt)
        return rslt
    
    def rf_prediction(self):
        feature = self.extract_features()
        feature = self.scaler.transform(feature.reshape(1,-1))
        rf = pickle.load(open('models\\new_models\\rf_model.pkl','rb'))
        classes = list(rf.classes_)
        
        rslt = rf.predict_proba(feature)
        dict = {}
        for idx,each in enumerate(classes):
            dict[each] = format(rslt[0][idx],'.4f')
        rslt = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        print(rslt)
        return rslt
    

    def mlp_prediction(self):
        feature = self.extract_features()
        feature = self.scaler.transform(feature.reshape(1,-1))
        mlp = pickle.load(open('models\\new_models\\mlp.pkl','rb'))
        classes = list(mlp.classes_)
        
        rslt = mlp.predict_proba(feature)
        dict = {}
        for idx,each in enumerate(classes):
            dict[each] = format(rslt[0][idx],'.4f')
        rslt = sorted(dict.items(), key=lambda x: x[1], reverse=True)
        print(rslt)
        return rslt

















'''
from pydub import AudioSegment
from pydub.silence import split_on_silence,detect_nonsilent()

myaudio = AudioSegment.from_file('uploads\\test.wav' , "wav") 
chunk_length_ms = 2000 # pydub calculates in millisec
#chunks = split_on_silence(myaudio, chunk_length_ms,silence_thresh = myaudio.dBFS - 16,with_timing=True) #Make chunks of one sec

#Export all of the individual chunks as wav files
chunks  = detect_nonsilent(myaudio,chunk_length_ms)

for i, chunk in enumerate(chunks):
    chunk_name = "chunk{0}.wav".format(i)
    print ("exporting", chunk_name)
    print(chunk)
    
    # chunk.export(chunk_name, format="wav")
    



print(len(myaudio))
import glob
finalLen = 0
for file in glob.glob('chunk*.wav'):
    temp = AudioSegment.from_file(file)
    finalLen += len(temp)
print(finalLen)

'''