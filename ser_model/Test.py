from SpeechEmotionRecognition import SER

ser = SER('.\model.pkl')
ser.load_model()
em = ser.find_emotion('speech-emotion-recognition-ravdess-data\\Actor_01\\03-01-01-01-02-01-01.wav')

print(em)
