# to use CPU
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import random
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=5,
                        inter_op_parallelism_threads=5, 
                        allow_soft_placement=True,
                        device_count = {'CPU' : 1,
                                        'GPU' : 0}
                       )
from model import get_model_emotions
from utils import clean_text, tokenize_words
from config import embedding_size, sequence_length
from preprocess import categories
from keras.preprocessing.sequence import pad_sequences


import pickle
import speech_recognition as sr
from gtts import gTTS 
import numpy as np
#from mpyg321.mpyg321 import MPyg321Player()
Questionlist=[]

mytext1 = 'Good evening Tim! How was your day!'
mytext2 = 'Hi Tim! What did you do today?'
mytext3 = 'Hi Tim,Could you describe your day?'
mytext4 = 'Hi Tim,How are you really feeling today?'
mytext5 = 'Hello Tim, Did you enjoy listening to your music recommendation?'
mytext6 = 'Good evening Tim! Did you enjoy your boxing session?'
mytext7=  'Hi Tim,Did you read anything interesting today?'
Textlist=[mytext1,mytext2,mytext3,mytext4,mytext5,mytext6,mytext7]
Textlist=np.array(Textlist)
language = 'en'
myobj1 = gTTS(text=mytext1, lang=language, slow=False)
myobj1.save("question1.mp3")
Questionlist.append("question1.mp3")
myobj2 = gTTS(text=mytext2, lang=language, slow=False)
myobj2.save("question2.mp3")
Questionlist.append('question2.mp3')
myobj3 = gTTS(text=mytext3, lang=language, slow=False)
myobj3.save("question3.mp3")
Questionlist.append('question3.mp3')
myobj4 = gTTS(text=mytext4, lang=language, slow=False)
myobj4.save("question4.mp3")
Questionlist.append('question4.mp3')
myobj5 = gTTS(text=mytext5, lang=language, slow=False)
myobj5.save("question5.mp3")
Questionlist.append('question5.mp3')
myobj6 = gTTS(text=mytext6, lang=language, slow=False)
myobj6.save("question6.mp3")
Questionlist.append('question6.mp3')
myobj7 = gTTS(text=mytext6, lang=language, slow=False)
myobj7.save("question7.mp3")
Questionlist.append('question7.mp3')
#player = MPyg321Player()

# Record Audio
r = sr.Recognizer()
with sr.Microphone() as source: 
    print("Say something!")
    
    index = random.randint(0,len(Questionlist)-1)
    c=Questionlist[index]
    os.system("mpg321 "+ c)
    
    #player.play_song("question.mp3")
    
    audio = r.listen(source)

print("Loading vocab2int")
vocab2int = pickle.load(open("Mood:Emotion Code/data/vocab2int.pickle", "rb"))

model = get_model_emotions(len(vocab2int), sequence_length=sequence_length, embedding_size=embedding_size)
model.load_weights("results/model_v1_0.59_0.76.h5")

if __name__ == "__main__":
    import argparse
    # parser = argparse.ArgumentParser(description="Emotion classifier using text")  
    # parser.add_argument("text", type=str, help="The text you want to analyze")

    # args = parser.parse_args()
    
    text = tokenize_words(clean_text(r.recognize_google(audio)), vocab2int)
    x = pad_sequences([text], maxlen=sequence_length)
    prediction = model.predict_classes(x)[0]
    
    probs = model.predict(x)[0]
    # print("hi:",index)
    print("Question asked: ",Textlist[index])
    print("You said: " + r.recognize_google(audio))
    print("Probs:")
    for i, category in categories.items():
        print(f"{category.capitalize()}: {probs[i]*100:.2f}%")
    
    print("The most dominant emotion:", categories[prediction])