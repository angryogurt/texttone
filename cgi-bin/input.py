#!/usr/bin/env python3
from keras.models import load_model
import pickle
from study.getdata import get_sequences, preprocess_text
import cgi
import os


def toFixed(numObj, digits=0):
    return f"{numObj:.{digits}f}"


def checkString(inputString):
    path = os.path.join(os.path.dirname(__file__),'..','models','cnn','CNN.h5')
    model = load_model(path)
    path = os.path.join(os.path.dirname(__file__),'..','models','tokenizer.pickle')
    with open(path, 'rb') as handle:
        tokenizer = pickle.load(handle)
    testString = [preprocess_text(inputString)]
    testStringSeq = get_sequences(tokenizer, testString)
    score = model.predict(testStringSeq, verbose=0)
    answer = score[0][0]*100
    return toFixed(answer, 2)


storage = cgi.FieldStorage()
data = storage.getvalue('data')
print('Status: 200 OK')
print('Content-Type: text/plain')
print('')
if data is not None:
    print(checkString(data))
