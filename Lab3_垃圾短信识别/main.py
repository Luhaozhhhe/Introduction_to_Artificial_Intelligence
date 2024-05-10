import os
import joblib
os.environ["HD5_USE_FILE_LOCKING"]="FALSE"

stopwords_path=r'scu_stopwords.txt'

def read_stopwords(stopwords_path):
    stopwords=[]
    with open(stopwords_path,'r',encoding='utf-8') as f:
        stopwords=f.read()
        stopwords=stopwords.splitlines()
    return stopwords

stopwords=read_stopwords(stopwords_path)
pipeline_path='results/pipeline_now_the_best.model'
pipeline=joblib.load(pipeline_path)

def predict(message):
    label=pipeline.predict([message])[0]
    proba=list(pipeline.predict_proba([message])[0])
    return label,proba