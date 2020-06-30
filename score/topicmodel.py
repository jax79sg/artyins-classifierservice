from score.model import ModelReport
from schema import Schema
from schema import Or
import os
import numpy as np
from config import InferenceConfig
import pickle
import sys
#import dataiku
import numpy as np
import pandas as pd
import sklearn as sk
#import dataiku.core.pandasutils as pdu
#from dataiku.doctor.preprocessing import PCA
from collections import defaultdict, Counter
import joblib
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition  import TruncatedSVD

text_svds = joblib.load('./score/svd.joblib')
clf=joblib.load('./score/logisticRegression.joblib')    
target_map = {u'TRAINING': 0, u'NOTTRAINING': 1}


class TopicModel(ModelReport):
    
    #Helper functions
    def coerce_to_unicode(self,x):
        if sys.version_info < (3, 0):
            if isinstance(x, str):
                return unicode(x,'utf-8')
            else:
                return unicode(x)
        else:
            return str(x)
    
    def preprocess(self,ml_dataset):
        #Preprocess test dataset
        ml_dataset[u'OBSERVATIONS'] = ml_dataset[u'OBSERVATIONS'].apply(self.coerce_to_unicode)
        text_transformed = text_svds.transform(HashingVectorizer(n_features=100000).transform(ml_dataset[u'OBSERVATIONS']))
    
        n_components = 50

        for i in range(0, n_components):
               ml_dataset[u'OBSERVATIONS' + ":text:" + str(i)] = text_transformed[:,i]
        ml_dataset.drop(u'OBSERVATIONS', axis=1, inplace=True)
        ml_dataset.head(2)
        return ml_dataset


    def __init__(self,config=None):
        if config == None:
           config = InferenceConfig() 
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, config.MODEL_DIR, config.MODEL_FILE), 'rb')
        self._svm_model = pickle.load(file)
        file.close()

    def predict(self, content):
        mydata=[]
        mydata.append(content)
        ml_dataset = pd.DataFrame(mydata,  columns =[u'OBSERVATIONS'])
        ml_dataset=self.preprocess(ml_dataset)
        #Perform prediction
        _predictions = clf.predict(ml_dataset)
        predictions = pd.Series(data=_predictions, index=ml_dataset.index, name='predicted_value')
        inv_map = { target_map[label] : label for label in target_map}
        results=predictions.map(inv_map)
        return results[0]

#if __name__=="__main__":
#    mymodel = TopicModel()
#    mydata=['This is a weather for Singapore','This is a training notice']
#    ml_dataset = pd.DataFrame(mydata,  columns =[u'OBSERVATIONS']) 
#    classification=mymodel.predict(ml_dataset)
#    print(classification)
	
