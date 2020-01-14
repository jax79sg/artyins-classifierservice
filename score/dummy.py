from score.model import ModelReport
from schema import Schema
from schema import Or
import os
import numpy as np
from config import InferenceConfig
import pickle
class DummyModel(ModelReport):
    # Note that this is overridden cos the one defined is meant for Keng On's use case 
    input_dataschema = Schema({'sepal_length': float,
                           'sepal_width': float,
                           'petal_length': float,
                           'petal_width': float})
    # Note that this is overriden cos the one defined is meant for Keng On's use case
    output_dataschema = Schema({'species': Or("setosa", 
                                          "versicolor", 
                                          "virginica")})
    def __init__(self,config=None):
        if config == None:
           config = InferenceConfig() 

    def predict(self, mydata):
        return ("TRAINING")

if __name__=="__main__":
    mymodel =DummyModel()
    classification=mymodel.predict(data)
    print(classification)
	
