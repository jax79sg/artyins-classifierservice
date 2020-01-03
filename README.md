[![Classifier Service](https://github.com/jax79sg/artyins-classifierservice/raw/master/images/SoftwareArchitectureClassifierService.jpg)]()

# Classifier Service For artyins deployment architecture
This is a submodule for the artyins architecture. Please refer to [main module](https://github.com/jax79sg/artyins) for full build details.

[![Build Status](https://travis-ci.com/jax79sg/artyins-classifierservice.svg?token=BREzYzgtHGHQp4of21Xp&branch=master)](https://travis-ci.com/jax79sg/artyins-classifierservice)

Refer to [Trello Task list](https://trello.com/c/qlDHKzBN) for running tasks.

---

## Table of Contents (Optional)

- [Usage](#Usage)
- [Virtualenv](#Virtualenv)
- [Tests](#Tests)

---

## Usage
All model inferencing classes needs to implement the abstract Model class from score/model.py. An example is created in score/testmodel.py. Contributors, please ensure that you add your test codes into unitest.py (See [Tests] before you push to master branch.

### Abstract Model Class
```python
from abc import ABC, abstractmethod
class Model(ABC):
    """  An abstract base class for ML model prediction code """
    @property
    @abstractmethod
    def input_dataschema(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def output_dataschema(self):
        raise NotImplementedError()

    @abstractmethod
    #Implement the loading of model file here
    def __init__(self):
        raise NotImplementedError()

    @abstractmethod
    def predict(self, data):
        self.input_dataschema.validate(data)

```
### An example on how to implement the Abstract model class
```python
from score.model import ModelReport
from schema import Schema
from schema import Or
import os
import numpy as np
from config import InferenceConfig
import pickle
class IrisSVCModel(ModelReport):
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
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, config.MODEL_DIR, config.MODEL_FILE), 'rb')
        self._svm_model = pickle.load(file)
        file.close()

    def predict(self, mydata):
        # calling the super method to validate against the
        # input_schema
        super().predict(data=mydata)
        # converting the incoming dictionary into a numpy array 
        # that can be accepted by the scikit-learn model
        X = np.array([mydata["sepal_length"], 
                   mydata["sepal_width"], 
                   mydata["petal_length"],
                   mydata["petal_width"]]).reshape(1, -1)
        # making the prediction 
        y_hat = int(self._svm_model.predict(X)[0])
        # converting the prediction into a string that will match 
        # the output schema of the model, this list will map the 
        # output of the scikit-learn model to the string expected by 
        # the output schema
        targets = ['setosa', 'versicolor', 'virginica']
        species = targets[y_hat]
        return {"species": species}

if __name__=="__main__":
    mymodel = IrisSVCModel()
    data=dict(sepal_length=1,sepal_width=2,petal_length=3,petal_width=4)
    classification=mymodel.predict(data)
    print(classification)
```
---

## Virtualenv
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt`
```
### Clone

- Clone this repo to your local machine using 
```shell
git clone https://github.com/jax79sg/artyins-classifierservice
```
---

## Tests 
This repository is linked to [Travis CI/CD](https://travis-ci.com/jax79sg/artyins-classifierservice). You are required to write the necessary unit tests if you introduce more model classes.
### Unit Tests
```python
import unittest

class TestModels(unittest.TestCase):

    def test_testmodel(self):
        print("Running TestModel Loading and Prediction")
        from score.testmodel import IrisSVCModel
        mymodel = IrisSVCModel()
        data=dict(sepal_length=1.0,sepal_width=2.0,petal_length=3.0,petal_width=4.0)
        classification=mymodel.predict(data)

    def test_modifiedtopicmodel(self):
        print("Running empty Modified Topic Model -  Sure pass")
        pass #Wei Deng to insert


    def test_bertmodel(self):
        print("Running empty Bert Model - Sure pass")
        pass #Kah Siong to insert

if __name__ == '__main__':
    unittest.main()
```

### Web Service Test
```
#Start gunicorn wsgi server
gunicorn --bind 0.0.0.0:9898 --daemon --workers 1 wsgi:app
```
Send test POST request
```python
import requests 

TEST_TYPE = 'flowers' #Options flowers sentiments
  
# api-endpoint
URL = None
DATA = None
if TEST_TYPE=='flowers':
   URL = "http://localhost:9898/predict_flowers"
   DATA = [{"petal_width":1.0, 'petal_length':2.0,'sepal_width':3.1,'sepal_length':4.3,}]
elif (TEST_TYPE=='sentiments'): 
   URL = "http://localhost:9898/predict_sentiments"
   DATA = {'sentences':['Physical pain is like a norm to me nowadays','Its been a painful year','I still try hard to be relavant']}
  
# sending get request and saving the response as response object 
r = requests.post(url = URL, json  = DATA) 
print(r) 
# extracting results in json format 
data = r.json()
print("Data sent:\n{}\n\nData received:\n{}".format(DATA,data))
```

---

