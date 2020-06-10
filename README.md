# Classifier Service For artyins deployment architecture
This is a submodule for the artyins architecture. Please refer to [main module](https://github.com/jax79sg/artyins) for full build details.

[![Build Status](https://travis-ci.com/jax79sg/artyins-classifierservice.svg?token=BREzYzgtHGHQp4of21Xp&branch=master)](https://travis-ci.com/jax79sg/artyins-classifierservice)
[![Container Status](https://quay.io/repository/jax79sg/artyins-classifierservice/status)](https://quay.io/repository/jax79sg/artyins-classifierservice)

Refer to [Trello Task list](https://trello.com/c/qlDHKzBN) for running tasks.

---

## Table of Contents (Optional)

- [Usage](#Usage)
- [Virtualenv](#Virtualenv)
- [Tests](#Tests)

---
## Usage
The classifier service can be called by a HTTP POST call. Primarily on http://artyins-classifierservice:9891/infer_content. It expects a json of the following format
```json
[{"id":1,"content":"adfsfswjhrafkf"},{"id":2,"content":"kfsdfjsfsjhsd"}]
```
After the content is successfully extracted, it will return a json of the following format
```python
{"results":[{"id":1,"class":"DOCTRINE"},{"id":2,"section":"class":"DOCTRINE"}]}
```

## Contributing
All model inferencing classes needs to implement the abstract Model class from score/model.py. An example is created in score/testmodel.py. All configuration should be set in config.py. Finally, add your model into the web service flask_app.py. Contributors, please ensure that you add your test codes into test.py 

### config.py
```python
class InferenceConfig():
    MODEL_SAMPLE_INPUT=dict(sepal_length=1.0,sepal_width=2.2,petal_length=3.3,petal_width=4.4)
    MODEL_MODULE="score.testmodel"
    MODEL_CLASS="IrisSVCModel"
    MODEL_DIR = "model_files"
    MODEL_FILE = "svc_iris_model.pickle"
```

### Abstract Model Class
```python
from abc import ABC, abstractmethod
from schema import Schema
class ModelReport(ABC):
    """  An abstract base class for ML model prediction code """
    @property
    @abstractmethod
    def input_dataschema(self):
        Schema({'sentence': str})

    @property
    @abstractmethod
    def output_dataschema(self):
        Schema({'category': And(str,lambda s: s in ('doctrine', 'training','personnel'))
               ,'prob': And(float,lambda n: 0 <= n <= 100)})

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
```

---

## Virtualenv
```shell
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt`
```
---

## Tests 
This repository is linked to [Travis CI/CD](https://travis-ci.com/jax79sg/artyins-classifierservice). You are required to write the necessary unit tests and edit `.travis.yml` if you introduce more model classes.
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
# api-endpoint
URL = "http://localhost:9891/test"
r = requests.get(url = URL) 
print(r) 
```

---

