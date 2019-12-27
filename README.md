[![Classifier Service](https://github.com/jax79sg/artyins-classifierservice/raw/master/SoftwareArchitecture-MonitorService.jpg)]()

# Classifier Service For artyins architecture
This is a submodule for the artyins architecture. Please refer to [main module](https://github.com/jax79sg/artyins) for full build details.

[![Build Status](https://travis-ci.com/jax79sg/artyins-classifierservice.svg?token=BREzYzgtHGHQp4of21Xp&branch=master)](https://travis-ci.com/jax79sg/artyins-classifierservice)

---

## Table of Contents (Optional)

- [Usage](#Usage)
- [Virtualenv](#Virtualenv)
- [Tests](#Tests)

---

## Usage
All model inferencing classes needs to implement the abstract Model class from score/model.py. An example is created in score/testmodel.py. Contributors, please ensure that you add your test codes into unitest.py (See [Tests] before you push to master branch.

Abstract Model Class
```
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
An example on how to implement the Abstract model class
```
from model import Model
from schema import Schema
from schema import Or
import os
import pickle
class IrisSVCModel(Model):
    # A demonstration of how to use 
    input_dataschema = Schema({'sepal_length': float,
                           'sepal_width': float,
                           'petal_length': float,
                           'petal_width': float})
    # the output of the model will be one of three strings
    output_dataschema = Schema({'species': Or("setosa", 
                                          "versicolor", 
                                          "virginica")})
    def __init__(self):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        file = open(os.path.join(dir_path, "model_files", "svc_iris_model.pickle"), 'rb')
        self._svm_model = pickle.load(file)
        file.close()

    def predict(self, data):
        # calling the super method to validate against the
        # input_schema
        super().predict(data=data)
        # converting the incoming dictionary into a numpy array 
        # that can be accepted by the scikit-learn model
        X = array([data["sepal_length"], 
                   data["sepal_width"], 
                   data["petal_length"],
                   data["petal_width"]]).reshape(1, -1)
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
	
```
---

## Virtualenv
`python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt`

### Clone

- Clone this repo to your local machine using `git clone https://github.com/jax79sg/artyins-classifierservice`
---

## Tests 
This repository is linked to [Travis CI/CD](https://travis-ci.com/jax79sg/artyins-classifierservice). You are required to write the necessary unit tests if you introduce more model classes.
Example of test.py
```
import unittest

class TestModels(unittest.TestCase):

    def test_testmodel(self):
        from score.testmodel import IrisSVCModel
        mymodel=IrisSVCModel()
        

    def test_modifiedtopicmodel(self):
        pass #Wei Deng to insert

    def test_bertmodel(self):
        pass #Kah Siong to insert

if __name__ == '__main__':
    unittest.main()
```

---

