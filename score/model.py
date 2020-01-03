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
