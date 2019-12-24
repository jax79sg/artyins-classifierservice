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
