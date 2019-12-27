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
	
