from score.model import Model
from schema import Schema
from schema import Or
import os
import numpy as np
import pickle
class IrisSVCModel(Model):
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
        if config==None:
            from config import InferenceConfig
            config=InferenceConfig()
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
	
