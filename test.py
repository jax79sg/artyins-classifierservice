import unittest

class TestModels(unittest.TestCase):

    def test_testmodel(self):
        from score.testmodel import IrisSVCModel
        mymodel = IrisSVCModel()
        data=dict(sepal_length=1.0,sepal_width=2.0,petal_length=3.0,petal_width=4.0)
        classification=mymodel.predict(data)
        print(classification)

    def test_modifiedtopicmodel(self):
        pass #Wei Deng to insert

    def test_bertmodel(self):
        pass #Kah Siong to insert

if __name__ == '__main__':
    unittest.main()
