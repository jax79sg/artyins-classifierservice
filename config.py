class InferenceConfig():
    GPU_COUNT = 1
    
    MODEL_SAMPLE_INPUT=dict(sepal_length=1.0,sepal_width=2.2,petal_length=3.3,petal_width=4.4)
    MODEL_MODULE="score.dummy"
    MODEL_CLASS="DummyModel"
    MODEL_DIR = "model_files"
    MODEL_FILE = "svc_iris_model.pickle"

