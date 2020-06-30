from score.topicmodel import TopicModel
import pandas as pd
mymodel = TopicModel()
mydata="This is a weather for Singapore"
classification=mymodel.predict(mydata)
print(classification)
