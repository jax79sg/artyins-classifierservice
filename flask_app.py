# Import libraries
import os
import sys
import random
import math
import re
import time
import numpy as np
import logging
import argparse
import json
import hashlib
import json
from time import time
from urllib.parse import urlparse
from uuid import uuid4
import requests
from flask import Flask, jsonify, request

# Root directory of the project
ROOT_DIR = os.path.abspath("../")
sys.path.append(ROOT_DIR)  # To find local version of the library

# Logging confg
logging.basicConfig(level=logging.DEBUG,handlers=[
        logging.FileHandler("{0}/{1}.log".format("/logs", "classifierservice-flask")),
        logging.StreamHandler()])

############################################################
#  Configurations
#  Inherits from config.py
############################################################
from config import InferenceConfig
config = InferenceConfig()

# Create model object in inference mode.
module = __import__(config.MODEL_MODULE, fromlist=[config.MODEL_CLASS])
my_class = getattr(module,config.MODEL_CLASS)
model = my_class(config)

#Make a prediction before starting the server (First prediction takes longer)
data=config.MODEL_SAMPLE_INPUT 
classification=model.predict(data)
logging.info('Model and weight have been loaded.')


def run_infer_content(data):
    #logging.info('Load data: %s',data)
    if isinstance(data,str):
        logging.info("Data received is string, converting to object")
        data = json.loads(data)
    allresults=[]
    for entry in data:
        results = model.predict(entry['content'])
        allresults.append({"id":entry['id'], "class":results})
    return allresults

def run_predict_flowers(data):
    #logging.info('Loading data: %s', data)
    allresults=[]
    for entry in data:
        results = model.predict(entry)
        allresults.append(results)

    return allresults


# Instantiate the Node
app = Flask(__name__)

@app.route('/predict_flowers', methods=['POST'])
def predict_flowers_get():

    if request.method == 'POST':
        request_json = request.get_json(force=True)
        result = run_predict_flowers(request_json)
        
        response_msg = json.dumps(result)
        response = {
            'results': result
        }
        return jsonify(response), 200

@app.route('/infer_content',methods=['POST'])
def infer_content_get():
     request_json = request.get_json(force=True)
     result = run_infer_content(request_json)
     response_msg = json.dumps(result)
     response = {
         'results': result
     }
     return jsonify(response), 200


@app.route('/test',methods=['GET'])
def test_get():
    logging.info("Version 0.10")
    response = {
        'message': 'ok'
    }
    return jsonify(response), 200


if __name__ == '__main__':

    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-p', '--port', default=9898, type=int, help='port to listen on')
    args = parser.parse_args()
    port = args.port

    app.run(host='0.0.0.0', port=port, debug=True)
