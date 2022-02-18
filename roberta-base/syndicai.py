import json

from transformers import pipeline


class PythonPredictor:
    def __init__(self, config):
        """This method is required. It is called once before the API 
        becomes available. It performes the setup such as downloading / 
        initializing the model.

        :param config (required): Dictionary passed from API configuration.
        """
        self.unmasker = pipeline('fill-mask', model='roberta-base')
    
    def predict(self, payload):
        """This method is required. It is called once per request. 
        Preprocesses the request payload, runs inference, and 
        postprocesses the inference output.

        :param payload (optional): The request payload
        :returns : Prediction or a batch of predictions.
        """
        output = self.unmasker(payload["text"])
        return json.dumps(output)
