import json


class Config:

    def __init__(self, string):
        config = json.loads(string)
        self.epochs = int(config.get('epochs'))
        self.learning_rate = float(config.get('learning_rate'))
