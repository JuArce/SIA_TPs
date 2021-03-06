import json


class Config:

    def __init__(self, string):
        config = json.loads(string)
        config = json.loads(string)
        self.output_layer_qty = config.get('output_layer_qty')
        self.max_iterations = int(config.get('max_iterations'))
        self.initial_radius = int(config.get('initial_radius'))
        self.learning_rate = float(config.get('learning_rate'))
