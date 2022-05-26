import json


class Config:

    def __init__(self, string):
        config = json.loads(string)
        self.max_iterations = int(config.get('max_iterations'))
