import neptune
import os

class NeptuneLogger():
    def __init__(self, debug: bool = False):
        self.run = neptune.init_run(
            project="slime67/ozn-image-matching",
            api_token=os.environ['NEPTUNE_API_TOKEN'],
            mode="debug" if debug else None
        )
    def add_scalar(self, name, value, step):
        self.run[name].append(value)