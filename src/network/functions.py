import numpy


class NeuralNetwork():

    @staticmethod
    def load_source(path: str) -> numpy.ndarray:
        return numpy.load(path, allow_pickle=True)