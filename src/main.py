from network.functions import NeuralNetwork

foo = NeuralNetwork.load_source(path="src/files/X.npy")

print(foo[0])
print(type(foo))