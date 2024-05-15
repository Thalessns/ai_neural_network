from src.loader.service import loader
from src.neuron.layer import InputLayer, HiddenLayer, OutputLayer
from src.neuron.functions import activation_functions, decay_functions
from typing import Callable
import random
import math


class NeuralNetwork:

    def __init__(self, input_size: int, hidden_size: int, output_size: int, initial_learning_rate: float):
        self.initial_learning_rate = initial_learning_rate  # usada para o decaimento da taxa de aprendizado
        self.learning_rate = initial_learning_rate  # usada para o treinamento
        self.input_layer = InputLayer()
        self.hidden_layer = HiddenLayer(num_neurons=hidden_size,
                                        activation_function=activation_functions.tanh,
                                        len_input=input_size)
        self.output_layer = OutputLayer(num_neurons=output_size,
                                        activation_function=activation_functions.tanh,
                                        len_input=hidden_size)

    @staticmethod
    async def compute_mean_squared_error(all_outputs: list[list[float]],
                                         all_expected_outputs: list[list[float]]) -> float:
        """Dado as saídas e saídas esperadas para o conjunto de dados completo, calcula o erro quadrático médio"""

        if len(all_outputs) != len(all_expected_outputs):
            raise ValueError("Listas de outputs com tamanhos diferentes")

        erros_neuronio = []
        for obtidos_neuronio, esperados_neuronio in zip(all_outputs, all_expected_outputs):
            erros_neuronio.append([obtido - esperado for obtido, esperado in zip(obtidos_neuronio, esperados_neuronio)])

        erro_instantaneo = []
        for erro_neuronioj in erros_neuronio:
            erro_instantaneo.append(sum(erro ** 2 for erro in erro_neuronioj) / 2.0)

        erro_quadratico_medio = sum(erro_instantaneo) / len(erro_instantaneo)

        return erro_quadratico_medio

    async def get_output(self, entrada: list[float]) -> list[float]:
        """Dado uma entrada, calcula a saída da rede neural"""

        hidden_inputs = await self.input_layer.feed_forward(entrada)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        output = await self.output_layer.feed_forward(hidden_outputs)
        return output

    async def train_one_sample(self, inputs: list[float], expected_outputs: list[float]) -> None:
        """Treina a rede neural com um exemplo de entrada e saída esperada"""

        # Feed forward
        hidden_inputs = await self.input_layer.feed_forward(inputs)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        outputs = await self.output_layer.feed_forward(hidden_outputs)

        # Back propagation
        hidden_error = await self.output_layer.back_propagation(inputs=hidden_outputs,
                                                                outputs=outputs,
                                                                expected_outputs=expected_outputs,
                                                                learning_rate=self.learning_rate)
        await self.hidden_layer.back_propagation(inputs=hidden_inputs,
                                                 erros_output=hidden_error,
                                                 learning_rate=self.learning_rate)

    async def do_one_epoch(self, inputs: list[list[float]], expected_outputs: list[list[float]]) -> None:
        """Treina uma época da rede neural"""

        if len(inputs) != len(expected_outputs):
            raise ValueError("Listas de inputs e outputs com tamanhos diferentes")

        for entrada, expected in zip(inputs, expected_outputs):
            await self.train_one_sample(inputs=entrada, expected_outputs=expected)

    async def update_learning_rate(self, max_epochs: int, epoch: int, decay_function: Callable) -> None:
        """Atualiza a taxa de aprendizado baseado em uma função de decaimento"""
        kwargs = {
            "max_epochs": max_epochs,
            "epoch": epoch,
            "initial_learning_rate": self.initial_learning_rate,
            "current_learning_rate": self.learning_rate}

        self.learning_rate = await decay_function(**kwargs)

    async def iniciar(self):
        # Pegando dados para a rede
        data = await loader.carregar_imagem("src/files/X_png/0.png")

        # Transformando em uma lista de valores
        data = [item for sublist in data for item in sublist]

        # Generate some random points
        num_points = 1000
        points = [[random.uniform(-3, 5), random.uniform(-3, 5)] for _ in range(num_points)]

        # Define the circle
        center = (1, 1)
        radius = 2.3
        labels = [[1 if math.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2) <= radius else 0] for x, y in points]

        train_points = points[:int(0.65 * num_points)]  # 65% dos pontos
        validation_points = points[int(0.65 * num_points):int(0.80 * num_points)]  # 15% dos pontos
        test_points = points[int(0.80 * num_points):]  # 20% dos pontos

        train_labels = labels[:int(0.65 * num_points)]
        validation_labels = labels[int(0.65 * num_points):int(0.80 * num_points)]
        test_labels = labels[int(0.80 * num_points):]

        inputs = train_points
        expected_outputs = train_labels
        validation_inputs = validation_points
        validation_expected_outputs = validation_labels

        best_hidden_weights = self.hidden_layer.weights
        best_output_weights = self.output_layer.weights
        best_validation_accuracy = 0
        epochs_without_improvement = 0

        # Treinando a rede
        max_epochs = 200
        for epoch in range(max_epochs):
            await self.do_one_epoch(inputs=inputs, expected_outputs=expected_outputs)
            await self.update_learning_rate(max_epochs=max_epochs,
                                            epoch=epoch, decay_function=decay_functions.linear)

            # parada antecipada
            outputs = []
            for entrada in validation_inputs:
                outputs.append(await self.get_output(entrada=entrada))

            correct_outputs = 0
            for resultado, esperado in zip(outputs, validation_expected_outputs):
                if round(resultado[0]) == esperado[0]:
                    correct_outputs += 1

            accuracy = correct_outputs / len(validation_expected_outputs)
            if accuracy > best_validation_accuracy:
                best_hidden_weights = self.hidden_layer.weights
                best_output_weights = self.output_layer.weights
                best_validation_accuracy = accuracy
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == 10:
                    print(f"Parando treinamento na época {epoch}")
                    print(f"Melhores pesos da camada escondida: {best_hidden_weights}")
                    print(f"Melhores pesos da camada de saída: {best_output_weights}")
                    print(f"Melhor acurácia de validação: {best_validation_accuracy}")
                    print(f"acurácia atual: {accuracy}")
                    break

        # Testando a rede
        self.hidden_layer.weights = best_hidden_weights
        self.output_layer.weights = best_output_weights

        outputs = []
        for entrada in test_points:
            outputs.append(await self.get_output(entrada=entrada))

        true_positives = 0
        true_negatives = 0
        false_positives = 0
        false_negatives = 0
        for resultado, esperado in zip(outputs, test_labels):
            if resultado[0] >= 0.5 and esperado[0] == 1:
                true_positives += 1
            elif resultado[0] < 0.5 and esperado[0] == 1:
                false_negatives += 1
            elif resultado[0] >= 0.5 and esperado[0] == 0:
                false_positives += 1
            elif resultado[0] < 0.5 and esperado[0] == 0:
                true_negatives += 1

        print(f"verdadeiros positivos: {true_positives}")
        print(f"falsos positivos: {false_positives}")
        print(f"verdadeiros negativos: {true_negatives}")
        print(f"falsos negativos: {false_negatives}")
        print(f"acurácia: {(true_positives + true_negatives) / len(test_labels)}")

        return 1
