from typing import Callable

from matplotlib import pyplot as plt

from src.database.service import database
from src.loader.service import Loader
from src.network.schemas import TreinamentoInput
from src.neuron.layer import InputLayer, HiddenLayer, OutputLayer
from src.network.utils import gerar_grafico, distribui_valores
import random


class NeuralNetwork:

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            activation_functions: list[Callable],
            learning_rate_function: Callable,
            initial_learning_rate: float,
            dropout_rate: float,
            max_epochs: int
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_functions = activation_functions
        self.dropout_rate = dropout_rate
        self.max_epochs = max_epochs

        self.initial_learning_rate = initial_learning_rate  # usada para o decaimento da taxa de aprendizado
        self.learning_rate = initial_learning_rate  # usada para o treinamento
        self.learning_rate_function = learning_rate_function

        self.input_layer = InputLayer()
        self.hidden_layer = HiddenLayer(
            num_neurons=hidden_size,
            activation_function=activation_functions[0],
            len_input=input_size
        )
        self.output_layer = OutputLayer(
            num_neurons=output_size,
            activation_function=activation_functions[1],
            len_input=hidden_size
        )

    @staticmethod
    async def obter_dados_treinamento(
            imgs_source: str,
            label_source: str) \
            -> tuple[list[list[float]], list[list[float]]]:
        """Obtém os dados e rótulos para treinamento da rede neural"""

        # Obtendo dados para o treinamento
        data = await Loader.carregar_todas_imagens(imgs_source)
        labels = await Loader.carregar_todos_rotulos(label_source)
        # Verificando se os tamanhos são diferentes
        if len(data) != len(labels):
            raise ValueError("Dados e rotulos com tamanhos diferentes")
        # Retornando dados para treinamento
        return data, labels

    @staticmethod
    async def compute_mean_squared_error(
            all_outputs: list[list[float]],
            all_expected_outputs: list[list[float]]
    ) -> float:
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

    @staticmethod
    async def apply_dropout(hidden_layer_output: list[float], dropout_rate: float) -> list[float]:
        """Aplica dropout na camada oculta"""
        if dropout_rate < 0 or dropout_rate > 1:
            raise ValueError("Taxa de dropout deve estar entre 0 e 1")

        if dropout_rate == 0:
            return hidden_layer_output

        dropout_mask = [1 if random.random() > dropout_rate else 0 for _ in hidden_layer_output]

        # Aplica dropout e ajusta a saída para compensar a redução de neurônios (inverted dropout)
        return [(value * mask) / (1.0 - dropout_rate) for value, mask in zip(hidden_layer_output, dropout_mask)]

    async def get_output(self, entrada: list[float]) -> list[int]:
        """Dado uma entrada, calcula a saída da rede neural"""

        hidden_inputs = await self.input_layer.feed_forward(entrada)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        output = await self.output_layer.feed_forward(hidden_outputs)

        maximo = max(output)
        output = [1 if value == maximo else 0 for value in output]

        # output = [1 if value >= 0.75 else 0 for value in output]

        return output

    async def train_one_sample(self, inputs: list[float], expected_outputs: list[float]) -> None:
        """Treina a rede neural com um exemplo de entrada e saída esperada"""

        # Feed forward
        hidden_inputs = await self.input_layer.feed_forward(inputs)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        hidden_outputs_after_dropout = await self.apply_dropout(hidden_outputs, self.dropout_rate)
        outputs = await self.output_layer.feed_forward(hidden_outputs_after_dropout)

        # Back propagation
        hidden_error = await self.output_layer.back_propagation(
            inputs=hidden_outputs,
            outputs=outputs,
            expected_outputs=expected_outputs,
            learning_rate=self.learning_rate
        )
        await self.hidden_layer.back_propagation(
            inputs=hidden_inputs,
            erros_output=hidden_error,
            learning_rate=self.learning_rate
        )

    async def do_one_epoch(self, inputs: list[list[float]], expected_outputs: list[list[float]]) -> None:
        """Treina uma época da rede neural"""

        # Embaralhando os dados sem alterar a ordem
        c = list(zip(inputs, expected_outputs))
        random.shuffle(c)
        inputs, expected_outputs = zip(*c)

        if len(inputs) != len(expected_outputs):
            raise ValueError("Listas de inputs e outputs com tamanhos diferentes")

        for entrada, expected in zip(inputs, expected_outputs):
            await self.train_one_sample(inputs=entrada, expected_outputs=expected)

    async def update_learning_rate(self, current_epoch: int, decay_function: Callable) -> None:
        """Atualiza a taxa de aprendizado baseado em uma função de decaimento"""
        kwargs = {
            "max_epochs": self.max_epochs,
            "epoch": current_epoch,
            "initial_learning_rate": self.initial_learning_rate,
            "current_learning_rate": self.learning_rate}

        self.learning_rate = await decay_function(**kwargs)

    async def treinar(self, imgs_source: str, label_source: str):
        # Pegando dados para a rede
        data, labels = await NeuralNetwork.obter_dados_treinamento(imgs_source, label_source)

        if len(data) != len(labels):
            raise ValueError("Dados e rotulos com tamanhos diferentes")

        # Embaralhando os dados sem alterar a ordem
        c = list(zip(data, labels))
        random.shuffle(c)
        data, labels = zip(*c)

        porcentagem_treino = 0.75  # porcentagem de dados para treinamento
        porcentagem_validacao = 0.15  # porcentagem de dados para validação

        # Dividindo os dados em treino, validação e teste
        train_data, train_labels, validation_data, validation_labels, test_data, test_labels = \
            await distribui_valores(data, labels, porcentagem_treino, porcentagem_validacao)

        print(f"Quantidade de dados de treino: {len(train_data)}")
        print(f"Quantidade de dados de validação: {len(validation_data)}")
        print(f"Quantidade de dados de teste: {len(test_data)}")

        best_hidden_weights = self.hidden_layer.weights
        best_output_weights = self.output_layer.weights
        best_validation_accuracy = 0
        epochs_without_improvement = 0
        best_training_accuracy = 0

        acuracias = list()
        train_acuracias = list()
        training_loss = list()
        validation_loss = list()

        # Treinando a rede
        for epoch in range(self.max_epochs):
            await self.do_one_epoch(inputs=train_data, expected_outputs=train_labels)
            await self.update_learning_rate(
                current_epoch=epoch,
                decay_function=self.learning_rate_function
            )

            # parada antecipada
            outputs = []
            for entrada in validation_data:
                resultado = await self.get_output(entrada=entrada)
                outputs.append(resultado)

            correct_outputs = 0
            for resultado, esperado in zip(outputs, validation_labels):
                if resultado == esperado:
                    correct_outputs += 1

            accuracy = correct_outputs / len(validation_labels)
            acuracias.append(accuracy)

            output_treino = []
            for entrada in train_data:
                resultado = await self.get_output(entrada=entrada)
                output_treino.append(resultado)

            correct_train_outputs = 0
            for resultado, esperado in zip(output_treino, train_labels):
                if resultado == esperado:
                    correct_train_outputs += 1

            train_accuracy = correct_train_outputs / len(train_labels)
            train_acuracias.append(train_accuracy)

            training_loss.append(await self.compute_mean_squared_error(output_treino, train_labels))
            validation_loss.append(await self.compute_mean_squared_error(outputs, validation_labels))

            print(
                f"Época: {epoch + 1} | Acurácia treino: {train_accuracy} | Acurácia validação: {accuracy}, "
                f"Taxa de aprendizado: {self.learning_rate}")

            if train_accuracy > best_training_accuracy:
                best_training_accuracy = train_accuracy

            if accuracy > best_validation_accuracy:
                best_hidden_weights = self.hidden_layer.weights
                best_output_weights = self.output_layer.weights
                best_validation_accuracy = accuracy
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == 40:
                    print(f"Parando treinamento na época {epoch + 1}")
                    print(f"Melhor acurácia de validação: {best_validation_accuracy}")
                    print(f"acurácia atual: {accuracy}")
                    break

        # Inserindo dados de treinamento no banco
        await database.insert(
            TreinamentoInput(
                input_size=self.input_size,
                hidden_size=self.hidden_size,
                output_size=self.output_size,
                hidden_weights=best_hidden_weights,
                output_weights=best_output_weights,
                initial_learning_rate=self.initial_learning_rate,
                activation_functions=[func.__name__ for func in self.activation_functions],
                learning_rate_function=self.learning_rate_function.__name__,
                accuracy=best_validation_accuracy
            )
        )

        # Testando a rede
        self.hidden_layer.weights = best_hidden_weights
        self.output_layer.weights = best_output_weights

        outputs_test = []
        for entrada in test_data:
            resultado = await self.get_output(entrada=entrada)
            outputs_test.append(resultado)

        correct_test_outputs = 0
        for resultado, esperado in zip(outputs_test, test_labels):
            print(
                f"guess: {Loader.converter_binario_rotulo(resultado)} | "
                f"true: {Loader.converter_binario_rotulo(esperado)}")
            if resultado == esperado:
                correct_test_outputs += 1

        print(f"melhor acurácia de treino: {best_training_accuracy}")
        print(f"melhor acurácia de validação: {best_validation_accuracy}")
        print(f"acurácia final: {correct_test_outputs / len(test_labels)}")

        epocas = [i + 1 for i in range(0, len(acuracias))]

        # Gerando gráfico de acuracias
        plt.figure()
        gerar_grafico(epocas, acuracias, "Época", "Acurácia", "Acurácia por época")

        gerar_grafico(epocas, train_acuracias, "Época", "Acurácia", "Acurácia por época")
        plt.legend(["Validação", "Treino"])

        plt.figure()
        gerar_grafico(epocas, validation_loss, "Época", "Erro", "Erro de validação por época")
        gerar_grafico(epocas, training_loss, "Época", "Erro", "MSE por época")
        plt.legend(["Validação", "Treino"])

        plt.show()

        return 1
