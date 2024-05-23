from typing import Callable
from src.loader.service import Loader
from src.neuron.layer import InputLayer, HiddenLayer, OutputLayer
from src.network.schemas import TreinamentoInput
from src.database.service import database
from src.network.utils import gerar_grafico


class NeuralNetwork:

    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            activation_functions: list[Callable],
            learning_rate_function: Callable,
            initial_learning_rate: float
    ) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_functions = activation_functions

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

    async def get_output(self, entrada: list[float]) -> list[float]:
        """Dado uma entrada, calcula a saída da rede neural"""

        hidden_inputs = await self.input_layer.feed_forward(entrada)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        output = await self.output_layer.feed_forward(hidden_outputs)

        # use max to transform output in one hot encoding
        output = [1 if value == max(output) else 0 for value in output]

        return output

    async def train_one_sample(self, inputs: list[float], expected_outputs: list[float]) -> None:
        """Treina a rede neural com um exemplo de entrada e saída esperada"""

        # Feed forward
        hidden_inputs = await self.input_layer.feed_forward(inputs)
        hidden_outputs = await self.hidden_layer.feed_forward(hidden_inputs)
        outputs = await self.output_layer.feed_forward(hidden_outputs)

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

    async def treinar(self, imgs_source: str, label_source: str):
        # Pegando dados para a rede
        data, labels = await NeuralNetwork.obter_dados_treinamento(imgs_source, label_source)

        if len(data) != len(labels):
            raise ValueError("Dados e rotulos com tamanhos diferentes")

        porcentagem_treino = 0.75  # porcentagem de dados para treinamento
        porcentagem_validacao = 0.1  # porcentagem de dados para validação
        cutoff_training = int(porcentagem_treino * len(data))
        cutoff_validation = cutoff_training + int(porcentagem_validacao * len(data))
        train_data = data[:cutoff_training]
        validation_data = data[cutoff_training:cutoff_validation]
        test_data = data[cutoff_validation:]

        train_labels = labels[:cutoff_training]
        validation_labels = labels[cutoff_training:cutoff_validation]
        test_labels = labels[cutoff_validation:]

        print(f"Quantidade de dados de treino: {len(train_data)}")
        print(f"Quantidade de dados de validação: {len(validation_data)}")
        print(f"Quantidade de dados de teste: {len(test_data)}")

        best_hidden_weights = self.hidden_layer.weights
        best_output_weights = self.output_layer.weights
        best_validation_accuracy = 0
        epochs_without_improvement = 0

        acuracias = list()

        # Treinando a rede
        max_epochs = 100
        for epoch in range(max_epochs):
            await self.do_one_epoch(inputs=train_data, expected_outputs=train_labels)
            await self.update_learning_rate(
                max_epochs=max_epochs,
                epoch=epoch,
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

            if accuracy > best_validation_accuracy:
                best_hidden_weights = self.hidden_layer.weights
                best_output_weights = self.output_layer.weights
                best_validation_accuracy = accuracy
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement == 20:
                    print(f"Parando treinamento na época {epoch}")
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

        outputs = []
        for entrada in test_data:
            resultado = await self.get_output(entrada=entrada)
            outputs.append(resultado)

        correct_outputs = 0
        for resultado, esperado in zip(outputs, test_labels):
            print(f"Resultado: {resultado} | Esperado: {esperado}")
            if resultado == esperado:
                correct_outputs += 1

        print(f"acurácia final: {correct_outputs / len(test_labels)}")

        epocas = [i + 1 for i in range(0, len(acuracias))]

        # Gerando gráfico de acuracias
        gerar_grafico(epocas, acuracias, "Época", "Acurácia", "Acurácia por época")

        return 1

