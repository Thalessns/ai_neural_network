import math


class LearningRateFunctions:

    @staticmethod
    async def constant(**kwargs) -> float:
        return kwargs["current_learning_rate"]

    @staticmethod
    async def exponential(**kwargs) -> float:
        decay_rate = 0.01
        return kwargs["current_learning_rate"] * math.exp(-decay_rate * kwargs["epoch"])

    @staticmethod
    async def time_based(**kwargs) -> float:
        decay_rate = 0.01
        return kwargs["current_learning_rate"] / (1 + decay_rate * kwargs["epoch"])

    @staticmethod
    async def linear(**kwargs) -> float:
        decay_rate = kwargs["epoch"] / kwargs["max_epochs"]
        return kwargs["initial_learning_rate"] * (1 - decay_rate)


decay_functions = LearningRateFunctions()


class ActivationFunctions:

    @staticmethod
    async def softmax(inputs: list[float]) -> list[float]:
        # Evita overflow
        maximo = max(inputs)
        exps = [math.exp(element - maximo) for element in inputs]
        soma = sum(exps)
        return [value / soma for value in exps]

    @staticmethod
    async def relu(x: float) -> float:
        return max(0.0, x)

    @staticmethod
    async def derivative_relu(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    @staticmethod
    async def elu(x: float, alpha: float = 1.0) -> float:
        return x if x >= 0 else alpha * (x * math.exp(x) - 1)

    @staticmethod
    async def derivative_elu(x: float, alpha: float = 1.0) -> float:
        return 1.0 if x >= 0 else math.exp(x) * (x + 1) * alpha

    @staticmethod
    async def swish(x: float) -> float:
        return x / (1 + math.exp(-x))

    @staticmethod
    async def derivative_swish(x: float) -> float:
        return (math.exp(x) * (x + math.exp(x) + 1)) / math.pow((math.exp(x) + 1), 2)

    @staticmethod
    async def selu(x: float, alpha: float = 1.0, beta: float = 1.0) -> float:
        return alpha * (max(0.0, x) + min(0.0, beta * (math.exp(x) - 1)))

    @staticmethod
    async def derivative_selu(x: float, alpha: float = 1.0, beta: float = 1.0) -> float:
        if x <= 0 and beta * math.exp(x) < beta:
            return alpha * beta * math.exp(x)
        if x > 0 and beta * math.exp(x) >= beta:
            return alpha
        if x > 0 and beta * math.exp(x) < beta:
            return alpha + alpha * beta * math.exp(x)

    @staticmethod
    async def soft_plus(x: float, beta: float = 1.0) -> float:
        return (1 / beta) * math.log(1 + math.exp(beta * x))

    @staticmethod
    async def derivative_soft_plus(x: float, beta: float = 1.0) -> float:
        return math.exp(beta * x) / (math.exp(beta * x) + 1)

    @staticmethod
    async def hard_swish(x: float) -> float:
        if x < -3:
            return 0
        elif x >= 3:
            return x
        return (x * (x + 3)) / 6

    @staticmethod
    async def derivative_hard_swish(x: float) -> float:
        if x < -3:
            return 0
        if x >= 3:
            return 1
        return (2 * x + 3) / 6

    @staticmethod
    async def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    @staticmethod
    async def derivative_sigmoid(x: float) -> float:
        exp_neg = math.exp(-x)
        return exp_neg / math.pow((1 + exp_neg), 2)

    @staticmethod
    async def soft_sign(x: float) -> float:
        return x / (1 + math.fabs(x))

    @staticmethod
    async def derivative_soft_sign(x: float) -> float:
        return 1 / math.pow((1 + math.fabs(x)), 2)

    @staticmethod
    async def tanh(x: float) -> float:
        return math.tanh(x)

    @staticmethod
    async def derivative_tanh(x: float) -> float:
        return 1 / math.pow(math.cosh(x), 2)

    @staticmethod
    async def hard_sigmoid(x: float) -> float:
        if x <= -3:
            return 0
        if x > 3:
            return 1
        return x / 6 + 1 / 2

    @staticmethod
    async def derivative_hard_sigmoid(x: float) -> float:
        if x <= 3 or x > 3:
            return 0
        return 1 / 6

    @staticmethod
    async def tanh_shrink(x: float) -> float:
        return x - math.tanh(x)

    @staticmethod
    async def derivative_tanh_shrink(x: float) -> float:
        return 1 - 1 / math.pow(math.cosh(x), 2)

    @staticmethod
    async def soft_shrink(x: float, lamb: float = 0.5) -> float:
        if x > lamb:
            return x - lamb
        elif x < -lamb:
            return x + lamb
        return 0

    @staticmethod
    async def derivative_soft_shrink(x: float, lamb: float = 0.5) -> float:
        if x > lamb or x < -lamb:
            return 1
        return 0

    @staticmethod
    async def hard_shrink(x: float, lamb: float = 0.5) -> float:
        if (x > lamb) or (x < -lamb):
            return x
        return 0

    @staticmethod
    async def derivative_hard_shrink(x: float, lamb: float = 0.5) -> float:
        if (x > lamb) or (x < -lamb):
            return 1
        return 0


activation_functions = ActivationFunctions()
