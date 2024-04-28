import math


class ActivationFunctions:

    async def relu(x: float) -> float:
        return max(0.0, x)

    async def derivative_relu(x: float) -> float:
        return 1.0 if x > 0 else 0.0

    async def elu(x: float, alpha: float = 1) -> float:
        return x if x >= 0 else alpha * (x * math.exp(x) - 1)

    async def derivative_elu(x: float, alpha: float = 1) -> float:
        return 1.0 if x >= 0 else math.exp(x) * (x + 1) * alpha

    async def swish(x: float) -> float:
        return x / (1 + math.exp(-x))

    async def derivative_swish(x: float) -> float:
        return (math.exp(x) * (x + math.exp(x) + 1)) / math.pow((math.exp(x) + 1), 2)

    async def selu(x: float, alpha: float = 1, beta: float = 1) -> float:
        return alpha * (max(0.0, x) + min(0.0, beta * (math.exp(x) - 1)))

    async def derivative_selu(x: float, alpha: float = 1, beta: float = 1) -> float:
        if x <= 0 and beta * math.exp(x) < beta:
            return alpha * beta * math.exp(x)
        if x > 0 and beta * math.exp(x) >= beta:
            return alpha
        if x > 0 and beta * math.exp(x) < beta:
            return alpha + alpha * beta * math.exp(x)

    async def soft_plus(x: float, beta: float = 1) -> float:
        return (1 / beta) * math.log(1 + math.exp(beta * x))

    async def derivative_soft_plus(x: float, beta: float = 1) -> float:
        return math.exp(beta * x) / (math.exp(beta * x) + 1)

    async def mish(x: float, beta: float = 1):
        return x * math.tanh((1 / beta) * math.log(1 + math.exp(beta * x)))

    async def hard_swish(x: float) -> float:
        if x < -3:
            return 0
        elif x >= 3:
            return x
        return (x * (x + 3)) / 6
    
    async def derivative_hard_swish(x: float) -> float:
        if x < -3:
            return 0
        if x >= 3:
            return 1
        return (2 * x + 3) / 6

    async def sigmoid(x: float) -> float:
        return 1 / (1 + math.exp(-x))

    async def derivative_sigmoid(x: float) -> float:
        return math.exp(-x) / math.pow((1 + math.exp(-x)), 2)

    async def soft_sign(x: float) -> float:
        return x / (1 + math.fabs(x))

    async def derivative_soft_sign(x: float) -> float:
        return 1 / math.pow((1 + math.fabs(x)), 2)

    async def tanh(x: float) -> float:
        return math.tanh(x)

    async def derivative_tanh(x: float) -> float:
        return 1 / math.pow(math.cosh(x), 2)

    async def hard_sigmoid(x: float) -> float:
        if x <= -3:
            return 0
        if x > 3:
            return 1
        return x / 6 + 1 / 2

    async def derivative_hard_sigmoid(x: float) -> float:
        if x <= 3 or x > 3:
            return 0
        return 1/6

    async def tanh_shrink(x: float) -> float:
        return x - math.tanh(x)

    async def derivative_tanh_shrink(x: float) -> float:
        return 1 - 1 / math.pow(math.cosh(x), 2)

    async def soft_shrink(x: float, lamb: float) -> float:
        if x > lamb:
            return x - lamb
        elif x < -lamb:
            return x + lamb
        return 0

    async def derivative_soft_shrink(x: float, lamb: float) -> float:
        if x > lamb or x < -lamb:
            return 1
        return 0

    async def hard_shrink(x: float, lamb: float) -> float:
        if (x > lamb) or (x < -lamb):
            return x
        return 0

    async def derivative_hard_shrink(x: float, lamb: float) -> float:
        if (x > lamb) or (x < -lamb):
            return 1
        return 0


functions = ActivationFunctions()
