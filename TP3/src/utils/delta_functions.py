from utils.PerceptronParameters import PerceptronParameters


def delta_function(perceptron_parameters: PerceptronParameters, x, y, h, o):
    return perceptron_parameters.eta * (y - o) * x


def delta_function_no_linear(perceptron_parameters: PerceptronParameters, x, y, h, o):
    return perceptron_parameters.eta * (y - o) * \
           perceptron_parameters.activation_function_derivative(h, perceptron_parameters) * x
