#ifndef EXTRA_PARAMETERS_H
#define EXTRA_PARAMETERS_H

#include "NeuralNetwork.h"

namespace ActivationFunction {
	GridTransformer vectorize(const std::function<float (float)>&);
	GridTransformer constant(float);

	void perceptron(const LayerGrid&, LayerGrid&);
	void reLU(const LayerGrid&, LayerGrid&);

	float sigmoid(float);
	float sigmoid_derivative(float);
	float tanh(float);
	float tanh_derivative(float);

	extern const Neuron Perceptron;
	extern const Neuron Sigmoid;
	extern const Neuron Tanh;
	extern const Neuron ReLU;
};

namespace CostFunction {
	void quadraticCost(const LayerGrid&, const LayerGrid&, LayerGrid&);
}

struct ExtraParameters {
	ExtraParameters() = default;
	ExtraParameters(float, float);
	float eta;
	float mu;
};

#endif
