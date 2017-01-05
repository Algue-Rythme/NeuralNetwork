#include <cmath>

#include "ExtraParameters.h"

using namespace ActivationFunction;
using namespace std;

const Neuron ActivationFunction::Perceptron(perceptron, constant(0.));
const Neuron ActivationFunction::Sigmoid(vectorize(sigmoid), vectorize(sigmoid_derivative));
const Neuron ActivationFunction::Tanh(vectorize(ActivationFunction::tanh), vectorize(tanh_derivative));
const Neuron ActivationFunction::ReLU(reLU, perceptron);

GridTransformer ActivationFunction::vectorize(const std::function<float (float)>& unary) {
	return [=](const LayerGrid& z, LayerGrid& output) {
		output = z.unaryExpr(unary);
	};
}

GridTransformer ActivationFunction::constant(float v) {
	return [=](const LayerGrid&, LayerGrid& grid) {grid.setConstant(v);};
}

void ActivationFunction::perceptron(const LayerGrid& z, LayerGrid& output) {
	LayerGrid full0(z.dimensions()); full0.setZero();
	LayerGrid full1(z.dimensions()); full1.setConstant(1.f);
	output = (z > 0.f).select(full1, full0);
}

void ActivationFunction::reLU(const LayerGrid& z, LayerGrid& output) {
	output = z.cwiseMax(0.f);
}

float ActivationFunction::sigmoid(float z) {
	return 1.f / (1.f + exp(-z));
}

float ActivationFunction::sigmoid_derivative(float z) {
	return sigmoid(z)*(1-sigmoid(z));
}

float ActivationFunction::tanh(float z) {
	return 2.f*sigmoid(2.f*z) - 1.f;
}

float ActivationFunction::tanh_derivative(float z) {
	return 4*sigmoid_derivative(2*z);
}

void CostFunction::quadraticCost(const LayerGrid& output, const LayerGrid& label, LayerGrid& cost) {
	cost = output - label;
}

ExtraParameters::ExtraParameters(float _eta, float _mu): eta(_eta), mu(_mu) {}
