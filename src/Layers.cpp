#include <chrono>
#include <iostream>
#include <random>

#include "Layers.h"
#include "ExtraParameters.h"

using namespace std;
using namespace Eigen;

static const unsigned seed = static_cast<unsigned int>(chrono::system_clock::now().time_since_epoch().count());
default_random_engine gen(seed);
normal_distribution<float> gaussian;

const std::array<Eigen::IndexPair<int>, 0> TensorProductWithoutContraction{};

/* Fully Connected Layer */

const std::array<IndexPair<int>, 1>
FullyConnectedLayer::backwardContraction {
	IndexPair<int>(0, 0) // match features
};

const std::array<IndexPair<int>, 3>
FullyConnectedLayer::forwardContraction {
	IndexPair<int>(3, 2), // match columns
	IndexPair<int>(2, 1), // match lines
	IndexPair<int>(1, 0) // match features
};

FullyConnectedLayer::FullyConnectedLayer(
	const Layout& _inputDim,
	int _nbNeurons,
	const Neuron& _activation):
	Layer(_inputDim, Layout(_nbNeurons, 1, 1), _activation),
	nbNeurons(_nbNeurons),
	column{nbNeurons},
	w{nbNeurons, inputDim.nbFeatures(), inputDim.nbLines(), inputDim.nbColumns()},
	b{nbNeurons},
	w_velocity(w.dimensions()), b_velocity(b.dimensions())
{
	auto initializer = [](float) { return gaussian(gen);};
	w = w.unaryExpr(initializer);
	b = b.unaryExpr(initializer);
	w_velocity.setZero();
	b_velocity.setZero();
}

void FullyConnectedLayer::forwardPropagation(
	const PropagationBlock& backwardBlock,
	PropagationBlock& currentBlock) const {
	// Tensor product : Rank(4)+Rank(3) = Rank(7)
	// Three contractions : Rank(7)-3*2 = Rank(1)
	// Then reshaped to Rank(3) -> OK
	auto z = w.contract(backwardBlock.a, forwardContraction) + b;
	currentBlock.z = z.reshape(std::array<int,3>{int(nbNeurons), 1, 1});
	activation(currentBlock.z, currentBlock.a);
}

void FullyConnectedLayer::backPropagation(
	PropagationBlock& currentBlock,
	PropagationBlock& backwardBlock) const {
	LayerGrid z_derivative;
	activation.derivative(currentBlock.z, z_derivative);
	currentBlock.delta = currentBlock.y * z_derivative;
	// Tensor product : Rank(4) + Rank(1) = Rank(5)
	// One contraction : Rank(5) - 1*2 = Rank(3) -> OK
	auto deltaColumn = currentBlock.delta.reshape(column);
	auto grid = w.contract(deltaColumn, backwardContraction);
	backwardBlock.y = grid.reshape(inputDim.dimensions());
}

Tensor<float, 1> FullyConnectedLayer::compute_pd_b(const MiniBatchBlock& blocks) const {
	LayerGrid pd_b(outputDim.dimensions()); pd_b.setZero();
	blocks.visit([&](const PropagationBlock&, const PropagationBlock& current) {
		pd_b += current.delta;
	});
	pd_b = pd_b*(1.f/blocks.miniBatchSize());
	return pd_b.reshape(column);
}

Tensor<float, 4> FullyConnectedLayer::compute_pd_w(const MiniBatchBlock& blocks) const {
	Tensor<float, 4> pd_w(w.dimensions()); pd_w.setZero();
	blocks.visit([&](const PropagationBlock& back, const PropagationBlock& current) {
		const auto& a = back.a;
		auto delta = current.delta.reshape(column);
		// Tensor product : Rank(3) + Rank(1) = Rank(4) -> OK
		// No contraction
		pd_w += delta.contract(a, TensorProductWithoutContraction);
	});
	pd_w = pd_w*(1.f/blocks.miniBatchSize());
	return pd_w;
}

void FullyConnectedLayer::updateParameters(MiniBatchBlock& blocks, const ExtraParameters& params) {
	Tensor<float, 1> pd_b = compute_pd_b(blocks);
	Tensor<float, 4> pd_w = compute_pd_w(blocks);

	w_velocity = w_velocity*(1.f-params.mu);
	w_velocity += -params.eta*pd_w;
	w += w_velocity;

	b_velocity = b_velocity*(1.f-params.mu);
	b_velocity += -params.eta*pd_b;
	b += b_velocity;
}

ostream& FullyConnectedLayer::writeParameters(ostream& out) const {
	out.write(reinterpret_cast<const char* const>(w.data()), w.size() * sizeof(decltype(w)::Scalar));
	out.write(reinterpret_cast<const char* const>(b.data()), b.size() * sizeof(decltype(w)::Scalar));
	return out;
}

istream& FullyConnectedLayer::readParameters(istream& in) {
	in.read(reinterpret_cast<char* const>(w.data()), w.size() * sizeof(decltype(w)::Scalar));
	in.read(reinterpret_cast<char* const>(b.data()), b.size() * sizeof(decltype(w)::Scalar));
	return in;
}

/* ConvvolutionLayer */
/*
ConvvolutionLayer::ConvvolutionLayer(
	const Layout& _inputDim,
	const Layout& _convolveDim,
	const Neuron& _activation):
	Layer(_inputDim, Layout(nbFeatures, _inputDim.nbLines() - _convolveDim.nbLines() + 1, _inputDim.nbColumns() - _inputDim.nbColumns() + 1), _activation),
	convolveDim(_convolveDim),nbFeatures(convolveDim.nbFeatures()),
	w(convolveDim.nbFeatures(), convolveDim.nbLines(), convolveDim.nbColumns()),
	b(outputDim.dimensions())
{}

void ConvvolutionLayer::forwardPropagation(
	const PropagationBlock& backwardBlock,
	PropagationBlock& currentBlock) const {
	std::array<int, 3> bcast{nbFeatures, 1, 1};
	auto a = backwardBlock.a.broadcast(bcast);
	currentBlock.z = a.convolve(w, std::array<int, 3>{0, 1, 2}) + b;
	activation(currentBlock.z, currentBlock.a);
}

void ConvvolutionLayer::backPropagation(
	PropagationBlock& currentBlock,
	PropagationBlock& backwardBlock) const {
	// ...
}

*/
