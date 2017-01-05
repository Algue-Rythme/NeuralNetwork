#ifndef LAYERS_H
#define LAYERS_H

#include <array>
#include <memory>
#include <vector>

#include "NeuralNetwork.h"

extern const std::array<Eigen::IndexPair<int>, 0> TensorProductWithoutContraction;

class FullyConnectedLayer : public Layer {
public:
	FullyConnectedLayer(const Layout&, int, const Neuron&);

	virtual void forwardPropagation(const PropagationBlock&, PropagationBlock&) const override;
	virtual void backPropagation(PropagationBlock&, PropagationBlock&) const override;
	virtual void updateParameters(MiniBatchBlock&, const ExtraParameters& params) override;

protected:
	std::ostream& writeParameters(std::ostream&) const override;
	std::istream& readParameters(std::istream&) override;

private:
	Eigen::Tensor<float, 1> compute_pd_b(const MiniBatchBlock&) const;
	Eigen::Tensor<float, 4> compute_pd_w(const MiniBatchBlock&) const;

	static const std::array<Eigen::IndexPair<int>, 3> forwardContraction;
	static const std::array<Eigen::IndexPair<int>, 1> backwardContraction;

	int nbNeurons;
	std::array<int, 1> column;
	Eigen::Tensor<float, 4> w;
	Eigen::Tensor<float, 1> b;
	Eigen::Tensor<float, 4> w_velocity;
	Eigen::Tensor<float, 1> b_velocity;
};
/*
class ConvolutionLayer : public Layer {
public:
	ConvvolutionLayer(const Layout&, const Layout&, const Neuron&);
	virtual void forwardPropagation(const PropagationBlock&, PropagationBlock&) const override;
	virtual void backPropagation(PropagationBlock&, PropagationBlock&) const override;
	virtual void updateParameters(MiniBatchBlock&, const ExtraParameters& params) override;
protected:
	virtual std::ostream& writeParameters(std::ostream&) const override;
	virtual std::istream& readParameters(std::istream&) override;
private:
	Layout convolveDim;
	unsigned int nbFeatures;
	Eigen::Tensor<float, 3> w;
	Eigen::Tensor<float, 3> b;
};*/

#endif
