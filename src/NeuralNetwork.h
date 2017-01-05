#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <array>
#include <memory>
#include <vector>

#include <unsupported/Eigen/CXX11/Tensor>

typedef Eigen::Tensor<float, 3> LayerGrid;
typedef std::function<void(const LayerGrid&, LayerGrid&)> GridTransformer;
typedef std::function<void(const LayerGrid&, const LayerGrid&, LayerGrid&)> CostDerivative;

class Layout {
public:
	Layout(int, int, int);
	int nbFeatures() const;
	int nbLines() const;
	int nbColumns() const;
	int featureSize() const;
	int size() const;
	std::array<long int, 3> dimensions() const;
private:
	std::array<long int, 3> index;
};

class Neuron {
public:
	Neuron(const GridTransformer&, const GridTransformer&);
	void operator()(const LayerGrid&, LayerGrid&) const;
	void derivative(const LayerGrid&, LayerGrid&) const;
private:
	GridTransformer m_function;
	GridTransformer m_derivative;
};

struct PropagationBlock {
	PropagationBlock() = default;
	// Forwardpropagation
	LayerGrid z; // Neuron input : z_l = w_l * a_{l-1}
	LayerGrid a; // Layer output a_l = f(z_l) with the neuron f
	// Backpropagation
	LayerGrid y; // Neuron reversed input : y_l = w_{l+1} * a_{l+1}
	LayerGrid delta; // Neuron reversed output : delta_l = y_l * f'(z_l)
};

typedef std::vector<PropagationBlock> ForwardPropagation;
typedef std::vector<ForwardPropagation> MiniBatch;

struct Sample {
	Sample() = default;
	Sample(const LayerGrid&, const LayerGrid&);
	LayerGrid input;
	LayerGrid label;
};

class MiniBatchBlock {
public:
	enum class BlockType { Begin, End };
	MiniBatchBlock(const MiniBatch&, BlockType);
	size_t length() const;
	size_t miniBatchSize() const;
	void nextBlock();
	void predBlock();
	bool isFirst() const;
	bool isLast() const;
	void visit(const std::function<void(const PropagationBlock&, const PropagationBlock&)>&) const;
private:
	const MiniBatch& exemples;
	unsigned int blockId;
};

struct ExtraParameters;

class Layer {
public:
	Layer(const Layout&, const Layout&, const Neuron&);
	virtual void forwardPropagation(const PropagationBlock&, PropagationBlock&) const = 0;
	virtual void backPropagation(PropagationBlock&, PropagationBlock&) const = 0;
	virtual void updateParameters(MiniBatchBlock&, const ExtraParameters& params) = 0;
	friend std::ostream& operator<<(std::ostream&, const Layer&);
	friend std::istream& operator>>(std::istream&, Layer&);
protected:
	virtual std::ostream& writeParameters(std::ostream&) const = 0;
	virtual std::istream& readParameters(std::istream&) = 0;
	Layout inputDim;
	Layout outputDim;
	const Neuron& activation;
};

class NeuralNetwork {
public:
	NeuralNetwork() = default;
	NeuralNetwork(const std::vector<Layer*>&);
	NeuralNetwork(const NeuralNetwork&) = delete;
	NeuralNetwork& operator=(const NeuralNetwork&) = delete;
	void predict(const LayerGrid&, LayerGrid&) const;
	void stochasticGradientDescent(const std::vector<Sample>&, size_t, const CostDerivative&, const ExtraParameters&);
	void write(std::string) const;
	void read(std::string);
private:
	void trainMiniBatch(
		std::vector<Sample>::const_iterator,
		std::vector<Sample>::const_iterator,
		const CostDerivative&, const ExtraParameters&);
	void forwardPropagation(const LayerGrid&, ForwardPropagation&) const;
	void backPropagation(ForwardPropagation&) const;
	void updateParameters(const MiniBatch&, const ExtraParameters& params);
	std::vector<std::unique_ptr<Layer>> layers;
};

#endif
