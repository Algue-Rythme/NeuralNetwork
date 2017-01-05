#include <fstream>
#include <iostream>

#include "NeuralNetwork.h"
#include "ExtraParameters.h"

using namespace std;

Layout::Layout(
	int _nbFeatures,
	int _nbLines,
	int _nbColumns)
{
	index[0] = _nbFeatures;
	index[1] = _nbLines;
	index[2] = _nbColumns;
}

int Layout::nbFeatures() const {return index[0];}
int Layout::nbLines() const {return index[1];}
int Layout::nbColumns() const {return index[2];}
int Layout::featureSize() const {return nbLines()*nbColumns();}
int Layout::size() const {return nbFeatures()*featureSize();}
array<long int, 3> Layout::dimensions() const {return index;}

Neuron::Neuron(
	const GridTransformer& _m_function,
	const GridTransformer& _m_derivative) :
	m_function(_m_function),
	m_derivative(_m_derivative)
{}

void Neuron::operator()(const LayerGrid& input, LayerGrid& output) const {m_function(input, output);}
void Neuron::derivative(const LayerGrid& input, LayerGrid& output) const {m_derivative(input, output);}

Layer::Layer(
	const Layout& _inputDim,
	const Layout& _outputDim,
	const Neuron& _activation):
	inputDim(_inputDim), outputDim(_outputDim), activation(_activation) {}

Sample::Sample(const LayerGrid& _input, const LayerGrid& _label): input(_input), label(_label)
{}

MiniBatchBlock::MiniBatchBlock(const MiniBatch& _exemples, BlockType type):
	exemples(_exemples), blockId(type == MiniBatchBlock::BlockType::Begin ? 1 : length()-1)
{}

size_t MiniBatchBlock::length() const { return exemples[0].size(); }
size_t MiniBatchBlock::miniBatchSize() const { return exemples.size(); }
void MiniBatchBlock::nextBlock() { if(!isLast()) ++blockId;}
void MiniBatchBlock::predBlock() { if(!isFirst()) --blockId;}
bool MiniBatchBlock::isFirst() const { return blockId == 1; }
bool MiniBatchBlock::isLast() const { return blockId == length()-1; }

void MiniBatchBlock::visit(const std::function<void(const PropagationBlock&, const PropagationBlock&)>& op) const {
	for (const auto& exemple : exemples)
		op(exemple[blockId - 1], exemple[blockId]);
}

ostream& operator<<(ostream& out, const Layer& layer) {
	return layer.writeParameters(out);
}

istream& operator>>(istream& in, Layer& layer) {
	return layer.readParameters(in);
}

NeuralNetwork::NeuralNetwork(const vector<Layer*>& _layers) {
	layers.reserve(_layers.size());
	for (const auto& layer : _layers)
		layers.emplace_back(layer);
}

void NeuralNetwork::forwardPropagation(const LayerGrid& input, ForwardPropagation& blocks) const {
	blocks.resize(layers.size() + 1);
	auto block = blocks.begin();
	block->a = input;
	for (const auto& layer : layers) {
		layer->forwardPropagation(*block, *next(block));
		++block;
	}
}

void NeuralNetwork::predict(const LayerGrid& input, LayerGrid& output) const {
	ForwardPropagation blocks;
	forwardPropagation(input, blocks);
	output = blocks.back().a;
}

void NeuralNetwork::backPropagation(ForwardPropagation& blocks) const {
	auto block = rbegin(blocks);
	for_each (layers.crbegin(),layers.crend(),
		[&](const std::unique_ptr<Layer>& layer) {
			layer->backPropagation(*block, *next(block));
			++block;
	});
}

void NeuralNetwork::updateParameters(const MiniBatch& miniBatch, const ExtraParameters& params) {
	MiniBatchBlock blocks(miniBatch, MiniBatchBlock::BlockType::Begin);
	for (auto& layer : layers) {
		layer->updateParameters(blocks, params);
		blocks.nextBlock();
	}
}

void NeuralNetwork::trainMiniBatch(
	vector<Sample>::const_iterator begin,
	vector<Sample>::const_iterator end,
	const CostDerivative& cost,
	const ExtraParameters& params) {
	MiniBatch miniBatch; miniBatch.reserve(end - begin);
	for_each (begin, end, [&](const Sample& sample) {
		miniBatch.emplace_back();
		ForwardPropagation& feed = miniBatch.back();
		forwardPropagation(sample.input, feed);

		PropagationBlock& outputBlock = feed.back();
		cost(outputBlock.a, sample.label, outputBlock.y);
		backPropagation(feed);
	});
	updateParameters(miniBatch, params);
}

void NeuralNetwork::stochasticGradientDescent(
	const std::vector<Sample>& samples,
	size_t miniBatchSize,
	const CostDerivative& cost,
	const ExtraParameters& params) {
	auto miniBatch = begin(samples);
	while (miniBatch != end(samples)) {
		size_t step = min(miniBatchSize, size_t(end(samples) - miniBatch));
		trainMiniBatch(miniBatch, miniBatch+step, cost, params);
		advance(miniBatch, step);
	}
}

void NeuralNetwork::write(string filename) const {
	ofstream out(filename, ios::binary);
	for (const auto& layer : layers)
		out << *layer;
}

void NeuralNetwork::read(string filename) {
	ifstream in(filename, ios::binary);
	for (auto& layer : layers)
		in >> *layer;
}
