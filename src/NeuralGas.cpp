#include "NeuralGas.hpp"

#include <algorithm>
#include <exception>
#include <iterator>

NeuralGas::Edge::Edge(NeuralGas::Neuron& _extr1, NeuralGas::Neuron& _extr2):
    extr1(_extr1), extr2(_extr2) {}

NeuralGas::Neuron::Neuron(const NeuralGas::Vector& _referent):
    referent(_referent) {}

void NeuralGas::Neuron::addNeighbour(Edge* edge) {
    edges.push_front(edge);
}

NeuralGas::LearningParams::LearningParams(
    unsigned int _nodeFrequency,
    unsigned int _edgeMaxAge,
    double _epsilonB,
    double _epsilonN,
    double _alpha,
    double _beta
):
nodeFrequency(_nodeFrequency), edgeMaxAge(_edgeMaxAge),
epsilonB(_epsilonB), epsilonN(_epsilonN), alpha(_alpha), beta(_beta) {}

NeuralGas::NeuralGas(
    unsigned int _dataSize,
    const Distance& _d,
    LearningParams _params,
    unsigned int seed):
    dataSize(_dataSize), d(_d), params(_params), gen(seed)
{}

void NeuralGas::addToDataBase(const std::vector<NeuralGas::Vector>& newDatas) {
    std::copy(begin(newDatas), end(newDatas), std::back_inserter(datas));
}

void NeuralGas::learnFromDataBase(unsigned int nbIterations) {
    if (datas.empty())
        throw std::exception();
    for (unsigned int iteration = 0; iteration < nbIterations; ++iteration)
        performIteration(datas[gen() % datas.size()]);
}

const NeuralGas::Neuron& NeuralGas::learnFrom(const NeuralGas::Vector& exemple) {
    datas.push_back(exemple);
    performIteration(exemple);
    return findBestNeuron(exemple);
}

const NeuralGas::Neuron& NeuralGas::findBestNeuron(const NeuralGas::Vector& exemple) const {
    if (neurons.empty())
        throw std::exception();
    // TODO
}

void NeuralGas::addNeuron(const Vector& exemple) {
    // TODO
}

void NeuralGas::performIteration(const NeuralGas::Vector& exemple) {
    // TODO
}

void NeuralGas::addEdge(NeuralGas::Neuron* extr1, NeuralGas::Neuron* extr2) {
    // TODO
}

void NeuralGas::deleteEdges(NeuralGas::Neuron* neuron) {
    // TODO
}
