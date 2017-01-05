#include "NeuralGas.hpp"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <cmath>

using NeuronPtr = NeuralGas::NeuronPtr;
using EdgePtr = NeuralGas::EdgePtr;
using std::make_shared;

NeuralGas::Edge::Edge(
    const NeuronPtr& _extr1,
    const NeuronPtr& _extr2):
    extr1(_extr1), extr2(_extr2), age(0) {}

NeuralGas::Neuron::Neuron(const NeuralGas::Vector& _referent):
    referent(_referent) {}

void NeuralGas::Neuron::addEdge(const EdgePtr& edge) {
    edges.emplace_front(edge);
}

void NeuralGas::Neuron::removeEdge(const EdgePtr& edge) {
    edges.remove(edge);
}

void NeuralGas::Neuron::removeOldestEdges(unsigned int maxAge) {
    edges.remove_if([maxAge](const EdgePtr& edge){
        return edge->age >= maxAge;
    });
}

void NeuralGas::Neuron::increaseEdgesAge() {
    for (const auto& edge : edges) {
        edge->age += 1;
    }
}

NeuralGas::LearningParams::LearningParams(
    unsigned int _nodeFrequency,
    unsigned int _maxAge,
    double _epsilonB,
    double _epsilonN,
    double _alpha,
    double _beta
):
nodeFrequency(_nodeFrequency), maxAge(_maxAge),
epsilonB(_epsilonB), epsilonN(_epsilonN), alpha(_alpha), beta(_beta) {}

NeuralGas::NeuralGas(
    unsigned int _dataSize,
    const Distance& _d,
    LearningParams _params,
    unsigned int seed):
    dataSize(_dataSize), d(_d), params(_params), gen(seed)
{
    Vector v1(1, dataSize); v1.setZero();
    Vector v2(1, dataSize); v2.setConstant(1.);
    auto neuron1 = addNeuron(make_shared<Neuron>(v1));
    auto neuron2 = addNeuron(make_shared<Neuron>(v2));
    addEdge(neuron1, neuron2);
}

NeuralGas::~NeuralGas() {
    for (const auto& maybeNeuron : neurons) {
        if (auto neuron = maybeNeuron.lock()) {
            neuron->edges.clear();
        }
    }
}

void NeuralGas::addToDataBase(const std::vector<NeuralGas::Vector>& newDatas) {
    datas.reserve(newDatas.size() + datas.size());
    std::copy(begin(newDatas), end(newDatas), std::back_inserter(datas));
}

void NeuralGas::learnFromDataBase(unsigned int nbIterations) {
    if (datas.empty())
        throw std::exception();
    for (unsigned int iteration = 0; iteration < nbIterations; ++iteration) {
        performIteration(datas[gen() % datas.size()]);
    }
}

const NeuralGas::Vector& NeuralGas::learnFrom(const NeuralGas::Vector& exemple) {
    datas.push_back(exemple);
    performIteration(exemple);
    return exemple; // TODO change that
}

void NeuralGas::performIteration(const NeuralGas::Vector& exemple) {
    auto bestNeurons = findBestNeurons(exemple);
    auto& v1 = bestNeurons.first;
    auto& v2 = bestNeurons.second;

    v1->increaseEdgesAge();
    // v1->updateError(d(v1->referent, exemple));
    // move v1 and it's neighbours toward
    // v1->updateEdgeToward(v2);
    v1->removeOldestEdges(params.maxAge);
    // removeDeadNeurons(); TODO change that

    // add new neuron
}

std::pair<NeuronPtr, NeuronPtr>
NeuralGas::findBestNeurons(const NeuralGas::Vector& exemple) const {
    NeuronPtr first = nullptr; double firstD = HUGE_VAL;
    NeuronPtr second = nullptr; double secondD = HUGE_VAL;
    for (const auto& maybeNeuron : neurons) {
        if (auto current = maybeNeuron.lock()) {
            double currentD = d(current->referent, exemple);
            if (!first || currentD < firstD) {
                second = first;
                secondD = firstD;

                first = current;
                firstD = currentD;
            } else if (!second || currentD < secondD) {
                second = current;
                secondD = currentD;
            }
        }
    }
    if (!first || !second)
        throw std::runtime_error("There is less than two neurons alive. This should never happened.");
    return std::make_pair(first, second);
}

void NeuralGas::removeDeadNeurons() {
    neurons.remove_if([](const std::weak_ptr<Neuron>& neuron) {
        return neuron.expired();
    });
}

const NeuronPtr& NeuralGas::addNeuron(const NeuronPtr& neuron) {
    neurons.emplace_front(neuron);
    return neuron;
}

void NeuralGas::addEdge(
    const NeuronPtr& extr1,
    const NeuronPtr& extr2) {
    auto edge = make_shared<Edge>(extr1, extr2);
    extr1->addEdge(edge);
    extr2->addEdge(edge);
}

void NeuralGas::removeEdge(EdgePtr& edge) {
    auto extr1 = edge->extr1;
    auto extr2 = edge->extr2;
    extr1->removeEdge(edge);
    extr2->removeEdge(edge);
    edge.reset();
}
