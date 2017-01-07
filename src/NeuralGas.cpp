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

const NeuronPtr& NeuralGas::Edge::neighbourOf(const NeuronPtr& origin) {
    if (extr1 == origin)
        return extr2;
    if (extr2 == origin)
        return extr1;
    throw std::runtime_error("Request neighbour on bad edge.");
}

NeuralGas::Neuron::Neuron(const NeuralGas::Vector& _referent, double _error):
    referent(_referent), error(_error) {}

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

bool NeuralGas::Neuron::updateEdgeToward(const NeuronPtr& target) {
    for (const auto& edge : edges) {
        if (edge->extr1 == target || edge->extr2 == target) {
            edge->age = 0;
            return true;
        }
    }
    return false;
}

void NeuralGas::Neuron::moveNeighbours(const Vector& direction) {
    for (const auto& edge : edges) {
        const auto& extr = neighbour(edge);
        extr->referent += direction;
    }
}

const NeuronPtr& NeuralGas::Neuron::neighbour(const EdgePtr& edge) const {
    if (edge->extr1.get() == this)
        return edge->extr2;
    if (edge->extr2.get() == this)
        return edge->extr1;
    throw std::runtime_error("Neuron have a non-adjacent edge.");
}

EdgePtr NeuralGas::Neuron::findMaxErrorNeighbour() const {
    return *std::max_element(begin(edges), end(edges),
            [&](const EdgePtr& leftEdge, const EdgePtr& rightEdge) {
                auto left = neighbour(leftEdge);
                auto right = neighbour(rightEdge);
                return left->error < right->error;;
    });
}

void NeuralGas::Neuron::clear() {
    edges.clear();
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
    LearningParams _params,
    unsigned int seed):
    dataSize(_dataSize), params(_params), gen(seed), nbIterations(0)
{
    Vector v1 = NeuralGas::Vector::Random(dataSize, 1);
    Vector v2 = NeuralGas::Vector::Random(dataSize, 1);
    auto neuron1 = addNeuron(make_shared<Neuron>(v1, 0.));
    auto neuron2 = addNeuron(make_shared<Neuron>(v2, 0.));
    addEdge(neuron1, neuron2);
}

NeuralGas::~NeuralGas() {
    for (const auto& maybeNeuron : neurons) {
        if (auto neuron = maybeNeuron.lock()) {
            neuron->clear();
        }
    }
}

std::vector<EdgePtr> NeuralGas::getEdges() const {
    std::vector<EdgePtr> edges;
    for (const auto& maybeNeuron : neurons) {
        if (auto neuron = maybeNeuron.lock()) {
            std::copy_if(begin(neuron->edges), end(neuron->edges),
            std::back_inserter(edges),
            [&](const EdgePtr& edge) {
                return edge->extr1 == neuron;
            });
        }
    }
    return edges;
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
    return findClosestRepresentant(exemple);
}

void NeuralGas::performIteration(const NeuralGas::Vector& exemple) {
    nbIterations += 1;

    auto bestNeurons = findBestNeurons(exemple);
    auto& v1 = bestNeurons.first;
    auto& v2 = bestNeurons.second;

    v1->increaseEdgesAge();

    Vector diff = exemple - v1->referent;
    v1->error += diff.squaredNorm();

    v1->referent += diff*params.epsilonB;
    v1->moveNeighbours(diff*params.epsilonN);

    if (!v1->updateEdgeToward(v2))
        addEdge(v1, v2);

    v1->removeOldestEdges(params.maxAge);

    if (nbIterations % params.nodeFrequency == 0) {
        removeDeadNeurons();
        createNeuron();
    }
}

const NeuralGas::Vector& NeuralGas::findClosestRepresentant(const Vector& exemple) {
    auto bestNeurons = findBestNeurons(exemple);
    return bestNeurons.first->referent;
}

std::pair<NeuronPtr, NeuronPtr>
NeuralGas::findBestNeurons(const NeuralGas::Vector& exemple) const {
    NeuronPtr first = nullptr; double firstD = HUGE_VAL;
    NeuronPtr second = nullptr; double secondD = HUGE_VAL;
    for (const auto& maybeNeuron : neurons) {
        if (auto current = maybeNeuron.lock()) {
            double currentD = (current->referent - exemple).squaredNorm();
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

void NeuralGas::createNeuron() {
    auto q = findMaxErrorNeuron();
    auto edgeF = q->findMaxErrorNeighbour();
    auto f = edgeF->neighbourOf(q);
    q->error *= params.alpha;
    f->error *= params.alpha;
    auto r = addNeuron(make_shared<Neuron>(0.5*(q->referent + f->referent), q->error));
    addEdge(q, r);
    addEdge(r, f);
    removeEdge(edgeF);
    for (const auto& maybeNeuron : neurons) {
        if (auto current = maybeNeuron.lock()) {
            if (current == q || current == f)
                continue ;
            current->error *= params.beta;
        }
    }
}

NeuronPtr NeuralGas::findMaxErrorNeuron() const {
    auto q = std::max_element(begin(neurons), end(neurons),
        []( const std::weak_ptr<Neuron>& weak_left,
            const std::weak_ptr<Neuron>& weak_right) {
                auto left = weak_left.lock();
                auto right = weak_right.lock();
                if (left && right)
                    return left->error < right->error;
                return (left && !right);
        })->lock();
    if (!q)
        throw std::runtime_error("Empty neuron found.");
    return q;
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
