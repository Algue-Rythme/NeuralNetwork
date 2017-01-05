#ifndef NEURAL_GAS_HPP
#define NEURAL_GAS_HPP

#include <cstdint>
#include <forward_list>
#include <memory>
#include <random>
#include <vector>

#include <Eigen/Dense>

class NeuralGas {
public:
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Distance = std::function<double (const Vector&, const Vector&)>;
    class Edge;
    class Neuron;

    struct Edge {
        Edge(Neuron*, Neuron*);
        std::shared_ptr<Neuron> extr1;
        std::shared_ptr<Neuron> extr2;
    };

    class Neuron {
    public:
        Neuron(const Vector&);
        Neuron(const Neuron&) = delete;
        Neuron& operator=(const Neuron&) = delete;
        void addNeighbour(Edge*);
        Vector referent;
        std::forward_list<Edge> edges;
    };

    struct LearningParams {
        LearningParams() = default;
        LearningParams(unsigned int, unsigned int, double, double, double, double);
        unsigned int nodeFrequency;
        unsigned int edgeMaxAge;
        double epsilonB;
        double epsilonN;
        double alpha;
        double beta;
    } params;

    NeuralGas(
        unsigned int,
        const Distance&,
        LearningParams,
        unsigned int);

    void addToDataBase(const std::vector<Vector>&);
    void learnFromDataBase(unsigned int);
    const Neuron& findBestNeuron(const Vector&) const;
    const Neuron& learnFrom(const Vector&);

private:

    void addNeuron(const Vector&);

    // TODO change pointers
    void addEdge(NeuralGas::Neuron*, NeuralGas::Neuron*);
    void deleteEdge(NeuralGas::Edge*);

    // TODO change pointer
    void deleteNeighbours(NeuralGas::Neuron*);
    void performIteration(const Vector&);

    void clearNodes();

    unsigned int dataSize;
    Distance d;
    std::default_random_engine gen;
    std::vector<Vector> datas;
    std::forward_list<std::weak_ptr<Neuron>> neurons;
};

#endif
