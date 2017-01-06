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
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    class Edge;
    class Neuron;
    using EdgePtr = std::shared_ptr<Edge>;
    using NeuronPtr = std::shared_ptr<Neuron>;

    class Neuron {
    public:
        Neuron(const Vector&);
        Neuron(const Neuron&) = delete;
        Neuron& operator=(const Neuron&) = delete;
        void addEdge(const EdgePtr&);
        void removeEdge(const EdgePtr&);
        void removeOldestEdges(unsigned int);
        void increaseEdgesAge();
        bool updateEdgeToward(const NeuronPtr&);
        void moveNeighbours(const Vector&);
        const NeuronPtr& neighbour(const EdgePtr&) const;
        Vector referent;
        double error;
        std::forward_list<EdgePtr> edges;
    };

    struct Edge {
        Edge(const NeuronPtr&, const NeuronPtr&);
        const NeuronPtr& neighbourOf(const NeuronPtr&);
        NeuronPtr extr1;
        NeuronPtr extr2;
        unsigned int age;
    };

    struct LearningParams {
        LearningParams() = default;
        LearningParams(unsigned int, unsigned int, double, double, double, double);
        unsigned int nodeFrequency;
        unsigned int maxAge;
        double epsilonB;
        double epsilonN;
        double alpha;
        double beta;
    } params;

    NeuralGas(
        unsigned int,
        LearningParams,
        unsigned int);

    NeuralGas(const NeuralGas&) = delete;
    NeuralGas& operator=(const NeuralGas&) = delete;

    void addToDataBase(const std::vector<Vector>&);
    void learnFromDataBase(unsigned int);
    const Vector& learnFrom(const Vector&);

    ~NeuralGas();

private:

    void performIteration(const Vector&);
    std::pair<NeuronPtr, NeuronPtr> findBestNeurons(const Vector&) const;
    void removeDeadNeurons();
    const NeuronPtr& addNeuron(const NeuronPtr&);
    void addEdge(const NeuronPtr&, const NeuronPtr&);
    void removeEdge(EdgePtr&);
    void createNeuron();

    unsigned int dataSize;
    std::default_random_engine gen;
    unsigned int nbIterations;
    std::vector<Vector> datas;
    std::forward_list<std::weak_ptr<Neuron>> neurons;
};

#endif
