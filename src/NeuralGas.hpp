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
    using Distance = std::function<double (const Vector&, const Vector&)>;

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
        Vector referent;
        std::forward_list<EdgePtr> edges;
    };

    struct Edge {
        Edge(const NeuronPtr&, const NeuronPtr&);
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
        const Distance&,
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

    unsigned int dataSize;
    Distance d;
    std::default_random_engine gen;
    std::vector<Vector> datas;
    std::forward_list<std::weak_ptr<Neuron>> neurons;
};

#endif
