#include "NeuralGas.hpp"

#include <chrono>
#include <exception>
#include <iostream>
#include <functional>

#include <SFML/Graphics.hpp>

using std::cout;

int GNG() {
    try {

        const unsigned int width = 1024;
        const unsigned int height = 512;

        unsigned int it = 0;
        const unsigned int maxIt = 50*1000;
        const unsigned int nbDims = 2;

        auto getPos = [=](const NeuralGas::Vector& vect) {
            float x = vect(0, 0)*(width / 2) + (width / 2);
            float y = vect(1, 0)*(height / 2) + (height / 2);
            return sf::Vector2f(x, y);
        };

        NeuralGas::LearningParams params(300, 80, 0.05, 0.0006, 0.05, 0.0005);
        NeuralGas gas(nbDims, params, std::chrono::system_clock::now().time_since_epoch().count());

        sf::VertexArray exemples(sf::PrimitiveType::Points);
        sf::RenderWindow window(sf::VideoMode(width, height), "Neural Gas");
        while (window.isOpen()) {

            if (it < maxIt) {
                NeuralGas::Vector exemple = NeuralGas::Vector::Random(nbDims, 1);
                if (exemple(0, 0) * exemple(1, 0) < 0)
                    exemple(0, 0) *= -1.;
                gas.learnFrom(exemple);
                exemples.append(sf::Vertex(getPos(exemple), sf::Color::Red));
                ++it;
            }

            std::vector<NeuralGas::EdgePtr> edges = gas.getEdges();
            sf::VertexArray lines(sf::PrimitiveType::Lines);
            for (const auto& edge : edges) {
                lines.append(sf::Vertex(getPos(edge->extr1->referent), sf::Color::White));
                lines.append(sf::Vertex(getPos(edge->extr2->referent), sf::Color::White));
            }

            sf::Event event;
            while (window.pollEvent(event))
            {
                if (event.type == sf::Event::Closed)
                    window.close();
            }

            window.clear(sf::Color::Black);
            window.draw(exemples);
            window.draw(lines);
            window.display();
        }
    } catch (const std::exception& e) {
        cout << e.what() << "\n";
        return EXIT_FAILURE;;
    }

    return EXIT_SUCCESS;
}
