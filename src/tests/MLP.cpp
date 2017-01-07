#include <iostream>
#include <cstdlib>

#include <SFML/Graphics.hpp>

#include "DataLoader.h"
#include "Layers.h"
#include "ExtraParameters.h"

using namespace std;
using namespace sf;

const string parametersFileName = "parameters.dat";
const string parametersFile = dataPath + parametersFileName;

void learnMNIST(NeuralNetwork& nn, const vector<Sample>& originalSet, const vector<Sample>& testSet, unsigned int nbEpochs) {
	cout << "Learning began... \n";
	vector<Sample> samples = originalSet;
	for (unsigned int epoch = 1; epoch <= nbEpochs; ++epoch) {
		cout << "Epoch " << epoch;
		Clock clock;
		nn.stochasticGradientDescent(samples, 10, CostFunction::quadraticCost, ExtraParameters(3.f, 1.f));
		cout << " finished after " << clock.getElapsedTime().asSeconds() << "s\n\t";
		checkMNISTSuccess(nn, testSet);
		random_shuffle(begin(samples), end(samples));
	}
}

int MLP()
{
	const unsigned int pixelSize = 10;
	const string testingSet = t10k;
	vector<MNIST_Image> data = loadMNIST_Images(dataPath + testingSet + "-images.idx3-ubyte");
	vector<uint8_t> labels = loadMNIST_Labels(dataPath + testingSet + "-labels.idx1-ubyte");

	NeuralNetwork nn(vector<Layer*>{
		static_cast<Layer*>(new FullyConnectedLayer(data.back().dimensions(), 30, ActivationFunction::Sigmoid)),
		static_cast<Layer*>(new FullyConnectedLayer(Layout(30, 1, 1), 10, ActivationFunction::Sigmoid))
	});

	vector<Sample> tests = makeMNISTSamples(data, labels);
	// const unsigned int nbEpochs = 1;
	// learnMNIST(nn, loadMNISTSamples(train), tests, nbEpochs);
	// nn.write(parametersFile);
	nn.read(parametersFile);
	checkMNISTSuccess(nn, tests);

	unsigned int id = 0;
	VertexArray img = data[id].getDrawing(pixelSize);
	Font font;
	font.loadFromFile("./fonts/arial.ttf");

	Text label(intToDigit(labels[id]), font, 60);
	label.setColor(Color::Red);
	label.setPosition(Vector2f(20, 0));
	Text predicted(intToDigit(labels[id]), font, 60);
	predicted.setColor(Color::Green);
	predicted.setPosition(Vector2f(20, 200));

	predicted.setString(intToDigit(MNIST_Prediction(nn, tests[id])));

    RenderWindow window(VideoMode(280, 280), "Convolutional Neural Network");

    while (window.isOpen())
    {
        Event event;
        while (window.pollEvent(event))
        {
            if (event.type == Event::Closed)
                window.close();
			if (event.type == Event::KeyPressed) {
				if (event.key.code == Keyboard::Key::Right)
					id = (id+1 == data.size())? 0 : id+1;
				if (event.key.code == Keyboard::Key::Left)
					id = (id == 0)? data.size()-1 : id-1;
				img = data[id].getDrawing(pixelSize);
				label.setString(intToDigit(labels[id]));
				predicted.setString(intToDigit(MNIST_Prediction(nn, tests[id])));

				LayerGrid output;
				nn.predict(tests[id].input, output);
				for (unsigned int digit = 0; digit < nbDigits; ++digit)
					cout << "\t" << digit << ": " << output(digit, 0, 0) << "\n";
				cout << "\n";
			}
        }

        window.clear();
        window.draw(img);
		window.draw(label);
		window.draw(predicted);
        window.display();
    }

    return EXIT_SUCCESS;
}
