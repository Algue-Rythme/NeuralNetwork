#ifndef DATA_LOADER_H
#define DATA_LOADER_H

#include <iostream>
#include <cstdint>

#include "SFML/Graphics.hpp"
#include <unsupported/Eigen/CXX11/Tensor>

#include "NeuralNetwork.h"

extern const unsigned int nbDigits;
extern const std::string dataPath;
extern const std::string train;
extern const std::string t10k;

class MNIST_Image {

public:
	typedef Eigen::Tensor<uint8_t, 2> Image_Type;

	MNIST_Image(unsigned int, unsigned int);
	MNIST_Image(unsigned int, const Image_Type&);

	Layout dimensions() const;
	LayerGrid input() const;
	sf::VertexArray getDrawing(unsigned int) const;
	friend std::istream& operator>>(std::istream&, MNIST_Image&);

private:
	unsigned int nbLines;
	unsigned int nbColumns;
	Image_Type data;
};

std::string intToDigit(uint8_t);
uint32_t reverseBits(uint32_t);
uint32_t readInteger(std::istream&);

std::vector<MNIST_Image> loadMNIST_Images(std::string);
std::vector<uint8_t> loadMNIST_Labels(std::string);
std::vector<Sample> makeMNISTSamples(const std::vector<MNIST_Image>&, const std::vector<uint8_t>&);
std::vector<Sample> loadMNISTSamples(std::string);

void checkMNISTSuccess(const NeuralNetwork&, const std::vector<Sample>&);

uint8_t MNIST_Prediction(const NeuralNetwork&, const Sample&);
uint8_t checkOutput(const LayerGrid&);

#endif