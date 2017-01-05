#include "DataLoader.h"

#include <fstream>

using namespace std;
using namespace sf;

const unsigned int nbDigits = 10;
const string dataPath = "./datas/";
const string train = "train";
const string t10k = "t10k";

MNIST_Image::MNIST_Image(unsigned int _nbLines, unsigned int _nbColumns):
	nbLines(_nbLines), nbColumns(_nbColumns), data(nbLines, nbColumns)
{}

MNIST_Image::MNIST_Image(unsigned int _nbLines, const Image_Type& _data):
	nbLines(_nbLines), nbColumns(_data.size()/nbLines), data(_data)
{}

Layout MNIST_Image::dimensions() const {
	return Layout(1, nbLines, nbColumns);
}

LayerGrid MNIST_Image::input() const {
	LayerGrid grid(1, int(nbLines), int(nbColumns));
	for (int line = 0; line < int(nbLines); ++line)
		for (int column = 0; column < int(nbColumns); ++column)
			grid(0, line, column) = static_cast<float>(data(line, column) / 255.f);
	return grid;
}

VertexArray MNIST_Image::getDrawing(unsigned int pixelSize) const {
	static const Vector2u dir[4] = {Vector2u(0, 0), Vector2u(1, 0), Vector2u(1, 1), Vector2u(0, 1)};

	VertexArray vertices(PrimitiveType::Quads, data.size()*4);
	for (unsigned int line = 0; line < nbLines; ++line) {
		for (unsigned int column = 0; column < nbColumns; ++column) {
			uint8_t color = (~uint8_t(0)) - data(line, column);
			unsigned int id = line*nbColumns + column;
			Vertex* quad = &vertices[id*4];

			for (unsigned int i = 0; i < 4; ++i) {
				quad[i].color = Color(color, color, color);
				quad[i].position = Vector2f(float((column+dir[i].x)*pixelSize),
					                        float((line+dir[i].y)*pixelSize));
			}
		}
	}

	return vertices;
}

istream& operator>>(istream& in, MNIST_Image& img) {
	vector<uint8_t> buffer(img.nbLines*img.nbColumns);
	in.read(reinterpret_cast<char*>(&buffer[0]), sizeof(uint8_t)*buffer.size());
	for (unsigned int line = 0; line < img.nbLines; ++line)
		for (unsigned int column = 0; column < img.nbColumns; ++column)
			img.data(line, column) = buffer[line*img.nbColumns + column];
	return in;
}

vector<MNIST_Image> loadMNIST_Images(string filename) {
	cout << ("Load images from "+filename+"... ");
	ifstream in(filename, ios::binary);
	uint32_t magicNumber = readInteger(in);
	uint32_t nbImages = readInteger(in);
	uint32_t nbLines = readInteger(in);
	uint32_t nbColumns = readInteger(in);
	vector<MNIST_Image> images(nbImages, MNIST_Image(nbLines, nbColumns));
	for (auto& img : images)
		in >> img;
	cout << "done !\n>>\t" << nbImages << " images " << nbLines << "x" << nbColumns << "\n\n";
	return images;
}

vector<uint8_t> loadMNIST_Labels(string filename) {
	cout << ("Load labels from "+filename+"... ");
	ifstream in(filename, ios::binary);
	uint32_t magicNumber = readInteger(in);
	uint32_t nbLabels = readInteger(in);
	vector<uint8_t> labels(nbLabels);
	in.read(reinterpret_cast<char*>(&labels[0]), sizeof(uint8_t)*labels.size());
	cout << "done !\n>>\t" << nbLabels << " labels\n\n";
	return labels;
}

uint32_t readInteger(istream& in) {
	uint32_t data;
	in.read(reinterpret_cast<char*>(&data), sizeof(data));
	return reverseBits(data);
}

uint32_t reverseBits(uint32_t n) {
	const unsigned int byteSize = 8;
	const uint8_t allOn = ~uint8_t(0);
	uint32_t reversed = 0;
	for (unsigned int i = 0; i < sizeof(uint32_t); ++i) {
		uint8_t block = allOn & (n >> (byteSize*i));
		reversed += (block << (byteSize*(sizeof(uint32_t) - i - 1)));
	}
	return reversed;
}

string intToDigit(uint8_t number) {
	char digit = char(number) + '0';
	return string(1, digit);
}

vector<Sample> makeMNISTSamples(const vector<MNIST_Image>& images, const vector<uint8_t>& labels) {
	vector<Sample> samples; samples.reserve(images.size());
	for (unsigned int sample = 0; sample < images.size(); ++sample) {
		LayerGrid expected(nbDigits, 1, 1);
		expected.setZero();
		expected(labels[sample], 0, 0) = 1.f;
		samples.emplace_back(images[sample].input(), expected);
	}
	return samples;
}

uint8_t checkOutput(const LayerGrid& grid) {
	unsigned int bestProb = 0;
	for (unsigned int digit = 0; digit < nbDigits; ++digit)
		if (grid(digit, 0, 0) > grid(bestProb, 0, 0))
			bestProb = digit;
	return static_cast<uint8_t>(bestProb);
}

vector<Sample> loadMNISTSamples(string fileName) {
	vector<MNIST_Image> data = loadMNIST_Images(dataPath + fileName + "-images.idx3-ubyte");
	vector<uint8_t> labels = loadMNIST_Labels(dataPath + fileName + "-labels.idx1-ubyte");
	return makeMNISTSamples(data, labels);
}

uint8_t MNIST_Prediction(const NeuralNetwork& nn, const Sample& sample) {
	LayerGrid output;
	nn.predict(sample.input, output);
	return checkOutput(output);
}

void checkMNISTSuccess(const NeuralNetwork& nn, const vector<Sample>& samples) {
	cout << "Evaluation : ";
	unsigned int success = 0;
	for (const auto& sample : samples)
		if (MNIST_Prediction(nn, sample) == checkOutput(sample.label))
			success += 1;
	cout << float(success) / float(samples.size())*100.f << "% accuracy !\n";
}
