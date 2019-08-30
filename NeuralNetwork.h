#ifndef NEURAL_NETWORK
  #define NEURAL_NETWORK
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sstream>
#include <iterator>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <utility>
using namespace std;

struct Neuron {
	double output = 0;
	double* weights = NULL;
	int weights_size = 0;
	double gradient = 0;
	~Neuron();
};

struct Layer {
	Neuron* neurons;
	int neuron_no;

	Layer();
	~Layer();
	Layer(int neuron_no);
};

struct Network {
	Layer* layers;
	int layer_no;
	Network(int layer_no);
	~Network();
};

void createTestNetwork(Network* &network); // remove this while submitting
void initializeNetwork(Network* &network, int layer_no, int n_inputs, int n_outputs, int* &n_hidden);
double activate(double* &weights, int weight_size, vector<double> &inputs, int input_size);
double activationFunction();
void forwardPropagate(Network* &network, double* &input, int input_size,  double* &output, int output_size);
double calculateActivationFunctionDerivative();
void backpropagate(Network* &network, double* &expected);
void updateWeights(Network* &network, double* &input, int input_size, double learning_rate);
void trainNetwork(Network* &network, double** &train_data, int row, int col, int* &labels, int epochs, double learning_rate, int labels_no);
int predict(Network* &network, double* &input, int col, int labels_no);
void accuracy();
void normalizeData();
void printNetwork(Network* &network);
void loadData(string filename, int& row, int& column, double** &features, int* &labels);
void free2D(double** &arr,int row);
// void freeNetwork(Network* &network);
#endif