#include "NeuralNetwork.h"


/*
 * 
 * A neural network implementation 
 * @author: Gizem Caylak
*/
Neuron::~Neuron() {
	delete weights;
	weights = NULL;
}

Layer::Layer() {
	neuron_no= 0;
	neurons = NULL;
}
Layer::~Layer() {
	delete[] neurons;
	neurons = NULL;
}
Layer::Layer(int neuron_no) {
	this->neuron_no = neuron_no;
	this->neurons = new Neuron[neuron_no];
}
Network::Network(int layer_no) {
	this->layer_no = layer_no;
	layers = new Layer[layer_no];
}

Network::~Network() {
	delete[] layers;
	layers = NULL;
}
void createTestNetwork(Network* &network) {
	network = new Network(2);
	network->layers[0] = *new Layer(2);

	// assign weights randomly
	// cout << network->layers[1].preLayer->neuron_no <<endl;
	int size = 2;
	// cout <<size<<endl;
	network->layers[0].neurons[0].weights = new double[size+1](); // +1 is bias
	network->layers[0].neurons[0].weights[0] = 1.482313569067226;
	network->layers[0].neurons[0].weights[1] = 1.8308790073202204;
	network->layers[0].neurons[0].weights[2] = 1.078381922048799;// bias
	network->layers[0].neurons[1].weights = new double[size+1](); // +1 is bias
	network->layers[0].neurons[1].weights[0] = 0.23244990332399884;
	network->layers[0].neurons[1].weights[1] = 0.3621998343835864;
	network->layers[0].neurons[1].weights[2] = 0.40289821191094327;// bias
	network->layers[0].neurons[0].weights_size = 3;
	network->layers[0].neurons[1].weights_size = 3;
	Layer* output_layer = new Layer(2);
	network->layers[1] = *output_layer;

	for(int j = 0; j < 2; j++) {
		network->layers[1].neurons[j].weights = new double[network->layers[0].neuron_no+1](); // +1 is bias
	}
	network->layers[1].neurons[0].weights_size = 3;
	network->layers[1].neurons[1].weights_size = 3;
	network->layers[1].neurons[0].output = 0.6213859615555266;
	network->layers[1].neurons[1].output = 0.6573693455986976;
	network->layers[1].neurons[0].weights[0] = 2.5001872433501404;
	network->layers[1].neurons[0].weights[1] = 0.7887233511355132;
	network->layers[1].neurons[0].weights[2] = -1.1026649757805829;
	network->layers[1].neurons[1].weights[0] = -2.429350576245497;
	network->layers[1].neurons[1].weights[1] = 0.8357651039198697;
	network->layers[1].neurons[1].weights[2] = 1.0699217181280656;
}
void initializeNetwork(Network* &network, int layer_no, int inputs_no, int outputs_no, int* &hidden_no) {
	
	network = new Network(layer_no+1);
	network->layers[0] = *new Layer(hidden_no[0]);

	for(int j = 0; j < hidden_no[0]; j++) {
		network->layers[0].neurons[j].weights = new double[inputs_no+1]; // +1 is bias
		network->layers[0].neurons[j].weights_size = inputs_no+1;
		for(int k = 0; k  < network->layers[0].neurons[j].weights_size; k++) {
			network->layers[0].neurons[j].weights[k] = ( (double)rand() / (RAND_MAX)) ;
		}
	}
	for(int i = 1; i < layer_no; i++) {
		network->layers[i] = *new Layer(hidden_no[i]);
		// assign weights randomly
		for(int j = 0; j < hidden_no[i]; j++) {
			network->layers[i].neurons[j].weights = new double[network->layers[i-1].neuron_no+1](); // +1 is bias
			for(int k = 0; k  <= network->layers[i-1].neuron_no; k++) {
				network->layers[i].neurons[j].weights[k] = ( (double)rand() / (RAND_MAX));
			}
			network->layers[i].neurons[0].weights_size = network->layers[i-1].neuron_no + 1;
		}
	}

	network->layers[layer_no] = *new Layer(outputs_no);
	for(int j = 0; j < outputs_no; j++) {
		network->layers[layer_no].neurons[j].weights = new double[network->layers[layer_no-1].neuron_no+1](); // +1 is bias
		network->layers[layer_no].neurons[j].weights_size = network->layers[layer_no-1].neuron_no+1;
		for(int k = 0; k  < network->layers[layer_no].neurons[j].weights_size; k++) {
			network->layers[layer_no].neurons[j].weights[k] = ((double)rand() / (RAND_MAX));
			cout <<network->layers[0].neurons[j].weights[k]<< " ";
		}
			cout << endl;
	}
}
void printNetwork(Network* &network) { // print network layer by layer
	cout.precision(15);
	// print other layers & weights
	for(int i = 0; i < network->layer_no; i++) {
		cout << endl << endl << "Layer-" << i ;
		for(int j = 0; j < network->layers[i].neuron_no; j++) {
			cout << endl <<"Neuron-" << j << " weights:" <<endl;
			for(int k = 0; k  < network->layers[i].neurons[j].weights_size; k++) {
				cout << k+1 << ": " << network->layers[i].neurons[j].weights[k] << " ";
			}
			cout << "output:"<< network->layers[i].neurons[j].output << " ";
			cout << "bias:"<< network->layers[i].neurons[j].weights[network->layers[i-1].neuron_no] << " ";
			cout << "delta:"<< network->layers[i].neurons[j].gradient << " ";
		}
	}
	cout << endl;
}

double activate(double* &weights, int weight_size, vector<double> &inputs, int input_size) {
	double activation = weights[weight_size-1]; // bias

	for(int i = 0; i < weight_size-1; i++) {
		activation += weights[i] * inputs[i];
	}
	return activation;
}

double activationFunction(double activation) {
	return 1.0 / (1 + exp(-activation));
}

void forwardPropagate(Network* &network, double* &input, int input_size, double* &output, int output_size) {
	vector<double> next_inputs; // inputs to the next layer
	vector<double> outputs;
	int n_inputs_size = input_size;
	// cout << input[0] <<endl;
	for(int i = 0; i < input_size; i++) {
		next_inputs.push_back(input[i]);
	}
	for(int i = 0; i < network->layer_no; i++) {
		outputs.clear();
		for(int j = 0; j < network->layers[i].neuron_no;j++) {
			double activation = activate(network->layers[i].neurons[j].weights,network->layers[i].neurons[j].weights_size,next_inputs,next_inputs.size());
			network->layers[i].neurons[j].output = activationFunction(activation);
			outputs.push_back(network->layers[i].neurons[j].output);
		}
		next_inputs.clear();
		for(int i = 0; i < outputs.size(); i++) {
			next_inputs.push_back(outputs[i]);
		}
	}
	output = new double[output_size]();
	for(int i = 0; i < next_inputs.size(); i++){
		output[i] = next_inputs[i];
	}
	next_inputs.clear();
	outputs.clear();
}

double calculateActivationFunctionDerivative(double output) {
	return (1.0-output)*output;	// derivative for sigmoid 
}
void backpropagate(Network* &network, double* &expected) {
	double err = 0;
	vector<double> errs;
	for(int i = network->layer_no-1; i >= 0; i--) {
		errs.clear();
		if( i != network->layer_no-1) {
			for(int j = 0; j < network->layers[i].neuron_no; j++) {
				err = 0;
				for(int k = 0; k < network->layers[i+1].neuron_no; k++) {
					err += (network->layers[i+1].neurons[k].gradient *network->layers[i+1].neurons[k].weights[j]) ;
					// cout <<err <<endl;
				}
				errs.push_back(err);
			}
		}
		else { // in output layer
			for(int j = 0; j < network->layers[i].neuron_no; j++) {
				errs.push_back(expected[j] - network->layers[i].neurons[j].output);
			}
		}

		for(int j = 0; j < network->layers[i].neuron_no; j++) {
			network->layers[i].neurons[j].gradient = errs[j] * calculateActivationFunctionDerivative(network->layers[i].neurons[j].output);
		}	
	}
	errs.clear();
}

void updateWeights(Network* &network, double* &input, int input_size, double learning_rate) {
	vector<double> temp;
	for(int i = 0; i < input_size; i++) {
		temp.push_back(input[i]);
	}
	for(int i = 0; i < network->layer_no; i++) {
		if( i != 0) {
			for(int j = 0; j < network->layers[i-1].neuron_no; j++) {// outputs of previous layer input to the next layer
				temp.push_back(network->layers[i-1].neurons[j].output);
			}
		} 
		for(int j = 0; j < network->layers[i].neuron_no; j++) {
			for(int k = 0; k < temp.size(); k++) {
				network->layers[i].neurons[j].weights[k] += learning_rate*temp[k]*network->layers[i].neurons[j].gradient;
			}
			network->layers[i].neurons[j].weights[temp.size()] += learning_rate * network->layers[i].neurons[j].gradient; //add bias
		}
		temp.clear();
	}
	temp.clear();
}
void trainNetwork(Network* &network, double** &train_data, int row, int col, int* &labels, int epochs, double learning_rate, int labels_no) {
	double total_err;
	double *outputs;
	double *expected = new double[labels_no]();

	for(int i = 0; i < epochs; i++){
		total_err = 0;
		for(int j = 0; j < row; j++) {
		// cout << train_data[j][0]<<endl;
			forwardPropagate(network, train_data[j], col, outputs, labels_no);
			// cout << labels_no<<endl;
			for(int k = 0; k < labels_no; k++) {
				expected[k] = 0;
				// cout <<outputs[k]<<" ";
			}
			// cout<< endl;
			expected[labels[j]] = 1;
			// cout << labels[j]<<endl;
			for(int k = 0; k < labels_no; k++) {
				total_err += pow((expected[k] - outputs[k]),2);
			}

			backpropagate(network,expected);
			updateWeights(network,train_data[j], col, learning_rate);

			delete[] outputs;
			outputs = NULL;
		}
		cout << "iteration: "<< i << " learning rate: "<< learning_rate << " error: " << total_err<<endl;
		// printNetwork(network);
	}
	delete[] expected;
	expected = NULL;
}

void normalizeData() {

}

int predict(Network* &network, double* &input, int col, int labels_no) {
	double* outputs;
	forwardPropagate(network, input, col, outputs, labels_no);
	double max = outputs[0];
	int ind_max = 0;
	for(int i = 1; i < labels_no; i++){
		if(outputs[i] >= max) {
			max = outputs[i];
			ind_max = i;
		}
	}
	delete[] outputs;
	outputs = NULL;
	return ind_max;
}
void loadData(string filename, int& row, int& column, double** &features, int* &labels) {
	ifstream file;
	file.open(filename);
	if( row == 0 || column == 0) {
		int row_no = 0;
		if(file.is_open()){
		    while(!file.eof()){
				int col_no = 0;
				string numbers;
				double data;
				getline(file,numbers); //read numbers
				istringstream iss(numbers);
				vector<string> tokens{istream_iterator<string>{iss},
                      istream_iterator<string>{}};
                if(tokens.size()!= 0) {
	                for(int i = 0; i < tokens.size(); i++) {
						data = stod(tokens[i].c_str()); //convert each token to integer
						col_no++;
	                }
	                column = col_no;
	                row_no++;
            	}
		    }

		}
		row = row_no;
	}
	file.close();
	file.open(filename);
	double num = 0;
	features = new double*[row];
    labels = new int[row];

	for(int i = 0; i < row; i++)
    	features[i] = new double[column-1];

	if (file.is_open()) {
		for(int j = 0; j < row; j++){
			for(int i = 0 ; i < column-1;i++) {
		    	file >> num;
		    	features[j][i] = num;
			}
			file >> num;
			labels[j] = num;
		}
	}
	file.close();
	
}
void free2D(double** &arr,int row) {
	for(int j = 0; j < row; j++) {
		delete[] arr[j];
	}	
	delete[] arr;
	arr = NULL;
}

// void freeNetwork(Network* &network) {
// 	for(int i = 0; i < network->layer_no; i++) {
// 		delete[] network->layers[i].neurons;
// 		for(int j = 0; j < network->layers[i]; j++)
// 	}
// }