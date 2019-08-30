#include "NeuralNetwork.h"

int main(int argc, const char *argv[])
{
	// double** train_features = NULL;
	// double** test_features = NULL;
	// int* train_labels = NULL;
	// int* test_labels = NULL;
	// int test_sample_size = 3428;
	// int train_sample_size = 3772;
	// int col_no = 0;
	// loadData("ann-train.data",train_sample_size,col_no,train_features, train_labels);
	// loadData("ann-test.data",test_sample_size,col_no,test_features,test_labels);
	// cout << train_sample_size<< endl<< col_no;
	// for(int i = 0; i < train_sample_size; i++) {
	// 	for(int j = 0; j < col_no; j++) {
	// 		cout << train_features[i][j]<<" ";
	// 	}
	// 	cout << train_labels[i] <<endl;
	// }


	// // deallocate memory
	// free2D(train_features,train_sample_size);
	// delete[] train_labels;
	// train_labels = NULL;
	// free2D(test_features,test_sample_size);
	// delete[] test_labels;
	// test_labels = NULL;
	double** train_features = NULL;
	int* train_labels = NULL;
	int train_sample_size = 10;
	int feat_no = 3;
	loadData("test.data",train_sample_size,feat_no, train_features, train_labels);
	// for(int i = 0; i < train_sample_size; i++) {
	// 	for(int j = 0; j < feat_no-1; j++) {
	// 		cout << train_features[i][j]<<" ";
	// 	}
	// 	cout << train_labels[i] <<endl;
	// }
	Network* network = NULL;
	int* n_hidden = new int[1];
	double* row = new double[2];
	row[0] = 0;
	row[1] = 1;
	n_hidden[0] = 2;
	initializeNetwork(network,1,2,2,n_hidden);
	// printNetwork(network);
	// printNetwork(network);
	// createTestNetwork(network);
	trainNetwork(network, train_features, 10, 2, train_labels, 20, 0.5, 2);
	double* output;
	// forwardPropagate(network,row,2,output,2);
	// backpropagate(network,row);
	// printNetwork(network);
	cout << endl;

	// for(int i = 0; i < 2;i++)
	//  	cout <<output[i]<<" ";

	for(int i = 0; i < train_sample_size; i++) {
		int prediction = predict(network,train_features[i],2,2);
		cout << "expected: " << train_labels[i] << " got: " << prediction <<endl;
 	}		
	free2D(train_features,train_sample_size);
	delete[] train_labels;
	train_labels = NULL;
	delete network;
	delete[] n_hidden;
	delete[] row;
	// delete output;

}