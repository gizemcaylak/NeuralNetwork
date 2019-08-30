ann: main.o NeuralNetwork.o
	g++ -o $@ $^
	rm *.o
	./ann
