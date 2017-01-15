#include "svm_additional_fnc.h"
#include "svm_classifier.h"

using std::cerr;
using std::endl;

int main(int argc, char *argv[])
{
	SVMClassifier *solver = new SVMClassifier();

	clock_t begin = clock();
	solver->train();
	clock_t train = clock();
	solver->predict();
	clock_t predict = clock();
	std::cout << "Train time:\t" << (double(train - begin) / CLOCKS_PER_SEC) << endl;
	std::cout << "Predict time:\t" << (double(predict - train) / CLOCKS_PER_SEC) << endl;
	std::cout << "Total time:\t" << (double(predict - begin) / CLOCKS_PER_SEC) << endl;
	int wait;
	std::cin >> wait;

}

