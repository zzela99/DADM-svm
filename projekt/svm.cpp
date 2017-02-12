#include "svm_additional_fnc.h"
#include "svm_classifier.h"

using std::cerr;
using std::endl;
<<<<<<< HEAD

int main(int argc, char *argv[])
{
	SVMClassifier *solver = new SVMClassifier();

	clock_t begin = clock();
	solver->train();
	clock_t train = clock();
	solver->predict();
	clock_t predict = clock();
=======
// g³ówna funkcja main
int main(int argc, char *argv[])
{
	SVMClassifier *our_classifier = new SVMClassifier(); 	// tworzenie nowego obiektu klasy SVMClassifier 

	clock_t begin = clock(); 								// zapisanie czasu rozpoczêcia treningu w zmiennej begin typy clock_t
	our_classifier->train();								// wywo³anie metody train na obiekcie
	clock_t train = clock();								//zapisanie czasu koñca treningu w zmiennej train typu clock_t
	our_classifier->predict();								// wywo³anie metody predict na obiekcie
	clock_t predict = clock();								//zapisanie czasu koñca predykcji do zmiennej predict typy clock_t
>>>>>>> Update
	std::cout << "Train time:\t" << (double(train - begin) / CLOCKS_PER_SEC) << endl;
	std::cout << "Predict time:\t" << (double(predict - train) / CLOCKS_PER_SEC) << endl;
	std::cout << "Total time:\t" << (double(predict - begin) / CLOCKS_PER_SEC) << endl;
	int wait;
<<<<<<< HEAD
	std::cin >> wait;
=======
	std::cin >> wait;										// zamkniêcie przez wpisanie liczby
>>>>>>> Update

}

