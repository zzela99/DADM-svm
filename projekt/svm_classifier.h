<<<<<<< HEAD
=======
//za³¹czenie wszystkich potrzebnych bibliotek
>>>>>>> Update
#include <iostream>
#include <fstream>
#include <strstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <cmath>
#include <cstdlib>
#include <string.h>
#include <iomanip>
<<<<<<< HEAD

#include "svm_additional_fnc.h"



class SVMClassifier{
public:
    SVMClassifier();
    ~SVMClassifier();
    int train();
    int predict();
protected:

    float errorRate();

    int loadResults(std::ifstream& is);
    void writeResultModel(std::ofstream& os);
    
    float kernel(int i1, int i2); 
    float learnedFnc(int k);
=======
// za³¹czenie pliku nag³ówkowego z funkcjami dodatkowymi
#include "svm_additional_fnc.h"


// deklaracja klasy SVMClassifier
class SVMClassifier{
public:					//sk³adniki publiczne klasy
    SVMClassifier(); 	//konstruktor
    ~SVMClassifier();	//destruktor	
    int train();		// metoda treningowa
    int predict();		// metoda predykcyjna
protected:				// sk³adniki chronione klasy

    float errorRate();	// wartoœæ b³êdu dla okreœlonej próbki testowej

    int loadResults(std::ifstream& is);			//wczytywanie próbek z pliku
    void writeResultModel(std::ofstream& os);	//
    
    float kernel(int i1, int i2); 				//metoda j¹dra
    float learnedFnc(int k);					
>>>>>>> Update

    int examineExample(int i1);
    int takeStep(int i1, int i2);

    double C;
    double epsilon;
<<<<<<< HEAD
    char fNameTrain[256];
    char fNameTest[256];
    char fNameResults[256];
    
    int N; 
    int NTestSamples;
    TVectorArray arrayX;
    TFloatArray arrayY;

    TFloatArray alpha;
    TFloatArray d;
    TFloatArray arrayError;
=======
    char fNameTrain[256];	// nazwa pliku treningowego
    char fNameTest[256];	// nazwa piku testowego
    char fNameResults[256];	// nazwa pliku z rezultatami
    
    int N; 					// iloœæ próbek treningowych
    int NTestSamples;		// iloœæ próbek testowych
    TVectorArray arrayX;	
    TFloatArray arrayY;		

    TFloatArray alpha;		// wektor alpha
    TFloatArray d;			
    TFloatArray arrayError;	
>>>>>>> Update
    float b;
    float bDiff;
    
};


