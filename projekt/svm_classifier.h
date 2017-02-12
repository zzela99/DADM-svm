<<<<<<< HEAD
=======
//za��czenie wszystkich potrzebnych bibliotek
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
// za��czenie pliku nag��wkowego z funkcjami dodatkowymi
#include "svm_additional_fnc.h"


// deklaracja klasy SVMClassifier
class SVMClassifier{
public:					//sk�adniki publiczne klasy
    SVMClassifier(); 	//konstruktor
    ~SVMClassifier();	//destruktor	
    int train();		// metoda treningowa
    int predict();		// metoda predykcyjna
protected:				// sk�adniki chronione klasy

    float errorRate();	// warto�� b��du dla okre�lonej pr�bki testowej

    int loadResults(std::ifstream& is);			//wczytywanie pr�bek z pliku
    void writeResultModel(std::ofstream& os);	//
    
    float kernel(int i1, int i2); 				//metoda j�dra
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
    
    int N; 					// ilo�� pr�bek treningowych
    int NTestSamples;		// ilo�� pr�bek testowych
    TVectorArray arrayX;	
    TFloatArray arrayY;		

    TFloatArray alpha;		// wektor alpha
    TFloatArray d;			
    TFloatArray arrayError;	
>>>>>>> Update
    float b;
    float bDiff;
    
};


