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

#include "svm_additional_fnc.h"

#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H


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

    int examineExample(int i1);
    int takeStep(int i1, int i2);

    double C;
    double epsilon;
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
    float b;
    float bDiff;
    
};

#endif
