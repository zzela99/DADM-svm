#include <sstream>
#include <string>
#include <string.h>
#include <vector>
#include <ctime>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <strstream>


#ifndef SVM_ADDITIONAL_FNC_H
#define SVM_ADDITIONAL_FNC_H

#define TOLERANCE 1e-6
#define NUMBER_OF_ITERATIONS 30000
#define ITERATIONS_WITH_CONST_ERR 3000

typedef std::string TString;
typedef std::vector<std::string> TStringArray;

typedef std::pair<int, float> TVectorDim;
typedef std::vector<TVectorDim> TVector;
typedef std::vector<TVector> TVectorArray;

typedef std::vector<float> TFloatArray;

int splitCSV(const TString& s, char c, TStringArray& v);
bool cmp(const TVectorDim& p1, const TVectorDim& p2);

int readSample(TString& s, TVector& x, float& y);
int writeSample(TString& s, TVector& x, float& y);
int partReadSample(std::ifstream& is, TVectorArray& arrayX, TFloatArray& arrayY, int& n);
int partWriteSample(std::ofstream& os, TVectorArray& arrayX, TFloatArray& arrayY, int& n);

float dotProduct(const TVector& v1, const TVector& v2);
TVector operator*(const TVector& v, float f);
TVector operator*(float f, const TVector& v);
TVector operator+(const TVector& v1, const TVector& v2);

#endif
