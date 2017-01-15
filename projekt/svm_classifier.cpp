#include "svm_additional_fnc.h"
#include "svm_classifier.h"

using std::ifstream;
using std::ofstream;
using std::setprecision;
using std::cerr;
using std::cout;
using std::endl;

SVMClassifier::SVMClassifier() {
	this->C = 1.0;
	this->epsilon = 0.0001;
	strcpy(fNameTrain, "trainset.csv");
	strcpy(fNameResults, "resul.csv");
	strcpy(fNameTest, "testset.csv");

	N = 0;
	NTestSamples = 0;
	b = 0.0;
	bDiff = 0.0;
}

SVMClassifier::~SVMClassifier() {}

float SVMClassifier::kernel(int i1, int i2) {
	float k = dotProduct(arrayX[i1], arrayX[i2]);
	k += C;
	k *= k;
	return k;
}

float SVMClassifier::learnedFnc(int k) {
	float s = 0.0;
	for (int i = 0; i < N; i++) {
		if (alpha[i] > 0) {
			s += alpha[i] * arrayY[i] * kernel(i, k);
		}
	}

	s -= b;
	return s;
}

int SVMClassifier::predict() {

	arrayX.clear();
	arrayY.clear();
	alpha.clear();

	ifstream resultsFileIn(fNameResults);
	loadResults(resultsFileIn);

	ifstream testFileIn(fNameTest);
	partReadSample(testFileIn, arrayX, arrayY, N);

	N = N + NTestSamples;

	d.resize(N);
	for (int i = 0; i < N; i++) {
		d[i] = dotProduct(arrayX[i], arrayX[i]);
	}


	int rightClassified = 0;
	float yClassifiedLabel = 0.0;

	for (int i = NTestSamples; i < N; i++) {
		float s = 0.0;

		for (int j = 0; j < NTestSamples; j++) {
			s += alpha[j] * arrayY[j] * kernel(i, j);
		}

		s -= b;
		yClassifiedLabel = s >= 0.0 ? 1.0 : -1.0;
		cout << yClassifiedLabel << endl;
		if ((yClassifiedLabel > 0 && arrayY[i] > 0) || (yClassifiedLabel < 0 && arrayY[i] < 0)) {
			rightClassified++;
		}
	}

	cerr << setprecision(5)
		<< "Accuracy: " << 100.0 * rightClassified / (N - NTestSamples)
		<< "% (" << rightClassified << "/" << (N - NTestSamples) << ")"
		<< endl;

	return 0;
}

int SVMClassifier::train() {

	arrayX.clear();
	arrayY.clear();
	d.clear();

	ifstream trainFileIn(fNameTrain);
	partReadSample(trainFileIn, arrayX, arrayY, N);

	alpha.resize(N, 0.0);
	b = 0.0;
	arrayError.resize(N, 0.0);

	d.resize(N);
	for (int i = 0; i < N; i++) {
		d[i] = dotProduct(arrayX[i], arrayX[i]);
	}


	int numChangedAlpha = 0;
	int prevNumChangedAlpha = 0;
	int examineAll = 1;
	int rounds = 0;
	int sameRounds = 0;
	while ((numChangedAlpha > 0 || examineAll) && rounds < NUMBER_OF_ITERATIONS && sameRounds < ITERATIONS_WITH_CONST_ERR) {
		prevNumChangedAlpha = numChangedAlpha;
		numChangedAlpha = 0;

		rounds++;
		if (examineAll) {
			for (int k = 0; k < N; k++) {
				numChangedAlpha += examineExample(k);
			}
		}
		else {
			for (int k = 0; k < N; k++) {
				if (alpha[k] != 0 && alpha[k] != C) {
					numChangedAlpha += examineExample(k);
				}
			}
		}

		if (examineAll == 1) {
			examineAll = 0;
		}
		else if (numChangedAlpha == 0) {
			examineAll = 1;
		}

		cerr << setprecision(5)
			<< "Iteration: " << rounds
			<< "\tError rate : " << errorRate()
			<< endl;
		if (prevNumChangedAlpha == numChangedAlpha)
		{
			sameRounds++;
		}
		else
			sameRounds = 0;
		for (int i = 0; i < N; i++) {
			if (alpha[i] < TOLERANCE) {
				alpha[i] = 0.0;
			}
		}
	}

	ofstream fileResultOut(fNameResults);
	writeResultModel(fileResultOut);

	return 0;
}

void SVMClassifier::writeResultModel(ofstream& os) {
	TString s;

	os << b << endl;
	NTestSamples = 0;
	for (int i = 0; i < N; i++) {
		if (alpha[i] > 0) {
			NTestSamples += 1;
		}
	}
	os << NTestSamples << endl;

	for (int i = 0; i < N; i++) {
		if (alpha[i] > 0) {
			os << alpha[i] << endl;
		}
	}

	for (int i = 0; i < N; i++) {
		if (alpha[i] > 0) {
			writeSample(s, arrayX[i], arrayY[i]);
			os << s << endl;
		}
	}

}

int SVMClassifier::loadResults(ifstream& is) {

	int d = 0;
	int m = 0;
	TString s;

	is >> b;
	is >> NTestSamples;
	alpha.resize(NTestSamples, 0.0);
	for (int i = 0; i < NTestSamples; i++) {
		is >> alpha[i];
	}
	getline(is, s, '\n');
	partReadSample(is, arrayX, arrayY, m);

	return 0;
}
double drand48()
{
	return  ((double)rand() / (RAND_MAX)) + 1;
}

float SVMClassifier::errorRate() {
	int numberOfErrors = 0;
	for (int i = 0; i < N; i++) {
		if ((learnedFnc(i) >= 0 && arrayY[i] < 0) || (learnedFnc(i) < 0 && arrayY[i] > 0)) {
			numberOfErrors++;
		}
	}
	return 1.0 * numberOfErrors / N;
}

int SVMClassifier::examineExample(int i1) {
	float y1 = 0.0;
	float alpha1 = 0.0;
	float e1 = 0.0;
	float r1 = 0.0;

	y1 = arrayY[i1];
	alpha1 = alpha[i1];
	if (alpha1 > 0 && alpha1 < C) {
		e1 = arrayError[i1];
	}
	else {
		e1 = learnedFnc(i1) - y1;
	}

	r1 = y1 * e1;
	if ((r1 < -TOLERANCE && alpha1 < C) || (r1 > TOLERANCE && alpha1 > 0)) {
		int k0 = 0;
		int k = 0;
		int i2 = -1;
		float tmax = 0.0;
		for (i2 = -1, tmax = 0, k = 0; k < N; k++) {
			if (alpha[k] > 0 && alpha[k] < C) {
				float e2 = 0.0;
				float temp = 0.0;
				e2 = arrayError[k];
				temp = fabs(e1 - e2);
				if (temp > tmax) {
					tmax = temp;
					i2 = k;
				}
			}
			if (i2 >= 0) {
				if (takeStep(i1, i2)) {
					return 1;
				}
			}
		}
		for (k0 = (int)(drand48() * N), k = k0; k < N + k0; k++) {
			i2 = k % N;
			if (alpha[i2] > 0 && alpha[i2] < C) {
				if (takeStep(i1, i2)) {
					return 1;
				}
			}
		}
		for (k0 = (int)(drand48() * N), k = k0; k < N + k0; k++) {
			i2 = k % N;
			if (takeStep(i1, i2)) {
				return 1;
			}
		}
	}
	return 0;
}

int SVMClassifier::takeStep(int i1, int i2) {
	float a1 = 0.0;
	float a2 = 0.0;
	float e1 = 0.0;
	float e2 = 0.0;

	

	if (i1 == i2) {
		return 0;
	}

	float alpha1 = alpha[i1];
	float alpha2 = alpha[i2];
	int y1 = arrayY[i1];
	int y2 = arrayY[i2];

	if (alpha1 > 0 && alpha1 < C) {
		e1 = arrayError[i1];
	}
	else {
		e1 = learnedFnc(i1) - y1;
	}
	if (alpha2 > 0 && alpha2 < C) {
		e2 = arrayError[i2];
	}
	else {
		e2 = learnedFnc(i2) - y2;
	}
	int s = y1 * y2;
	float L = 0.0;
	float H = 0.0;
	if (y1 == y2) {
		float gamma = alpha1 + alpha2;
		if (gamma > C) {
			L = gamma - C;
			H = C;
		}
		else {
			L = 0;
			H = gamma;
		}
	}
	else {
		float gamma = alpha1 - alpha2;
		if (gamma > 0) {
			L = 0;
			H = C - gamma;
		}
		else {
			L = -gamma;
			H = C;
		}
	}

	if (fabs(L - H) < 1e-6) {
		return 0;
	}
	float k11 = 0.0;
	float k22 = 0.0;
	float k12 = 0.0;
	k11 = kernel(i1, i1);
	k12 = kernel(i1, i2);
	k22 = kernel(i2, i2);
	float eta = 2 * k12 - k11 - k22;

	float Lobj = 0.0;
	float Hobj = 0.0;
	if (eta < 0) {
		a2 = alpha2 + y2 * (e2 - e1) / eta;
		if (a2 < L) {
			a2 = L;
		}
		else if (a2 > H) {
			a2 = H;
		}
	}
	else {
		float c1 = eta / 2.0;
		float c2 = y2 * (e1 - e2) - eta * alpha2;
		Lobj = c1 * L * L + c2 * L;
		Hobj = c1 * H * H + c2 * H;
		if (Lobj > Hobj + epsilon) {
			a2 = L;
		}
		else if (Lobj < Hobj - epsilon) {
			a2 = H;
		}
		else {
			a2 = alpha2;
		}
	}

	if (fabs(a2 - alpha2) < epsilon * (a2 + alpha2 + epsilon)) {
		return 0;
	}
	a1 = alpha1 + s * (alpha2 - a2);
	if (a1 < 0) {
		a2 += s * a1;
		a1 = 0;
	}
	else if (a1 > C) {
		float t = a1 - C;
		a2 += s * t;
		a1 = C;
	}

	float b1 = 0.0;
	float b2 = 0.0;
	float bnew = 0.0;
	if (a1 > 0 && a1 < C) {
		bnew = b + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
	}
	else if (a2 > 0 && a2 < C) {
		bnew = b + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
	}
	else {
		b1 = b + e1 + y1 * (a1 - alpha1) * k11 + y2 * (a2 - alpha2) * k12;
		b2 = b + e2 + y1 * (a1 - alpha1) * k12 + y2 * (a2 - alpha2) * k22;
		bnew = (b1 + b2) / 2.0;
	}

	bDiff = bnew - b;
	b = bnew;

	float t1 = y1 * (a1 - alpha1);
	float t2 = y2 * (a2 - alpha2);

	for (int i = 0; i < N; i++) {
		if (alpha[i] > 0 && alpha[i] < C) {
			arrayError[i] += t1 * kernel(i1, i) + t2 * kernel(i2, i) - bDiff;
		}
	}

	arrayError[i1] = 0.0;
	arrayError[i2] = 0.0;

	alpha[i1] = a1;
	alpha[i2] = a2;
	return 1;
}

