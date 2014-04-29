#ifndef TRIANGLE_EMBEDDING_H
#define TRIANGLE_EMBEDDING_H

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <mkl.h>

using namespace std;

class TeVector  {
    public:
        TeVector(string codeBookFile, string rMeanFile, string projMatFile);
        ~TeVector();
        bool initTeV(int integralSize = 50, int startFrame = 0, int endFrame = INT_MAX);
        bool addPoint(vector<double> feat, int frameNum);
        vector<double> getTeV();
        vector<vector<double> > getTeV(int stepSize, int windowSize);
        void normTeV(vector<double> &v);
        bool clearTeV();

    private:
        vector<vector<double> > codeBook;
        int featDim;
        int numClusters;
        int rDim;       // dimension of R
        int tevDim;     // after first featDim components removed
        double *projMat;
        double *r0;

        int intSize;
        int startFrame;
        int endFrame;

        vector<double> tev;
        vector<vector<double> > int_tev;
        int numPts;
};

#endif
