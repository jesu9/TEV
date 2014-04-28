#include "tev.h"

TeVector::TeVector(string codeBookFile, string sigmaInvFile, string rMeanFile, string pcaFile)  {
    ifstream fin;
    fin.open(codeBookFile.c_str());
    string line;
    double val;
    int i;
    stringstream ss;
    if (!fin.is_open()) {
        cout<<"Cannot open "<<codeBookFile<<endl;
        return;
    }
    featDim = 0;
    while (getline(fin, line))  {
        ss<<line;
        vector<double> codeWord;
        while (ss>>val)
            codeWord.push_back(val);
        if (featDim > 0 && featDim != codeWord.size())  {
            cout<<"Codebook dimension not match"<<endl;
            exit(0);
        }   else if (featDim == 0)    {
            featDim = codeWord.size();
        }
        codeBook.push_back(codeWord);
        ss.clear();
        ss.str("");
    }
    fin.close();
    numClusters = codeBook.size();
    rDim = numClusters * featDim;
    fin.open(sigmaInvFile.c_str());
    sigmaInv = new double[rDim*rDim];
    i = 0;
    while (fin>>val)    {
        sigmaInv[i] = val;
        i++;
    }
    if (i != rDim*rDim) {
        cout<<"Sigma inverse matrix invalid"<<endl;
        exit(0);
    }
    fin.close();
    fin.open(rMeanFile.c_str());
    i = 0;
    r0 = new double[rDim];
    while (fin>>val)    {
        r0[i] = val;
        i++;
    }
    if (i != rDim)  {
        cout<<"R0 invalid"<<endl;
        exit(0);
    }
    fin.close();
    tevDim = rDim - featDim;
    eigenVectors = new double[tevDim*rDim];
    fin.open(pcaFile.c_str());
    i = 0;
    while (fin>>val)    {
        eigenVectors[i] = val;
        i++;
    }
    if (i != tevDim*rDim)   {
        cout<<"PCA matrix invalid"<<endl;
        exit(0);
    }
    fin.close();
};

TeVector::~TeVector()   {
    if (sigmaInv)
        delete [] sigmaInv;
    if (r0)
        delete [] r0;
    if (eigenVectors)
        delete [] eigenVectors;
};

bool TeVector::initTeV(int integralSize, int startFrame, int endFrame)   {
    if (startFrame < 0 || startFrame > endFrame)    {
        cout<<"Invalid range of frames."<<endl;
        return false;
    }
    this->intSize = integralSize;
    this->startFrame = startFrame;
    this->endFrame = endFrame;
    numPts = 0;
    if (tev.size() > 0)
        tev.clear();
    if (int_tev.size() > 0)
        int_tev.clear();
    fv.resize(tevDim, 0.0);
    vector<double> ltev(tevDim, 0.0);
    int_tev.push_back(ltev);
    return true;
};

bool TeVector::addPoint(vector<double> feat, int frameNum)  {
    if (frameNum < startFrame || frameNum > endFrame)
        return false;
    int intInd = frameNum/intSize;
    if (int_tev.size() < intInd + 1)    {
        vector<double> ltev(int_tev[0].size(), 0.0);
        int start = int_tev.size();
        for (int i = start; i < intInd + 1; i++)
            int_tev.push_back(ltev);
    }
    vector<double> ltev(tevDim, 0.0);
    // begin computing R
    for (int i = 0; i < numClusters; i++)   {
        double sum = 0.0;
        for (int j = 0; j < featDim; j++)   {
            ltev[i*featDim+j] = feat[j] - codeBook[i][j];
            sum += ltev[i*featDim+j]*ltev[i*featDim+j];
        }
        sum = sqrt(sum);
        if (sum < 1e-10)
            continue;
        for (int j = 0; j < featDim; j++)
            ltev[i*featDim+j] /= sum;
    }
    // compute complete, try average aggregation first
    for (int i = 0; i < ltev.size(); i++)   {
        tev[i] += ltev[i];
        int_tev[intInd][i] += ltev[i];
    }
};

vector<double> &TeVector::getTeV()  {

};

vector<vector<double> > TeVector::getTeV(int stepSize, int windowSize)    {

};

bool TeVector::normTeV(vector<double> &v) {

};

bool TeVector::clearTeV() {

};


