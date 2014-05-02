#include "tev.h"

TeVector::TeVector(string codeBookFile, string rMeanFile, string projMatFile)  {
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
    tevDim = rDim - featDim;
    fin.open(projMatFile.c_str(), ios::in|ios::binary);
    projMat = new double[tevDim*rDim];
    i = 0;
    fin.read((char*) projMat, tevDim*rDim*sizeof(double));
    /*while (fin>>val)    {
        projMat[i] = val;
        i++;
    }
    if (i != tevDim*rDim) {
        cout<<"Projection matrix invalid"<<endl;
        exit(0);
    }*/
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
    
    // manage cache
    cacheSize = rDim/2;
    cachePtr = 0;
    cache = new double[cacheSize*rDim];
    ind_cache = new int[cacheSize];
};

TeVector::~TeVector()   {
    if (projMat)
        delete [] projMat;
    if (r0)
        delete [] r0;
    if (cache)
        delete [] cache;
    if (ind_cache)
        delete [] ind_cache;
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
    tev.resize(tevDim, 0.0);
    vector<double> ltev(tevDim, 0.0);
    int_tev.push_back(ltev);

    cachePtr = 0;

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
    vector<double> ltev(rDim, 0.0);
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
    // minus by R0
    for (int i = 0; i < rDim; i++)
        ltev[i] = ltev[i] - r0[i];
    // add to cache
    for (int i = 0; i < rDim; i++)
        cache[cachePtr*rDim+i] = ltev[i];
    ind_cache[cachePtr] = intInd;
    cachePtr++;
    if (cachePtr > cacheSize - 1)
        aggregate();
    numPts++;
    return true;
};

void TeVector::aggregate()  {
    // multiply by projection matrix
    if (cachePtr < 1)
        return;
    int m = cachePtr;
    int n = tevDim;
    int k = rDim;
    double *C = new double[tevDim*cachePtr];
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasTrans, m, n, k, 1, cache, k, projMat, k, 0, C, n);
    // normalize with l2 norm
    for (int c = 0; c < cachePtr; c++)  {
        vector<double> ltev(C+c*tevDim, C+(c+1)*tevDim);
        int intInd = ind_cache[c];
        double sum = 0.0;
        for (int i = 0; i < ltev.size(); i++)
            sum += ltev[i]*ltev[i];
        sum = sqrt(sum);
        if (sum > 1e-10)    {
            for (int i = 0; i < ltev.size(); i++)
                ltev[i] /= sum;
        }
        // compute complete, try average aggregation first
        for (int i = 0; i < ltev.size(); i++)   {
            tev[i] += ltev[i];
            int_tev[intInd][i] += ltev[i];
        }
    }
    
    delete [] C;
    cachePtr = 0;
};

vector<double> TeVector::getTeV()  {
    // in case there are points not aggregated yet
    aggregate();
    if (numPts < 1)
        return tev;
    vector<double> retTeV = tev;
    normTeV(retTeV);
    return retTeV;
};

vector<vector<double> > TeVector::getTeV(int stepSize, int windowSize)    {
    // in case there are points not aggregated yet
    aggregate();
    vector<vector<double> > retTeVs;
    if (stepSize % intSize != 0 || windowSize % intSize != 0)   {
        cout<<"Integral window size should be the gcd of step size and window size"<<endl;
        return retTeVs;
    }
    for (int frameNum = 0; ; frameNum += stepSize)  {
        int beginInd = frameNum / intSize;
        int endInd = (frameNum + windowSize) / intSize;
        if (beginInd > int_tev.size() - 1)
            break;
        if (endInd > int_tev.size())
            endInd = int_tev.size();
        vector<double> localTeV(int_tev[0].size(), 0.0);
        for (int i = beginInd; i < endInd; i++) {
            for (int j = 0; j < int_tev[i].size(); j++)
                localTeV[j] += int_tev[i][j];
        }
        normTeV(localTeV);
        retTeVs.push_back(localTeV);
    }
    return retTeVs;
};

void TeVector::normTeV(vector<double> &v) {
    // TODO: Implement RN
    // Power-law norm
    double sum = 0.0;
    for (int i = 0; i < v.size(); i++)  {
        if (v[i] < 0)
            v[i] = - sqrt(-v[i]);
        else
            v[i] = sqrt(v[i]);
        sum += v[i]*v[i];
    }
    sum = sqrt(sum);
    if (sum < 1e-10)
        return;
    for (int i = 0; i < v.size(); i++)
        v[i] /= sum;
};

bool TeVector::clearTeV() {
    intSize = 50;
    startFrame = 0;
    endFrame = INT_MAX;
    tev.clear();
    int_tev.clear();
    numPts = 0;
    cachePtr = 0;
    return true;
};


