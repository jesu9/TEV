#include "tev.h"
#include "feature.h"

/*
 * A sample use of fisher vector coding with DT.
 * Require 5 PCA projection matrices and 5 GMM codebooks 
 * for each component of DT: TRAJ, HOG, HOF, MBHX, MBHY
 * Read input from stdin
 */

int main(int argc, char **argv) {
    if (argc < 5)   {
        cout<<"Usage: "<<argv[0]<<" projMatList rMeanList codeBookList outputBase [windowSize] [stepSize] [intSize]"<<endl;
        return 0;
    }
    string projMatList(argv[1]);
    string rMeanList(argv[2]);
    string codeBookList(argv[3]);
    string outputBase(argv[4]);
    int windowSize = 100;
    int stepSize = 100;
    int intSize = 100;
    if (argc > 5)
        windowSize = atoi(argv[5]);
    if (argc > 6)
        stepSize = atoi(argv[6]);
    if (argc > 7)
        intSize = atoi(argv[7]);
    if (windowSize % intSize != 0 || stepSize % intSize != 0)   {
        cout<<"Integral size must be the gcd of window size and step size."<<endl;
        return 0;
    }
    string types[5] = {"traj", "hog", "hof", "mbhx", "mbhy"};
    vector<TeVector*> tevs(5, NULL);

    ifstream fin1, fin2, fin3;
    fin1.open(projMatList.c_str());
    if (!fin1.is_open())    {
        cout<<"Cannot open "<<projMatList<<endl;
        return 0;
    }
    fin2.open(codeBookList.c_str());
    if (!fin2.is_open())    {
        cout<<"Cannot open "<<codeBookList<<endl;
        return 0;
    }
    fin3.open(rMeanList.c_str());
    if (!fin3.is_open())    {
        cout<<"Cannot open "<<rMeanList<<endl;
        return 0;
    }
    string projMatFile, codeBookFile, rMeanFile;
    //for (int i = 0; i < fvs.size(); i++)    {
    for (int i = 1; i < tevs.size(); i++)    {
        getline(fin1, projMatFile);
        getline(fin2, codeBookFile);
        getline(fin3, rMeanFile);
        tevs[i] = new TeVector(codeBookFile, rMeanFile, projMatFile);
        tevs[i]->initTeV(intSize);  // 1 layer of spatial pyramids
    }
    fin1.close();
    fin2.close();
    fin3.close();

    string line;
    cout<<"Start loading points..."<<endl;
    while (getline(cin, line))  {
        DTFeature feat(line);
        //TODO: Store feature of DT with vector<double>
        //vector<double> traj(feat.traj, feat.traj+TRAJ_DIM);
        vector<double> hog(feat.hog, feat.hog+HOG_DIM);
        vector<double> hof(feat.hof, feat.hof+HOF_DIM);
        vector<double> mbhx(feat.mbhx, feat.mbhx+MBHX_DIM);
        vector<double> mbhy(feat.mbhy, feat.mbhy+MBHY_DIM);
        //tevs[0]->addPoint(traj, feat.frameNum);
        tevs[1]->addPoint(hog, feat.frameNum);
        tevs[2]->addPoint(hof, feat.frameNum);
        tevs[3]->addPoint(mbhx, feat.frameNum);
        tevs[4]->addPoint(mbhy, feat.frameNum);
    }

    cout<<"Points load complete."<<endl;
    //for (int i = 0; i < tevs.size(); i++)    {
    for (int i = 1; i < tevs.size(); i++)    {
        ofstream fout;
        string outName = outputBase + "." + types[i] + ".tev.txt";
        fout.open(outName.c_str());
        vector<double> tev = tevs[i]->getTeV();
        fout<<tev[0];
        for (int j = 1; j < tev.size(); j++)
            fout<<" "<<tev[j];
        fout<<endl;
        fout.close();
        outName = outputBase + "." + types[i] + ".tev.seq";
        fout.open(outName.c_str());
        vector<vector<double> > localTeVs = tevs[i]->getTeV(stepSize, windowSize);
        for (int j = 0; j < localTeVs.size(); j++)   {
            fout<<localTeVs[j][0];
            for (int k = 1; k < localTeVs[j].size(); k++)
                fout<<" "<<localTeVs[j][k];
            fout<<endl;
        }
        fout.close();
        tevs[i]->clearTeV();
    }

    for (int i = 1; i < tevs.size(); i++)
    //for (int i = 0; i < fvs.size(); i++)
        delete tevs[i];
    return 0;
}
