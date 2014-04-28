#include "fisher.h"
#include "feature.h"

/*
 * A sample use of fisher vector coding with DT.
 * Require 5 PCA projection matrices and 5 GMM codebooks 
 * for each component of DT: TRAJ, HOG, HOF, MBHX, MBHY
 * Read input from stdin
 */

int main(int argc, char **argv) {
    if (argc < 4)   {
        cout<<"Usage: "<<argv[0]<<" pcaList codeBookList outputBase [windowSize] [stepSize] [intSize]"<<endl;
        return 0;
    }
    // Important: GMM uses OpenMP to speed up
    // This will cause problem on HPCC where all
    // cores from a node run this binary
    vl_set_num_threads(1);
    string pcaList(argv[1]);
    string codeBookList(argv[2]);
    string outputBase(argv[3]);
    int windowSize = 100;
    int stepSize = 50;
    int intSize = 50;
    if (argc > 4)
        windowSize = atoi(argv[4]);
    if (argc > 5)
        stepSize = atoi(argv[5]);
    if (argc > 6)
        intSize = atoi(argv[6]);
    if (windowSize % intSize != 0 || stepSize % intSize != 0)   {
        cout<<"Integral size must be the gcd of window size and step size."<<endl;
        return 0;
    }
    string types[5] = {"traj", "hog", "hof", "mbhx", "mbhy"};
    vector<FisherVector*> fvs(5, NULL);

    ifstream fin1, fin2;
    fin1.open(pcaList.c_str());
    if (!fin1.is_open())    {
        cout<<"Cannot open "<<pcaList<<endl;
        return 0;
    }
    fin2.open(codeBookList.c_str());
    if (!fin2.is_open())    {
        cout<<"Cannot open "<<codeBookList<<endl;
        return 0;
    }
    string pcaFile, codeBookFile;
    //for (int i = 0; i < fvs.size(); i++)    {
    for (int i = 1; i < fvs.size(); i++)    {
        getline(fin1, pcaFile);
        getline(fin2, codeBookFile);
        fvs[i] = new FisherVector(pcaFile, codeBookFile);
        fvs[i]->initFV(1, intSize);  // 1 layer of spatial pyramids
    }
    fin1.close();
    fin2.close();

    string line;
    while (getline(cin, line))  {
        DTFeature feat(line);
        //TODO: Store feature of DT with vector<double>
        //vector<double> traj(feat.traj, feat.traj+TRAJ_DIM);
        vector<double> hog(feat.hog, feat.hog+HOG_DIM);
        vector<double> hof(feat.hof, feat.hof+HOF_DIM);
        vector<double> mbhx(feat.mbhx, feat.mbhx+MBHX_DIM);
        vector<double> mbhy(feat.mbhy, feat.mbhy+MBHY_DIM);
        //fvs[0]->addPoint(traj, feat.x_pos, feat.y_pos, feat.frameNum);
        fvs[1]->addPoint(hog, feat.x_pos, feat.y_pos, feat.frameNum);
        fvs[2]->addPoint(hof, feat.x_pos, feat.y_pos, feat.frameNum);
        fvs[3]->addPoint(mbhx, feat.x_pos, feat.y_pos, feat.frameNum);
        fvs[4]->addPoint(mbhy, feat.x_pos, feat.y_pos, feat.frameNum);
    }

    cout<<"Points load complete."<<endl;
    //for (int i = 0; i < fvs.size(); i++)    {
    for (int i = 1; i < fvs.size(); i++)    {
        ofstream fout;
        string outName = outputBase + "." + types[i] + ".fv.txt";
        fout.open(outName.c_str());
        vector<double> fv = fvs[i]->getFV();
        fout<<fv[0];
        for (int j = 1; j < fv.size(); j++)
            fout<<" "<<fv[j];
        fout<<endl;
        fout.close();
        outName = outputBase + "." + types[i] + ".fv.seq";
        fout.open(outName.c_str());
        vector<vector<double> > localFVs = fvs[i]->getFV(stepSize, windowSize);
        for (int j = 0; j < localFVs.size(); j++)   {
            fout<<localFVs[j][0];
            for (int k = 1; k < localFVs[j].size(); k++)
                fout<<" "<<localFVs[j][k];
            fout<<endl;
        }
        fout.close();
        fvs[i]->clearFV();
    }

    for (int i = 1; i < fvs.size(); i++)
    //for (int i = 0; i < fvs.size(); i++)
        delete fvs[i];
    return 0;
}
