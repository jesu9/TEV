# extract DT features on the fly and select feature points randomly
# need select_pts and dt binaries

import subprocess, os, ffmpeg
import sys
from load_balance import lb
from datetime import datetime

dtBin = '/auto/iris-00/rn/shared/bin/DenseTrackStab'
tevBin = '/auto/iris-00/rn/chensun/MED_14/TEV/bin/compute_tev'
# HPCC temp directory
tmpDir = os.environ['TMPDIR']
# HPCC process id
pID = os.environ['PBS_VNODENUM']
# Projection matrix list
projMatList = '/auto/iris-00/rn/chensun/MED_14/TEV/codebooks/projMat.lst'
# KMeans list
codeBookList = '/auto/iris-00/rn/chensun/MED_14/TEV/codebooks/codeBook.lst'
# R0 list
rMeanList = '/auto/iris-00/rn/chensun/MED_14/TEV/codebooks/rMean.lst'

def extract(videoName, outputBase):
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    if check_dup(outputBase):
        print '%s processed' % videoName
        return True
    resizedName = os.path.join(tmpDir, os.path.basename(videoName))
    if not ffmpeg.resize(videoName, resizedName):
        resizedName = videoName     # resize failed, just use the input video
    subprocess.call('%s %s | %s %s %s %s %s' % (dtBin, resizedName, tevBin, projMatList, rMeanList, codeBookList, outputBase), shell=True)
    return True

def check_dup(outputBase):
    """
    Check if fv of all modalities have been extracted
    """
    if not os.path.isfile(outputBase+'.log'):
        return False
    return True

if __name__ == '__main__':
    videoList = sys.argv[1]
    outputBase = sys.argv[2]
    totalTasks = int(sys.argv[3])
    videos = lb(videoList, totalTasks, outputBase, int(pID))
    try:
        for video in videos:
            print pID, video
            outputName = os.path.join(outputBase, os.path.basename(video))
            # timing
            tstart = datetime.now()
            extract(video, outputName)
            tend = datetime.now()
            ttotal = tend - tstart
            log = open(outputName + '.log', 'w')
            log.write('%d\n' % ttotal.seconds)
            log.close()
    except IOError:
        sys.exit(0)
