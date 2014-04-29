import ffmpeg
import re, os, sys
import numpy as np

def lb(videoListName, numBins, outputBase, pID):
    f = open(videoListName)
    videos = [line.rstrip() for line in f.readlines()]
    f.close()
    # remove processed videos, this step should be really fast
    videos = [video for video in videos if not os.path.isfile(os.path.join(outputBase, os.path.basename(video))+'.log')]
    videos = [(video, ffmpeg.get_duration(video)) for video in videos]
    videos = [video for video in videos if video[1] > 0]
    videos = sorted(videos, key=lambda tup: tup[1])
    bins = np.zeros([numBins, 1])

    selectedVids = []
    for video in videos:
        binNum = bins.argmin()
        bins[binNum] = bins[binNum] + video[1]
        if binNum == pID:
            selectedVids.append(video[0])
    return selectedVids

