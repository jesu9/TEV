# python wrapper for basic ffmpeg operations
# resize video, check if a video is corrupted, etc.

import subprocess, re, os

# provide your own ffmpeg here
ffmpeg = '/auto/iris-00/rn/shared/bin/ffmpeg'

# resize videoName to 320x240 and store in resizedName
# if succeed return True
def resize(videoName, resizedName):
    if not os.path.exists(videoName):
        print '%s does not exist!' % videoName
        return False
    # call ffmpeg and grab its stderr output
    p = subprocess.Popen([ffmpeg, "-i", videoName], stderr=subprocess.PIPE)
    out, err = p.communicate()
    # search resolution info
    if err.find('differs from') > -1:
        return False
    reso = re.findall(r'Video.*, ([0-9]+)x([0-9]+)', err)
    if len(reso) < 1:
        return False
    # call ffmpeg again to resize
    subprocess.call([ffmpeg, '-i', videoName, '-s', '320x240', resizedName])
    return check(resizedName)

# check the video file is corrupted or not
def check(videoName):
    if not os.path.exists(videoName):
        return False
    p = subprocess.Popen([ffmpeg, '-i', videoName], stderr=subprocess.PIPE)
    out, err = p.communicate()
    if err.find('Invalid') > -1:
        return False
    return True

# TODO: Get the duration of a video
def get_duration(videoName):
    if not os.path.exists(videoName):
        return -1
    p = subprocess.Popen([ffmpeg, '-i', videoName], stderr=subprocess.PIPE)
    out, err = p.communicate()
    duration = re.findall(r'Duration: (\d\d):(\d\d):(\d\d)', err)
    if len(duration) < 1:
        return -1
    duration = duration[0]
    return 3600*int(duration[0])+60*int(duration[1])+int(duration[2])
