#!/bin/bash
#####################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#####################################################################################
# Extract audio (.wav) from videos (.mp4)
# (from original study, only tasks 2 and 3 (since task 4 does not have speech!))
# & Downmix stereo into mono stream 
# & Downsample to 16kHz
#####################################################################################

CNT=0
#for f in ./../Dataset/Videos/PID*[!AR]Task[!4].mp4; do 
for f in ./../Dataset/Videos/PID*.mp4; do 
# for f in ./../Dataset/Videos/PID20Task2.mp4; do 

	# Extract wav from mp4 
    # & Downmix stereo into mono stream (https://trac.ffmpeg.org/wiki/AudioChannelManipulation)
    # Downsample to 16kHz
	ffmpeg -i $f -ac 1 -ar 16000 ./../Dataset/AudioWav_16kHz/${f:(-14):-4}.wav

	((CNT++))
done

echo "Total count of files: " $CNT

# => 40 .wavs, 20 subjects, 2 tasks each