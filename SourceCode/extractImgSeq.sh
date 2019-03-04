#!/bin/bash
#####################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#####################################################################################
# Generate image sequences from videos
#####################################################################################

cwdir=$PWD
CNT=0
for f in ./../Dataset/Videos/PID*.mp4; do 

	mkdir ./../Dataset/ImgSeq/${f:(-14):-4}
	ffmpeg -i $f ./../Dataset/ImgSeq/${f:(-14):-4}/${f:(-14):-4}_%06d.jpg

	echo ${f:(-14):-4}
	#echo $cwdir/../Dataset/PoseFeatures/${f:(-14):-4}/

	((CNT++))
done

echo "Total count of files: " $CNT

# => 40 folders of json files (one per frame), 20 subjects, 2 tasks each

# 'PID02Task2',
# 'PID02Task3',
# 'PID05Task2',
# 'PID05Task3',
# 'PID06Task2',
# 'PID06Task3',
# 'PID08Task2',
# 'PID08Task3',
# 'PID09Task2',
# 'PID09Task3',
# 'PID10Task2',
# 'PID10Task3',
# 'PID11Task2',
# 'PID11Task3',
# 'PID13Task2',
# 'PID13Task3',
# 'PID15Task2',
# 'PID15Task3',
# 'PID16Task2',
# 'PID16Task3',
# 'PID17Task2',
# 'PID17Task3',
# 'PID18Task2',
# 'PID18Task3',
# 'PID19Task2',
# 'PID19Task3',
# 'PID20Task2',
# 'PID20Task3',
# 'PID21Task2',
# 'PID21Task3',
# 'PID22Task2',
# 'PID22Task3',
# 'PID23Task2',
# 'PID23Task3',
# 'PID24Task2',
# 'PID24Task3',
# 'PID25Task2',
# 'PID25Task3',
# 'PID26Task2',
# 'PID26Task3',