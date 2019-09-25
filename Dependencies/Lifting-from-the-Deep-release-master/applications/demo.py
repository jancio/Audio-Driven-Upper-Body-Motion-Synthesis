#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Dec 20 17:39 2016

@author: Denis Tome'
"""
#######################################################################################################################
# Original code modified for the project:
#
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#######################################################################################################################
# Run LFTD to extract x, y, z joint coordinates of all joints from all frames from all videos.
# Save the joint coordinates as a .npz file, one for each video.
# (Used on HPC, to extract all 40 videos.)
#######################################################################################################################

import __init__

from lifting import PoseEstimator
from lifting.utils import draw_limbs
from lifting.utils import plot_pose
#from lifting.utils import Prob3dPose

import cv2
import matplotlib.pyplot as plt
from os.path import dirname, realpath
import glob
import numpy as np

DIR_PATH = dirname(realpath(__file__))
PROJECT_PATH = realpath(DIR_PATH + '/..')
SAVED_SESSIONS_DIR = PROJECT_PATH + '/data/saved_sessions'
SESSION_PATH = SAVED_SESSIONS_DIR + '/init_session/init'
PROB_MODEL_PATH = SAVED_SESSIONS_DIR + '/prob_model/prob_model_params.mat'

IMAGE_SIZE = (1080, 1920, 3)


def main():

    # Get video IDs (VIDs) already processed
    VIDsDONE = [x.split('/')[-1][:-4] for x in glob.glob('./*.npz')]
    print VIDsDONE

    # Extract 3D joints for all VIDs
    for dir_path in sorted(glob.glob('/home/jo356/rds/hpc-work/P3/ImgSeq/*')):

        VID = dir_path.split('/')[-1]
        # Skip VIDs that were already processed
        if VID in VIDsDONE:
            continue
        print VID

        # If the first frame is missing, then record the number of first frames missing and later when a non-missing frame comes in, include it (1 + N_first_skipped)-times
        N_first_skipped = 0

        joints_3D = []
        # Count missing/skipped frames
        cnt = 0    

        img_filenames = sorted(glob.glob(dir_path + '/*.jpg'))
        # Create pose estimator
        pose_estimator = PoseEstimator(IMAGE_SIZE, SESSION_PATH, PROB_MODEL_PATH)
        # Load model
        pose_estimator.initialise()

        for frame_num, img_filename in enumerate(img_filenames):

            image = cv2.imread(img_filename)
            # Conversion to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            # Estimation
            try:
                pose_2d, visibility, pose_3d = pose_estimator.estimate(image)
                joints_3D.append( pose_3d[0].T )

                if N_first_skipped > 0:
                    for i in range(N_first_skipped):
                        joints_3D.append( pose_3d[0].T )
                    N_first_skipped = 0

                #print pose_3d
                #print pose_3d[0].T
                #print np.reshape( np.reshape(pose_3d[0], (17, 3)), (3,17) )
                #print "wrong reshaping !!!"


                print img_filename.split('/')[-1]

            # Frame missing => pose was not recognised/estimated reliably
            except ValueError: 
                cnt += 1
                print img_filename.split('/')[-1], " is missing"

                # If the first frame is missing or a non-missing frame still did not come in, then just record the number to replicate later when a non-missing frame comes in
                if frame_num == 0 or N_first_skipped > 0:
                    N_first_skipped += 1
                # Replicate previous frame
                else:
                    joints_3D.append( joints_3D[-1] )
            # Treated as above
            except: 
                print "2D joints not identified:", sys.exc_info()[0]
                cnt += 1
                print img_filename.split('/')[-1], " is missing"

                # If the first frame is missing or a non-missing frame still did not come in, then just record the number to replicate later when a non-missing frame comes in
                if frame_num == 0 or N_first_skipped > 0:
                    N_first_skipped += 1
                # Replicate previous frame
                else:
                    joints_3D.append( joints_3D[-1] )

            # print pose_2d
            # print pose_3d
            # print visibility

        # Close model
        pose_estimator.close()

        np.savez('./' + VID + '.npz', joints_3D=np.reshape(joints_3D, (-1, 17, 3)), skipped_frames=cnt)
        print "Skipped frames: ",cnt," / ",len(img_filenames)

        # Show 2D and 3D poses
        #display_results(image, pose_2d, visibility, pose_3d)


def display_results(in_image, data_2d, joint_visibility, data_3d):
    """Plot 2D and 3D poses for each of the people in the image."""
    plt.figure()
    draw_limbs(in_image, data_2d, joint_visibility)
    plt.imshow(in_image)
    plt.axis('off')

    # Show 3D poses
    for single_3D in data_3d:
        #plot_pose(Prob3dPose.centre_all(single_3D))
        plot_pose(single_3D)

    plt.show()

if __name__ == '__main__':
    import sys
    sys.exit(main())
