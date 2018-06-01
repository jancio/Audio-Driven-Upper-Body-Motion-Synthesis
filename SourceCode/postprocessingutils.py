#####################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#####################################################################################
# Utility functions for post-processing operations (smoothing, calls to evaluation methods and saving of results) after prediction of movements. 
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import time
from geoutils import radToDeg, degToRad
from evalutils import inv_norm_Y, eval_test, eval_test2, eval_wo_truth, robot_angle_constraints_rad
from pykalman import KalmanFilter
from scipy.signal import butter, filtfilt
from sklearn.metrics import mean_squared_error

###############################################################################################################
# Apply low-pass Butterworth filter with given cut-off frequency
def smoother_LPBF(Y_raw, fc, constrain=False):
    fs = 100. # sampling frequency
    # For digital filters, Wn is normalized from 0 to 1, where 1 is the Nyquist frequency, pi radians/sample
    b, a = butter(N=5, Wn=2*fc/fs, btype='low', analog=False, output='ba')
    Y_smooth = filtfilt(b, a, Y_raw, axis=0)
    if constrain:
        for i, angle_vector in enumerate(Y_smooth):
            for j, angle in enumerate(angle_vector):
                Y_smooth[i,j] = max( min(angle, robot_angle_constraints_rad[1,j]), robot_angle_constraints_rad[0,j] )
    return Y_smooth

###############################################################################################################
# Apply Kalman Smoother with EM algo
# DOES NOT PERFORM WELL ! not used
def smoother_KS(Y_raw):
    raise ValueError("Do you really want to use Kalman Smoother?")
    N_targets = Y_raw.shape[1]
    FPS = 100. # After linear interpolation
    dt = 1./FPS
    # Minimum jerk model
    TM = np.array([[1., dt, 0.5 * (dt**2)],   # transition matrix
                   [0., 1.,            dt],
                   [0., 0.,            1.]])
    OM = np.array([[1., 0., 0.]])             # observation matrix

    Y_smooth = np.copy(Y_raw)
    for iA in range(N_targets): # omitting HipPitch, iA is angle index
        kf = KalmanFilter(
            transition_matrices = TM, observation_matrices = OM
            # initial_state_mean = initial_state_mean, 
            # em_vars=['transition_covariance', 'initial_state_covariance']
            # default: transition_covariance, observation_covariance, initial_state_mean, and initial_state_covariance
        )
        # EM algorithm to estimate paramters
        kf = kf.em(Y_raw[:, iA])#, n_iter=5) # default 15 iterations
        # Kalman smoothing (fixed-interval, RST method)
        Y_smooth[:, iA] = kf.smooth(Y_raw[:, iA])[0][:,0]
    return Y_smooth
        

###############################################################################################################
# Smooth data
def post_smooth(Y_raw, smoother):
    if smoother == 'KS':
        Y_smooth = smoother_KS(Y_raw)
    elif smoother[:5] == 'LPBF_':
        fc = float(smoother.split('_')[-1]) # cut-off frequency
        Y_smooth = smoother_LPBF(Y_raw, fc)
    else:
        raise ValueError("Incorrect/unknown smoother specified!")
    return Y_smooth

###############################################################################################################
# Save true Y, raw and smoothed predicted Y to .npz: input in 0-1 scale; 
# angles saved in RADIANS (for generation on robot); evaluation in DEGREES (for plotting);
# Also do evaluation
def save_predictions_and_eval(filename, X_test, Y_test_true, Y_test_pred, model_type, seg_len, test_VIDs, test_VIDs_ind_cnts, 
                    SD_offsets=None, smoother='LPBF_4', N_params=None, N_epochs=None):
    '''
    smoother: 'KS'     => Kalman Smoother with EM algo; 
              'LPBF_4' => low-pass Butterworth filter with cut-off frequency 4 Hz
    '''
    N_targets = Y_test_true.shape[1]
    
    # Record overall loss (as optimsed): in 0-1 range
    test_loss = mean_squared_error(Y_test_true, Y_test_pred)
    
    # Convert to radians
    Y_test_pred_rad = inv_norm_Y(Y_test_pred)
    Y_test_true_rad = inv_norm_Y(Y_test_true)

    Y_raw_list = [] # list by testing VIDs
    Y_smooth_list = [] # list by testing VIDs
    Y_true_list = []

    # Iterate over testing VIDs
    offset = 0
    for i, cnt in enumerate(test_VIDs_ind_cnts):
        Y_raw_list.append( Y_test_pred_rad[offset:offset+cnt] )
        Y_true_list.append( Y_test_true_rad[offset:offset+cnt] )
        offset += cnt

    ######################################
    # Apply the specified smoother; for each VID
    for Y_raw in Y_raw_list: # iterate over testing VIDs
        Y_smooth = post_smooth(Y_raw, smoother)
        Y_smooth_list.append( Y_smooth )
        
    ######################################
    # Evaluate both: raw Y and smoothed Y
    print "Raw Y evaluation ..."
    results_raw = eval_test(X_test, Y_test_true_rad, Y_test_pred_rad, model_type, seg_len, test_VIDs, test_VIDs_ind_cnts)
    print "Smoothed Y evaluation ..."
    results_smooth = eval_test(X_test, Y_test_true_rad, np.concatenate(Y_smooth_list, axis=0), 
                               model_type, seg_len, test_VIDs, test_VIDs_ind_cnts)
    
    # Save results
    np.savez(filename + '.npz', 
             X_test=X_test, 
             Y_true_list=Y_true_list, Y_raw_list=Y_raw_list, Y_smooth_list=Y_smooth_list, 
             test_VIDs=test_VIDs, SD_offsets=SD_offsets, 
             results_raw=results_raw, results_smooth=results_smooth, 
             N_params=N_params, N_epochs=N_epochs, test_loss=test_loss)

    print "Saved (raw & smoothed predictions with results) to:", filename
    
###############################################################################################################
# As above but fixed JERK calculation 
# & evaluation is done only on smoothed data (not raw any more)
# & no calculation of global CCA
def save_predictions_and_eval2(filename, X_test, Y_test_true, Y_test_pred, model_type, seg_len, test_VIDs, test_VIDs_ind_cnts, 
                    SD_offsets=None, smoother='LPBF_4', N_params=None, N_epochs=None):
    '''
    smoother: 'KS'     => Kalman Smoother with EM algo; 
              'LPBF_4' => low-pass Butterworth filter with cut-off frequency 4 Hz
    '''
    N_targets = Y_test_true.shape[1]
    
    # Record overall loss (as optimsed): in 0-1 range
    test_loss = mean_squared_error(Y_test_true, Y_test_pred)
    
    # Convert to radians
    Y_test_pred_rad = inv_norm_Y(Y_test_pred)
    Y_test_true_rad = inv_norm_Y(Y_test_true)

    Y_raw_list = [] # list by testing VIDs
    Y_smooth_list = [] # list by testing VIDs
    Y_true_list = []

    # Iterate over testing VIDs
    offset = 0
    for i, cnt in enumerate(test_VIDs_ind_cnts):
        Y_raw_list.append( Y_test_pred_rad[offset:offset+cnt] )
        Y_true_list.append( Y_test_true_rad[offset:offset+cnt] )
        offset += cnt

    ######################################
    # Apply the specified smoother; for each VID
    for Y_raw in Y_raw_list: # iterate over testing VIDs
        Y_smooth = post_smooth(Y_raw, smoother)
        Y_smooth_list.append( Y_smooth )
        
    ######################################
    # Evaluate both: raw Y and smoothed Y
    print "Smoothed Y evaluation ..."
    results_smooth = eval_test2(X_test, Y_test_true_rad, np.concatenate(Y_smooth_list, axis=0), 
                               model_type, seg_len, test_VIDs, test_VIDs_ind_cnts)
    
    # Save results
    np.savez(filename + '.npz', 
             X_test=X_test, 
             Y_true_list=Y_true_list, Y_raw_list=Y_raw_list, Y_smooth_list=Y_smooth_list, 
             test_VIDs=test_VIDs, SD_offsets=SD_offsets, 
             results_smooth=results_smooth, 
             N_params=N_params, N_epochs=N_epochs, test_loss=test_loss)

    print "Saved (raw & smoothed predictions with results) to:", filename
    
###############################################################################################################
# Save true Y, raw and smoothed predicted Y to .npz: input in 0-1 scale; 
# angles saved in RADIANS (for generation on robot); 
# No evaluation
def save_predictions_no_eval(filename, X_test, Y_test_true, Y_test_pred, model_type, seg_len, test_VIDs, test_VIDs_ind_cnts, 
                    SD_offsets=None, smoother='LPBF_4', N_params=None, N_epochs=None):
    '''
    smoother: 'KS'     => Kalman Smoother with EM algo; 
              'LPBF_4' => low-pass Butterworth filter with cut-off frequency 4 Hz
    '''
    N_targets = Y_test_true.shape[1]
    
    # Record overall loss (as optimsed): in 0-1 range
    test_loss = mean_squared_error(Y_test_true, Y_test_pred)
    
    # Convert to radians
    Y_test_pred_rad = inv_norm_Y(Y_test_pred)
    Y_test_true_rad = inv_norm_Y(Y_test_true)

    Y_raw_list = [] # list by testing VIDs
    Y_smooth_list = [] # list by testing VIDs
    Y_true_list = []

    # Iterate over testing VIDs
    offset = 0
    for i, cnt in enumerate(test_VIDs_ind_cnts):
        Y_raw_list.append( Y_test_pred_rad[offset:offset+cnt] )
        Y_true_list.append( Y_test_true_rad[offset:offset+cnt] )
        offset += cnt

    ######################################
    # Apply the specified smoother; for each VID
    for Y_raw in Y_raw_list: # iterate over testing VIDs
        Y_smooth = post_smooth(Y_raw, smoother)
        Y_smooth_list.append( Y_smooth )
        
    # Save results
    np.savez(filename + '.npz', 
             X_test=X_test, 
             Y_true_list=Y_true_list, Y_raw_list=Y_raw_list, Y_smooth_list=Y_smooth_list, 
             test_VIDs=test_VIDs, SD_offsets=SD_offsets, 
             N_params=N_params, N_epochs=N_epochs, test_loss=test_loss)

    print "Saved (raw & smoothed predictions WITHOUT results) to:", filename

###############################################################################################################
# Save true Y, raw and filtered predicted Y to .npz: input in 0-1 scale, convert to RADIANS for generation on robot;
# Also do some evaluation
# This is for one video (one sequence); 
# OFFLINE TTS prediction => perform smoothing
def save_predictions_and_eval_wo_truth_TTS(filename, X, Y_raw, seg_len, smoother='LPBF_4'):
    
    N_targets = Y_raw.shape[1]
    
    # Convert to radians
    Y_raw_rad = inv_norm_Y(Y_raw)
    
    ######################################
    # Apply specified smoother;
    Y_smooth_rad = post_smooth(Y_raw_rad, smoother)

    ######################################
    # Evaluate both: raw Y and smoothed Y
    print "Raw Y evaluation ..."
    results_raw = eval_wo_truth(X, Y_raw_rad, seg_len)
    print "Smoothed Y evaluation ..."
    results_smooth = eval_wo_truth(X, Y_smooth_rad, seg_len)
    
    # Save results
    np.savez(filename + '.npz', 
             X=X, Y_raw=Y_raw_rad, Y_smooth=Y_smooth_rad, 
             results_raw=results_raw, results_smooth=results_smooth)

    print "Saved (raw & smoothed predictions with results) to:", filename
    
    
###############################################################################################################
# Save true Y, raw and filtered predicted Y to .npz: input in RADIANS; for generation on robot;
# Also do some evaluation
# This is for one video (one sequence); 
# ONLINE prediction
# If Y_smooth provided (by Kalman filter) => don't smooth
# If Y_smooth == []                       => smooth
def save_predictions_and_eval_wo_truth_ONLINE(filename, X, Y_raw, Y_smooth, seg_len, smoother='LPBF_4'):
    
    N_targets = Y_raw.shape[1]
    
    # If smooth predictions are not provided, apply smoother
    if len(Y_smooth) == 0:
        ######################################
        # Apply specified smoother;
        Y_smooth = post_smooth(Y_raw, smoother)

    ######################################
    # Evaluate both: raw Y and filtered/smoothed Y
    print "Raw Y evaluation ..."
    results_raw = eval_wo_truth(X, Y_raw, seg_len)
    print "Smoothed Y evaluation ..."
    results_smooth = eval_wo_truth(X, Y_smooth, seg_len)
    
    # Save results
    np.savez(filename + '.npz', 
             X=X, Y_raw=Y_raw, Y_smooth=Y_smooth, 
             results_raw=results_raw, results_smooth=results_smooth)

    print "Saved (raw & smoothed predictions with results) to:", filename


