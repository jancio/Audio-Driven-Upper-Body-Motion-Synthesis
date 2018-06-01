#####################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#####################################################################################
# Utility functions for evaluation (various metrics) and plotting of results
#####################################################################################


import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.cross_decomposition import CCA
from geoutils import radToDeg, degToRad
from sklearn.metrics import mean_squared_error
import scipy.io.wavfile as wav

angles_names = [
    "HeadPitch", "HeadYaw", 
    "LShoulderRoll", "LShoulderPitch", "LElbowRoll", "LElbowYaw",
    "RShoulderRoll", "RShoulderPitch", "RElbowRoll", "RElbowYaw", 
    "HipRoll", "HipPitch"
]
# http://doc.aldebaran.com/2-4/family/pepper_technical/joints_pep.html
robot_angle_constraints_deg = np.array([
    [-40.5, -119.5,  0.5, -119.5, -89.5, -119.5, -89.5, -119.5,   0.5, -119.5, -29.5, -59.5], # minima of (N_JOINT_ANGLES) of joint angles
    [ 36.5,  119.5, 89.5,  119.5,  -0.5,  119.5,  -0.5,  119.5,  89.5,  119.5,  29.5,  59.5]  # maxima
])
robot_angle_constraints_rad = degToRad(robot_angle_constraints_deg)

############################################################################
# Scale pose features to range 0-1 scale; each angle independently
def norm_Y(a):
    N_targets = a.shape[1]
    return (a - robot_angle_constraints_rad[0,:N_targets]) / (robot_angle_constraints_rad[1,:N_targets] - robot_angle_constraints_rad[0,:N_targets])
# Scale pose features back to radians (ORG), from 0-1 scale
def inv_norm_Y(a):
    N_targets = a.shape[1]
    return a * (robot_angle_constraints_rad[1,:N_targets] - robot_angle_constraints_rad[0,:N_targets]) + robot_angle_constraints_rad[0,:N_targets] 
# For single vector: special for ONLINE prediction
def inv_norm_Y_vec(a):
    N_targets = len(a)
    return a * (robot_angle_constraints_rad[1,:N_targets] - robot_angle_constraints_rad[0,:N_targets]) + robot_angle_constraints_rad[0,:N_targets] 
    
#######################################################################################################
# Canonical Correlation Analysis
# https://stackoverflow.com/questions/37398856/how-to-get-the-first-canonical-correlation-from-sklearns-cca-module
# checked with: https://github.com/stochasticresearch/depmeas/blob/master/python/rdc.py
# and produces same results
# can also compare with: 
# https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5118469/
# https://github.com/gallantlab/pyrcca
#######################################################################################################

def get_global_cca(Xi, Yi):
    U_c, V_c = CCA(n_components=1).fit_transform(Xi, Yi)
    return np.corrcoef(U_c.T, V_c.T)[0, 1]

# Calculate local CCA over all windows, return list of local CCAs
def get_local_cca(Xi, Yi, window_length):
    ccas = []
    Ni = len(Xi) - window_length + 1
    for i in range(Ni):
        U_c, V_c = CCA(n_components=1).fit_transform(Xi[i:i+window_length], Yi[i:i+window_length])
        ccas.append( np.corrcoef(U_c.T, V_c.T)[0, 1] )
    return ccas

#########################################
# Jerkiness: IN RADIANS
# for one sequence only
# input in RADIANS
# normalised by length of sequence
def calculate_norm_jerk(Y, fr=100.):
    N_targets = Y.shape[1]
    jerkiness = np.zeros(N_targets)
    cnt = Y.shape[0]
    # Calculate jerkiness for each joint angle separately, over all frames
    for angle_type in range(N_targets): 
        fwd_diffs = np.diff( Y[:, angle_type], n=3)
        for j in range(0, cnt - 3):
            jerkiness[angle_type] += (fwd_diffs[j] ** 2)
    jerkiness = 0.5 * jerkiness * (fr**5) / (cnt - 3)
    return jerkiness

#########################################
# Delta Jerkiness: absolute difference betwen TRUE and PREDICTED(smoothed) movements: IN RADIANS
# for one sequence only
# input in RADIANS
# normalised by length of sequence
def calculate_norm_delta_jerk(Y_true, Y_pred, fr=100.):
    N_targets = Y_true.shape[1]
    jerkiness = {}
    jerkiness['true'] = np.zeros(N_targets)
    jerkiness['pred'] = np.zeros(N_targets)
    cnt = Y_true.shape[0]
    # Calculate jerkiness for each joint angle separately, over all frames
    for angle_type in range(N_targets): 
        fwd_diffs = np.diff( Y_true[:, angle_type], n=3)
        for j in range(0, cnt - 3):
            jerkiness['true'][angle_type] += (fwd_diffs[j] ** 2)

        fwd_diffs = np.diff( Y_pred[:, angle_type], n=3)
        for j in range(0, cnt - 3):
            jerkiness['pred'][angle_type] += (fwd_diffs[j] ** 2)

    jerkiness['true'] = 0.5 * jerkiness['true'] * (fr**5) / (cnt - 3)
    jerkiness['pred'] = 0.5 * jerkiness['pred'] * (fr**5) / (cnt - 3) 
    
    return np.abs( jerkiness['true'] - jerkiness['pred'] )

#########################################
# As above but averaged over list of input matrices
def calculate_norm_delta_jerk_onList(Y_true_list, Y_pred_list, fr=100.):
    DELTA_JERK = []
    for Y_true, Y_pred in zip(Y_true_list, Y_pred_list):
        DELTA_JERK.append(
            calculate_norm_delta_jerk(Y_true, Y_pred, fr)
        )
    return np.mean(DELTA_JERK, axis=0)

#######################################################################################################
# Statistical significance test: 
# https://machinelearningmastery.com/use-statistical-significance-tests-interpret-machine-learning-results/
from scipy.stats import normaltest, ttest_ind, ks_2samp

def statistical_significance(data_A, data_B, p_diff, verbose=True):
    
    # Normality test
    v, p = normaltest(data_A)
    if verbose:
        print "Normlaity test: data_A"
        print "\t", v, p
    A_gauss = p >= 0.05
    if A_gauss:
        if verbose:
            print "\t data_A is likely Gaussian"
    else:
        if verbose:
            print "\t data_A is NOT Gaussian with 95% confidence"
        
    # Normality test
    v, p = normaltest(data_B)
    if verbose:
        print "Normlaity test: data_B"
        print "\t", v, p
    B_gauss = p >= 0.05
    if B_gauss:
        if verbose:
            print "\t data_B is likely Gaussian"
    else:
        if verbose:
            print "\t data_B is NOT Gaussian with 95% confidence"
        
    
    if A_gauss and B_gauss:
        if np.std(data_A) == np.std(data_B):
            if verbose:
                print "Student t-test:"
            v, p = ttest_ind(data_A, data_B, equal_var=True)
        else:
            if verbose:
                print "Welch's t-test:"
            v, p = ttest_ind(data_A, data_B, equal_var=False)
            
    else:
        if verbose:
            print "Kolmogorov-Smirnov test:"
        v, p = ks_2samp(data_A, data_B)
        
    print "\t", v, p
    stat_sig_dif = p <= p_diff # are they signific. different?
    if not stat_sig_dif:
        print "\t samples A and B are NOT significantly different"
        #print('Samples are likely drawn from the same distributions (accept H0)')
    else:
        print "\t samples A and B are significantly different"
        #print('Samples are likely drawn from different distributions (reject H0)')
        
    return stat_sig_dif

###########################################################################################################
# Plot stat sig bar graph, with asterisk if stat sig
def stat_sig_plot(dataA, dataB, y_label, data_labels, markHigher, show=True, showErr=True):
    
    meanA = np.mean(dataA)
    meanB = np.mean(dataB)
    stdA = np.std(dataA)
    stdB = np.std(dataB)
    ss = statistical_significance(dataA, dataB, 0.05, False)
    
    plt.figure()
    w = 0.2 # bar width  
    ylim_max = 0.5
    if showErr:
        plt.bar(0.-w/2., meanA, yerr=stdA, width=w, align='center', label=data_labels[0])      # bar A
        plt.bar(0.,      meanB, yerr=stdB, width=w, align='edge', label=data_labels[1])        # bar B
    else:
        plt.bar(0.-w/2., meanA, width=w, align='center', label=data_labels[0])      # bar A
        plt.bar(0.,      meanB, width=w, align='edge', label=data_labels[1])        # bar B
    ax = plt.gca()
    p = ax.patches
    if ss:
        if (meanA > meanB and markHigher) or (meanA <= meanB and not markHigher):
            # mark bar A
            i = 0
        else:
            # mark bar B
            i = 1
        ax.text(p[i].get_x()+w/1.4, p[i].get_height(), '*', fontsize=20)#, color='dimgrey')
    plt.ylabel(y_label) #, fontsize=14)
    #plt.xlabel('Personality trait')
    #plt.xticks(xaxis, personality_dims, rotation=45)
    # plt.ylim(0, 28)
    #plt.xticks([0.-w/2., 0.], ['', ''])
    ax.set_xticks([])
    plt.xlim(-ylim_max, ylim_max)
    plt.legend()
    if show:
        plt.show()

#######################################################################################################
# Plot RMSE by angles for 4 feature sets / N methods to compare : input is Nx11 array
def plot_4RMSE(rmses_byAngles, labels, stds=[], y_max=28.):
    
    N_series = len(labels)
    N_angles = len(rmses_byAngles[0])
    xaxis = np.arange(N_angles)
    w = 0.2 # bar width
#     plt.figure(figsize=(15,10))
    plt.figure()
    for i in range(N_series):
        if len(stds) == 0:
            plt.bar(xaxis - 0.5*N_series*w + i*w, rmses_byAngles[i], width=w, align='edge', label=labels[i])
        else:
            plt.bar(xaxis - 0.5*N_series*w + i*w, rmses_byAngles[i], yerr=stds[i], width=w, align='edge', label=labels[i])
    plt.ylabel('Root mean squared error ($^{\circ}$)') #, fontsize=14)
    plt.xlabel('Joint angle')
    plt.xticks(xaxis, angles_names[:N_angles], rotation=45)
    plt.ylim(0, y_max)
    plt.legend()
    plt.show()
#     plt.bar(xaxis, rmse, width=w,
#             label='Overall RMSE = {:.2f}'.format(rmse_overall) + '$^{\circ}$', 
#            )
    
#######################################################################################################
# Evaluate testing results; Input: Y in radians, X z-normalised  
def eval_test(X_test, Y_test_true, Y_test_pred, model_type, seg_len, test_VIDs, test_VIDs_ind_cnts, verbose=True):
    
    N_targets = Y_test_true.shape[1]
    
    # Convert to degrees
    Y_test_pred = radToDeg( Y_test_pred )
    Y_test_true = radToDeg( Y_test_true )
    
    Y_diff_sq = np.square( Y_test_pred - Y_test_true ) # MSE 2D matrix
    
    #########################################
    # Evaluate RMSE
    mse = mean_squared_error(Y_test_true, Y_test_pred, multioutput='raw_values') # by angles
    rmse = np.sqrt( mse )
    rmse_overall = np.sqrt( np.mean(mse) )
    
    # Calculated according to: https://www.itl.nist.gov/div898/handbook/mpc/section5/mpc552.htm
    mse_stds = np.std( Y_diff_sq, axis=0 ) # stds by angles
    rmse_stds = np.sqrt( mse_stds )                         # simple method
    rmse_overall_std = np.sqrt( np.std(Y_diff_sq) )

    #########################################
    # Evaluate CCA (globally and locally)
    cca_global = {}
    cca_global['XYt']  = []
    cca_global['XYp']  = []
    cca_global['YtYp'] = []
    cca_local = {}
    cca_local['XYt']  = []
    cca_local['XYp']  = []
    cca_local['YtYp'] = []
    offset = 0
    # Iterate over testing VIDs
    for i, cnt in enumerate(test_VIDs_ind_cnts):
        
        cca_global['XYt'].append(  get_global_cca(X_test[offset:offset+cnt], Y_test_true[offset:offset+cnt]) )
        cca_global['XYp'].append(  get_global_cca(X_test[offset:offset+cnt], Y_test_pred[offset:offset+cnt]) )
        cca_global['YtYp'].append( get_global_cca(Y_test_true[offset:offset+cnt], Y_test_pred[offset:offset+cnt]) )
        
        cca_local['XYt'].extend(  get_local_cca(X_test[offset:offset+cnt], Y_test_true[offset:offset+cnt], seg_len) )
        cca_local['XYp'].extend(  get_local_cca(X_test[offset:offset+cnt], Y_test_pred[offset:offset+cnt], seg_len) )
        cca_local['YtYp'].extend( get_local_cca(Y_test_true[offset:offset+cnt], Y_test_pred[offset:offset+cnt], seg_len) )    
                                
        offset += cnt
        #print test_VIDs[i], np.mean
                                
    # Calculate CCA means over testing VIDs, stds only for local CCA since for global there are only few testing sequences
    cca_global['XYt']  = np.mean( cca_global['XYt'] )
    cca_global['XYp']  = np.mean( cca_global['XYp'] )
    cca_global['YtYp'] = np.mean( cca_global['YtYp'] )
                                
    cca_local['XYt']  = [np.mean( cca_local['XYt'] ),  np.std( cca_local['XYt'] )]
    cca_local['XYp']  = [np.mean( cca_local['XYp'] ),  np.std( cca_local['XYp'] )]
    cca_local['YtYp'] = [np.mean( cca_local['YtYp'] ), np.std( cca_local['YtYp'] )]
            
    #########################################
    # Jerkiness: of TRUE and PREDICTED movements: IN RADIANS
    jerkiness = {}
    jerkiness['true'] = np.zeros(N_targets)
    jerkiness['pred'] = np.zeros(N_targets)
    FPS = 100.
    offset = 0
    # Iterate over testing VIDs
    for i, cnt in enumerate(test_VIDs_ind_cnts):
        
        # Calculate jerkiness for each joint angle separately, over all frames
        for angle_type in range(N_targets): 
            fwd_diffs = np.diff( degToRad(Y_test_true[offset:offset+cnt, angle_type]) , n=3)
            for j in range(0, cnt - 3):
                jerkiness['true'][angle_type] += (fwd_diffs[j] ** 2)
                
            fwd_diffs = np.diff( degToRad(Y_test_pred[offset:offset+cnt, angle_type]) , n=3)
            for j in range(0, cnt - 3):
                jerkiness['pred'][angle_type] += (fwd_diffs[j] ** 2)
            
        offset += cnt
        
    jerkiness['true'] = 0.5 * jerkiness['true'] / FPS  
    jerkiness['pred'] = 0.5 * jerkiness['pred'] / FPS  
    
    #########################################
    # Show these results
    results = ([rmse, rmse_stds], [rmse_overall, rmse_overall_std], cca_global, cca_local, jerkiness)
    show_test_results(results)
 
    return results

#######################################################################################################
# As above but fixed JERK calculation and without global CCA
def eval_test2(X_test, Y_test_true, Y_test_pred, model_type, seg_len, test_VIDs, test_VIDs_ind_cnts, verbose=True):
    
    N_targets = Y_test_true.shape[1]
    
    # Convert to degrees
    Y_test_pred = radToDeg( Y_test_pred )
    Y_test_true = radToDeg( Y_test_true )
    
    Y_diff_sq = np.square( Y_test_pred - Y_test_true ) # MSE 2D matrix
    
    #########################################
    # Evaluate RMSE
    mse = mean_squared_error(Y_test_true, Y_test_pred, multioutput='raw_values') # by angles
    rmse = np.sqrt( mse )
    rmse_overall = np.sqrt( np.mean(mse) )
    
    # Calculated according to: https://www.itl.nist.gov/div898/handbook/mpc/section5/mpc552.htm
    mse_stds = np.std( Y_diff_sq, axis=0 ) # stds by angles
    rmse_stds = np.sqrt( mse_stds )                         # simple method
    rmse_overall_std = np.sqrt( np.std(Y_diff_sq) )

    #########################################
    # Evaluate CCA (globally and locally)
    cca_local = {}
    cca_local['XYt']  = []
    cca_local['XYp']  = []
    cca_local['YtYp'] = []
    offset = 0
    # Iterate over testing VIDs
    for i, cnt in enumerate(test_VIDs_ind_cnts):

        cca_local['XYt'].extend(  get_local_cca(X_test[offset:offset+cnt], Y_test_true[offset:offset+cnt], seg_len) )
        cca_local['XYp'].extend(  get_local_cca(X_test[offset:offset+cnt], Y_test_pred[offset:offset+cnt], seg_len) )
        cca_local['YtYp'].extend( get_local_cca(Y_test_true[offset:offset+cnt], Y_test_pred[offset:offset+cnt], seg_len) )    
                                
        offset += cnt
        #print test_VIDs[i], np.mean
                                
    # Calculate CCA means and stds over testing VIDs            
    cca_local['XYt']  = [np.mean( cca_local['XYt'] ),  np.std( cca_local['XYt'] )]
    cca_local['XYp']  = [np.mean( cca_local['XYp'] ),  np.std( cca_local['XYp'] )]
    cca_local['YtYp'] = [np.mean( cca_local['YtYp'] ), np.std( cca_local['YtYp'] )]
            
    #########################################
    # Jerkiness: of TRUE and PREDICTED movements: IN RADIANS
    jerkiness = {}
    jerkiness['true'] = np.zeros(N_targets)
    jerkiness['pred'] = np.zeros(N_targets)
    
    FPS = 100.
    offset = 0
    # Iterate over testing VIDs
    for i, cnt in enumerate(test_VIDs_ind_cnts):
        
        # Calculate jerkiness for true & predicted sequence of all angles
        jerkiness['true'] += calculate_norm_jerk(Y_test_true)
        jerkiness['pred'] += calculate_norm_jerk(Y_test_pred)
        
        offset += cnt
        
    jerkiness['delta'] = np.abs( jerkiness['true'] - jerkiness['pred'] )
    jerkiness['delta'] = jerkiness['delta'] / len(test_VIDs_ind_cnts)  # average over all VIDs
    
    jerkiness['true'] = jerkiness['true'] / len(test_VIDs_ind_cnts)  # average over all VIDs
    jerkiness['pred'] = jerkiness['pred'] / len(test_VIDs_ind_cnts)  # average over all VIDs 
    
    #########################################
    # Show these results
    results = ([rmse, rmse_stds], [rmse_overall, rmse_overall_std], None, cca_local, jerkiness)
    show_test_results(results)
 
    return results
    
#######################################################################################################
# Show testing results;
def show_test_results(data_in):
    
    [rmse, rmse_stds], [rmse_overall, rmse_overall_std], cca_global, cca_local, jerkiness = data_in
    
    N_targets = len(rmse)
    
    # RMSE
    print "RMSE by angles: ", rmse, " (deg)"
    print "\t- stds:", rmse_stds, " (deg)"
    print "RMSE overall: ", rmse_overall, " (deg)"
    print "\t- std:", rmse_overall_std, " (deg)"

    # With errorbars
    plt.figure() #plt.figure(figsize=(12,8))
    xaxis = np.arange(N_targets)
    w = 0.3 # bar width
    plt.bar(xaxis, rmse, width=w, yerr=rmse_stds, 
            label='Overall RMSE = {:.2f} $\pm$ {:.2f}'.format(rmse_overall, rmse_overall_std) + '$^{\circ}$', 
           )
    plt.xticks(xaxis, angles_names[:N_targets], rotation=45)
    plt.xlabel('Joint angle')
    plt.ylabel('Root mean squared error ($^{\circ}$)')
    plt.legend()
    plt.show()

    # Without errorbars
    plt.figure() #plt.figure(figsize=(12,8))
    xaxis = np.arange(N_targets)
    w = 0.3 # bar width
    plt.bar(xaxis, rmse, width=w,
            label='Overall RMSE = {:.2f}'.format(rmse_overall) + '$^{\circ}$', 
           )
    plt.xticks(xaxis, angles_names[:N_targets], rotation=45)
    plt.xlabel('Joint angle')
    plt.ylabel('Root mean squared error ($^{\circ}$)')
    plt.legend()
    plt.show()
    
    #########################################
    # CCA (globally and locally)
    if cca_global != None:
        print "Global CCAs:"
        for k,v in cca_global.items():
            print "\t", k, ": ", v
    print "Local CCAs:"
    for k,v in cca_local.items():
        print "\t", k, ": ", v[0], " +/- ", v[1]
            
    #########################################
    # Jerkiness: of TRUE and PREDICTED movements: IN RADIANS
    
    print "Jerkiness:"
    print "\t(true) by angles: ", jerkiness['true']
    print "\t(true) summed:    ", np.sum(jerkiness['true'])
    print "\t(pred) by angles: ", jerkiness['pred']
    print "\t(pred) summed:    ", np.sum(jerkiness['pred'])

    plt.figure() #plt.figure(figsize=(12,8))
    xaxis = np.arange(N_targets + 2)
    w = 0.3 # bar width
    plt.bar(xaxis-w/2, jerkiness['true'].tolist() + [0., np.sum(jerkiness['true'])], 
            width=w, label='Ground truth movement.')
    plt.bar(xaxis, jerkiness['pred'].tolist() + [0., np.sum(jerkiness['pred'])], 
            width=w, label='Predicted movement.', align='edge')
    plt.axvline(x=N_targets, color='k', linestyle='--')
    plt.xticks(xaxis, angles_names[:N_targets] + ['', 'Overall'], rotation=45)
    plt.xlabel('Joint angle')
    plt.ylabel('Jerkiness ($rad^{2}s^{-5}$)')
    plt.legend()
    plt.show()

    # Log scale
    plt.figure() #plt.figure(figsize=(12,8))
    xaxis = np.arange(N_targets + 2)
    w = 0.3 # bar width
    plt.bar(xaxis-w/2, jerkiness['true'].tolist() + [0., np.sum(jerkiness['true'])], 
            width=w, label='Ground truth movement.', bottom=0.001)
    plt.bar(xaxis, jerkiness['pred'].tolist() + [0., np.sum(jerkiness['pred'])], 
            width=w, label='Predicted movement.', align='edge', bottom=0.001)
    plt.axvline(x=N_targets, color='k', linestyle='--')
    plt.xticks(xaxis, angles_names[:N_targets] + ['', 'Overall'], rotation=45)
    plt.xlabel('Joint angle')
    plt.ylabel('Jerkiness ($rad^{2}s^{-5}$)')
    plt.yscale('log')
    plt.legend()
    plt.show()
    
    
#######################################################################################################
# Plot predictions against ground truths and audio (Plot audio along with true and predicted pose angle)
# Input in radians.

def plot_predictions(Y_test_true, Y_test_pred, Y_smooth, model_type, angles_to_show, 
                     plot_start, plot_length, input_mode, SD_offset, 
                     test_VID, customAudioPath=None):  
    '''
    Y_test_true, Y_test_pred, Y_smooth: (N_datapoints, N_targets) matrices of ground truths and (raw/smooth) predictions
    model_type: MLP_SD / MLP_SI / LSTM_SD / LSTM_SI
    angles_to_show: all / head / hands / head+hip / hands+hip
    input_mode: input mode whether (plot_start, plot_length) are in time / #samples
    SD_offset: sample at which the test set begins in the testing VID sequence, ONLY for subject dependent case, otherwise None
    '''
    
    print test_VID

    # If subject-dependent model, Y_test_* contains only testing part => need to offset
    offset = 0
    if model_type[-1] == 'D' and SD_offset != None: 
        offset = offset - SD_offset
    elif model_type[-1] != 'D' and SD_offset == None: 
        pass
    else:
        raise ValueError("Check SD_offset and model_type aggreement!")
    
    # Convert to degrees (for plotting)
    Y_test_pred = radToDeg( Y_test_pred )
    if len(Y_test_true) != 0:
        Y_test_true = radToDeg( Y_test_true )
    if len(Y_smooth) != 0:
        Y_smooth = radToDeg( Y_smooth )
    
    N_targets = Y_test_pred.shape[1]
    
    FPS = 100. # rate of frames within segment
    if customAudioPath == None:
        (audio_rate, audio_sig) = wav.read('./../Dataset/AudioWav_16kHz/' + test_VID + '.wav')
    else:
        (audio_rate, audio_sig) = wav.read(customAudioPath)
        
    # Input in seconds
    if input_mode == 'time':
        start_time = plot_start
        end_time = plot_start + plot_length
        
        start_angle_sample = int( start_time / (1./FPS))
        end_angle_sample   = int( end_time / (1./FPS))
        N_angle_samples    = end_angle_sample - start_angle_sample  
        
    # Input in number of (angle) samples (at rate 100Hz)
    elif input_mode == 'samples':
        start_angle_sample = plot_start
        end_angle_sample   = plot_start + plot_length
        N_angle_samples    = plot_length
    
    else:
        raise ValueError("Invalid in_mode!")        
        
    start_audio_sample = int( start_angle_sample*(1./FPS) / (1./audio_rate))
    N_audio_samples = int( N_angle_samples*(1./FPS) / (1./audio_rate))
    end_audio_sample = start_audio_sample + N_audio_samples
    
    # Check if given start and length of the segment to show are within bounds of this video
    if end_audio_sample >= len(audio_sig):
        raise ValueError("Requested segment to plot is out of bounds!")
    
    # Time axis for angles
    angle_x = np.linspace(start_angle_sample*(1./FPS), end_angle_sample*(1./FPS), N_angle_samples, endpoint=False)
    # Time axis for audio
    audio_x = np.linspace(start_angle_sample*(1./FPS), end_angle_sample*(1./FPS), N_audio_samples, endpoint=False)

    # print start_angle_sample, end_angle_sample, N_angle_samples
    # print start_audio_sample, end_audio_sample, N_audio_samples, len(angle_x), len(audio_x)
    # print offset+start_angle_sample, offset+end_angle_sample
    # print Y_test_true.shape

    # What angles to plot
    if angles_to_show == 'head':
        ind_angles_to_show = np.array([0, 1])
    elif angles_to_show == 'hands':
        ind_angles_to_show = np.array([2, 3, 4, 5, 6, 7, 8, 9])
    elif angles_to_show == 'head+hip':
        ind_angles_to_show = np.array([0, 1, 10])
    elif angles_to_show == 'hands+hip':
        ind_angles_to_show = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10])
    else: # plot all
        ind_angles_to_show = np.arange(N_targets)
        
    for iA in ind_angles_to_show:
        plt.figure(figsize=(15,7)) # 15,80 # plt.figure() # ax = plt.subplot(N_targets, 1, iA+1)
        ax = plt.gca()
        axA = ax.twinx() # audio x-axis
        axA.fill_between(audio_x, 
                         #np.negative(audio_sig[start_audio_sample:end_audio_sample]), 
                         audio_sig[start_audio_sample:end_audio_sample], 
                         'x-', color='gray', facecolor='gray', alpha=0.4, label='Audio waveform')      # audio
        # axA.plot(audio_x, audio_sig[start_audio_sample:end_audio_sample], 
        #                  'x-', color='gray', alpha=0.4, label='Audio waveform')      # audio
        if len(Y_test_true) != 0:
            ax.plot(angle_x, Y_test_true[offset+start_angle_sample:offset+end_angle_sample, iA], 
                'bx-', label='True angle')       # true
        ax.plot(angle_x, Y_test_pred[offset+start_angle_sample:offset+end_angle_sample, iA], 
                'rx-', label='Predicted angle')  # predicted
        if len(Y_smooth) != 0:
            if model_type == 'ONLINE':
                ax.plot(angle_x, Y_smooth[offset+start_angle_sample:offset+end_angle_sample, iA], 
                    'gx-', label='Predicted filtered angle')  # predicted & filtered
            else:
                ax.plot(angle_x, Y_smooth[offset+start_angle_sample:offset+end_angle_sample, iA], 
                    'gx-', label='Predicted smoothed angle')  # predicted & smoothed
        
        ax.set_ylabel(angles_names[iA] + r' angle $\theta_3$ ($^{\circ}$)')#, color='blue')
        ax.set_xlabel("Time (s)")

        ln, lb =   ax.get_legend_handles_labels()
        lnA, lbA = axA.get_legend_handles_labels()
        axA.legend(ln + lnA, lb + lbA)

        axA.set_axis_off()

        #plt.title(angles_names[iA])
        plt.show()
        
#######################################################################################################
# Evaluate when tested on data without ground truth: local CCA, jerkiness & audio-predictedY plots
# Input in radians
def eval_wo_truth(X_test, Y, seg_len):
    
    N_targets = Y.shape[1]
    
    # Convert to degrees
    Y_test_pred = radToDeg( Y )
    
    #########################################
    # Evaluate CCA (globally and locally)
    cca_global = {}
    cca_global['XYp']  = get_global_cca(X_test, Y_test_pred)
    cca_local = {}
    cca_local['XYp']  = get_local_cca(X_test, Y_test_pred, seg_len)
    
    #########################################
    # Jerkiness: of PREDICTED movements: IN RADIANS
    jerkiness = {}
    jerkiness['pred'] = np.zeros(N_targets)
    FPS = 100.

    # Calculate jerkiness for each joint angle separately, over all frames
    for angle_type in range(N_targets): 
        fwd_diffs = np.diff( degToRad(Y_test_pred[:, angle_type]) , n=3)
        for fwd_diff in fwd_diffs:
            jerkiness['pred'][angle_type] += (fwd_diff ** 2)

    jerkiness['pred'] = 0.5 * jerkiness['pred'] / FPS  
    
    #########################################
    # Show these results
    results = (cca_global, cca_local, jerkiness)
    show_results_wo_truth(results)
 
    return results

#######################################################################################################
# Show results computed by eval_wo_truth
def show_results_wo_truth(data_in):
    
    cca_global, cca_local, jerkiness = data_in
    
    N_targets = 11
    
    print "Global CCAs:"
    for k,v in cca_global.items():
        print "\t", k, ": ", v
    print "Local CCAs:"
    for k,v in cca_local.items():
        print "\t", k, ": ", v[0], " +/- ", v[1]
    
    print "Jerkiness:"
    print "\t(pred) by angles: ", jerkiness['pred']
    print "\t(pred) summed:    ", np.sum(jerkiness['pred'])

    plt.figure() #plt.figure(figsize=(12,8))
    xaxis = np.arange(N_targets + 2)
    w = 0.3 # bar width
    plt.bar(xaxis, jerkiness['pred'].tolist() + [0., np.sum(jerkiness['pred'])], 
            width=w, label='Predicted movement.')
    plt.axvline(x=N_targets, color='k', linestyle='--')
    plt.xticks(xaxis, angles_names[:N_targets] + ['', 'Overall'], rotation=45)
    plt.xlabel('Joint angle')
    plt.ylabel('Jerkiness ($rad^{2}s^{-5}$)')
    plt.legend()
    plt.show()

    # Log scale
    plt.figure() #plt.figure(figsize=(12,8))
    xaxis = np.arange(N_targets + 2)
    w = 0.3 # bar width
    plt.bar(xaxis, jerkiness['pred'].tolist() + [0., np.sum(jerkiness['pred'])], 
            width=w, label='Predicted movement.', bottom=0.001)
    plt.axvline(x=N_targets, color='k', linestyle='--')
    plt.xticks(xaxis, angles_names[:N_targets] + ['', 'Overall'], rotation=45)
    plt.xlabel('Joint angle')
    plt.ylabel('Jerkiness ($rad^{2}s^{-5}$)')
    plt.yscale('log')
    plt.legend()
    plt.show()

#######################################################################################################
# Plot training history: aimed for LSTM models to show loss curves
# Input: dictionary with keys: loss, val_loss
def plot_training_history(h):
    plt.figure()
    ep = np.arange(len(h['loss'])) + 1
    plt.plot(ep, h['loss'], label='Training')
    plt.plot(ep, h['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Mean squared error loss (scaled)')
    plt.legend()
    plt.show()
    print "Train/val loss = MSE when Target angles were scaled to range 0-1"
    print "Total epochs: ", len(h['loss'])
    print "Min loss:    ", np.min(h['loss']), " at epoch", np.argmin(h['loss'])+1
    print "Min val_loss:", np.min(h['val_loss']), " at epoch", np.argmin(h['val_loss'])+1
    
    