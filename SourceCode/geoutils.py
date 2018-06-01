#####################################################################################
# Audio-driven upper-body motion synthesis on a humanoid robot
# Computer Science Tripos Part III Project
# Jan Ondras (jo356@cam.ac.uk), Trinity College, University of Cambridge
# 2017/18
#####################################################################################
# Utility functions for:
# - angle conversions 
# - calculation of 11 upper-body joint angles from the extracted 3D joint positions
#####################################################################################

import numpy as np
import time
from geo import *

##############################################################################################################
# Angle radian <-> degree conversion

# By default all angles in radians
def radToDeg(angle):
    return 180. * angle / np.pi

def degToRad(angle):
    return np.pi * angle / 180.

##############################################################################################################
# Calculate 11 upper-body joint angles from the extracted 3D joint positions
# For OpenPose based methods (OP+M, OP+C), nose joint is not present => can only estimate head yaw

def xyz_to_angles(HT, NO, NE, PE, LS, LE, LW, RS, RE, RW):
    '''
    Joint positions (x,y,z) as Point objects in the ordering:
        HeadTop, Nose, Neck, Pelvis, 
        LShoulder, LElbow, LWrist, 
        RShoulder, RElbow, RWrist
        
    Auxiliary geometric entities --------------------- can complete the list below
        p_... = plane
        l_... = line
        P_... = point
        
    Points:
        P_PHT
        P_PNO
        
    Lines:
        l_NE_PE
        l_NE_HT
        l_NE_ChestNormal
        l_NE_PHT
        l_iHead
    
    Planes:
        p_Chest
        p_NeckH
        p_NeckF
        p_NeckS
        p_Head
        
    '''
    #st = time.time()
    
    ####################################################################
    # HeadPitch (up/down); positive = down
    
    p_Chest = Plane(PE, LS, RS) # facing plane of chest, normal forward (from person to camera)
    
    l_NE_PE = Line(NE, PE)
    l_NE_HT = Line(NE, HT)
    p_NeckH = Plane(NE, l_NE_PE) # horizontal plane passing thru NE, normal downwards to PE
    
    l_NE_ChestNormal = p_Chest.normal().projected_on(p_NeckH) # chest normal projected on plane p_NeckH
    p_NeckF = Plane(NE, l_NE_ChestNormal) # facing plane passing thru neck, perpendicular to p_NeckH, normal forward, not collinear with chestNormal
    
    HeadPitch = l_NE_PE.angle_to(l_NE_HT) * p_NeckF.orientation(HT) # positive if bow forward
    
    ####################################################################
    # HeadYaw (left/right); positive = left
    
    p_NeckS = Plane(NE, Point(p_NeckH.normal().r2), Point(p_NeckF.normal().r2)) # side plane, perpendicular to p_NeckH and p_NeckF, normal to right shoulder
    
    if NO == None: # Nose joint is not available, neglect roll rotation of head
        P_PHT = HT.projected_on(p_NeckH) # project onto horizontal neck plane
        l_NE_PHT = Line(NE, P_PHT)
        
        if p_NeckF.orientation(P_PHT) > 0: # is in front
            HeadYaw = - l_NE_PHT.angle_to(p_NeckS) * p_NeckS.orientation(P_PHT) # positive towards left shoulder
        else:
            HeadYaw = - (np.pi - l_NE_PHT.angle_to(p_NeckS)) * p_NeckS.orientation(P_PHT)
        
    else: # Nose is available, can calculate more accurately, w/o neglecting roll rotation of head
        p_Head = Plane(NE, NO, HT)
        l_iHead = p_Head.intersection(p_NeckH) # line from intersection of head plane and neck horizontal plane
        P_PNO = NO.projected_on(l_iHead) # nose projected on this line
        
        if p_NeckF.orientation(P_PNO) > 0: # is in front
            HeadYaw = - l_iHead.angle_to(p_NeckS) * p_NeckS.orientation(P_PNO) # positive towards left shoulder
        else:
            HeadYaw = - (np.pi - l_iHead.angle_to(p_NeckS)) * p_NeckS.orientation(P_PNO)
    
    ####################################################################
    # LShoulderRoll (left/right); positive = left, away from body

    l_LS_RS = Line(LS, RS) # connects shoulders
    P_PPE = PE.projected_on(l_LS_RS) # projection of Pelvis onto shoulders connector
    l_PPE_PE = Line(P_PPE, PE)
    p_ShouldersH = Plane(P_PPE, l_PPE_PE) # horizontal plane passign thru shoulders, perpendicular to p_Chest, normal downwards
    
    p_LShoulder = Plane(LS, l_LS_RS) # vertical side plane, thru LS, perpendicular to p_ShouldersH, normal to RS
    l_LS_LE = Line(LS, LE)
    
    if p_ShouldersH.orientation(LE) > 0: # elbow below shoulder
        LShoulderRoll = - p_LShoulder.angle_to(l_LS_LE) * p_LShoulder.orientation(LE)
    else:
        LShoulderRoll = np.pi + p_LShoulder.angle_to(l_LS_LE) * p_LShoulder.orientation(LE)
        
    ####################################################################
    # LShoulderPitch (up/down); positive = down, neutral position is hands forward
    
    if p_Chest.orientation(LE) > 0: # elbow in front of the chest plane
        LShoulderPitch = p_ShouldersH.angle_to(l_LS_LE) * p_ShouldersH.orientation(LE)
    else:
        LShoulderPitch = (np.pi - p_ShouldersH.angle_to(l_LS_LE)) * p_ShouldersH.orientation(LE)
    
    ####################################################################
    # LElbowRoll negative only
    
    l_LE_LS = Line(LE, LS)
    l_LE_LW = Line(LE, LW)
    p_LElbowP = Plane(LE, l_LE_LS) # plane perpendicular to limb elbow-shoulder, passing thru LE, normal to LS
    
    if p_LElbowP.orientation(LW) > 0: # acute angle LW-LE-LS
        LElbowRoll = - p_LElbowP.angle_to(l_LE_LW) - (np.pi/2)
    else:
        LElbowRoll = p_LElbowP.angle_to(l_LE_LW) - (np.pi/2)

    ####################################################################
    # LElbowYaw positive = rotate to body center
    
    P_LECHN = Point(LE.r + p_Chest.n) # endpoint of chest normal moved to LE joint
    p_LElbowN = Plane(LE, LS, P_LECHN) # plane perpendicular to p_Chest, passign thru LS, LE, normal away from body center

    P_LEENN = Point(LE.r + p_LElbowN.n) # endpoint of p_LElbowN's normal, equivalent to Point(p_LElbowN.normal().r2)
    p_LElbow = Plane(LE, LS, P_LEENN) # plane perpendicular to p_LElbowN, containing limb LE-LS, normal backwards - away from camera

    p_LArm = Plane(LE, LW, LS) # plane of all 3 left-hand joints, normal towards RS if elbow bent and LElbowYaw=-pi/2
    
    if p_LElbowN.orientation(LW) > 0: # is on the further side - away from body center
        LElbowYaw = (np.pi - p_LElbow.angle_to(p_LArm)) * p_LElbow.orientation(LW)
    else:
        LElbowYaw = p_LElbow.angle_to(p_LArm) * p_LElbow.orientation(LW)
            
    
    ####################################################################
    # RShoulderRoll (left/right); negative = right, away from body
    
    l_RS_LS = Line(RS, LS)
    p_RShoulder = Plane(RS, l_RS_LS) # vertical side plane, thru RS, perpendicular to p_ShouldersH, normal to LS
    l_RS_RE = Line(RS, RE)
    
    if p_ShouldersH.orientation(RE) > 0: # elbow below shoulder
        RShoulderRoll = p_RShoulder.angle_to(l_RS_RE) * p_RShoulder.orientation(RE)
    else:
        RShoulderRoll = - np.pi - p_RShoulder.angle_to(l_RS_RE) * p_RShoulder.orientation(RE)
        
    ####################################################################
    # RShoulderPitch (up/down); positive = down
    
    if p_Chest.orientation(RE) > 0: # elbow in front of the chest plane
        RShoulderPitch = p_ShouldersH.angle_to(l_RS_RE) * p_ShouldersH.orientation(RE)
    else:
        RShoulderPitch = (np.pi - p_ShouldersH.angle_to(l_RS_RE)) * p_ShouldersH.orientation(RE)
    
    ####################################################################
    # RElbowRoll positive only
    
    l_RE_RS = Line(RE, RS)
    l_RE_RW = Line(RE, RW)
    p_RElbowP = Plane(RE, l_RE_RS) # plane perpendicular to limb elbow-shoulder, passing thru RE, normal to RS
    
    if p_RElbowP.orientation(RW) > 0: # 
        RElbowRoll = (np.pi/2) + p_RElbowP.angle_to(l_RE_RW) 
    else:
        RElbowRoll = (np.pi/2) - p_RElbowP.angle_to(l_RE_RW)
    
    ####################################################################
    # RElbowYaw positive = rotate away from body center
    
    P_RECHN = Point(RE.r + p_Chest.n) # endpoint of chest normal moved to RE joint
    p_RElbowN = Plane(RE, RS, P_RECHN) # plane perpendicular to p_Chest, passign thru RS, RE, normal towards body center
    
    P_REENN = Point(RE.r + p_RElbowN.n) # endpoint of p_RElbowN's normal, equivalent to Point(p_RElbowN.normal().r2)
    p_RElbow = Plane(RE, RS, P_REENN) # plane perpendicular to p_RElbowN, containing limb RE-RS, normal backwards - away from camera
    
    p_RArm = Plane(RE, RS, RW) # plane of all 3 left-hand joints, normal towards LS if elbow bent and RElbowYaw=pi/2
    
    if p_RElbowN.orientation(RW) > 0: # is on the closer side - closer to body center
        RElbowYaw = - p_RElbow.angle_to(p_RArm) * p_RElbow.orientation(RW)
    else:
        RElbowYaw = - (np.pi - p_RElbow.angle_to(p_RArm)) * p_RElbow.orientation(RW)
        
    ####################################################################
    # HipRoll positive = rotate to left
   
    # Alternative
    P_SM = LS.midpoint_to(RS) # midpoint between shoulders
    l_SM_PE = Line(P_SM, PE)
    p_ShouldersS = Plane(P_SM, l_LS_RS) # side plane perpendicular to shoulders connector, normal towards RS
    
    HipRoll = p_ShouldersS.angle_to(l_SM_PE) * p_ShouldersS.orientation(PE)
    
    ####################################################################
    # HipPitch positive = rotate backwards
    
    HipPitch = np.nan # not supported
    
    #print time.time()-st
    return np.array([
        HeadPitch, HeadYaw, 
        LShoulderRoll, LShoulderPitch, LElbowRoll, LElbowYaw,
        RShoulderRoll, RShoulderPitch, RElbowRoll, RElbowYaw, 
        HipRoll, HipPitch
    ])

