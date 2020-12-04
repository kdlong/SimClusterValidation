import numpy as np
import math
from matplotlib import cm
import random

def vertex(obj, name, evt, i=None):
    x = getattr(obj, "_".join([name, "x"]))[evt]
    y = getattr(obj, "_".join([name, "y"]))[evt]
    z = getattr(obj, "_".join([name, "z"]))[evt]

    if i is None:
        return np.array([x, y, z])

    return np.array([v.to_numpy()[i] for v in [x, y, z]])

def chargedTrajectory(initPos, initMom, endz, q):
    M0 = initPos
    P0 = initMom

    T0 = P0/np.linalg.norm(P0)
    H = np.array([0,0,1])

    s = (endz-M0[2])/T0[2]
    endz = s

    HcrossT = np.cross(H, T0)
    alpha = np.linalg.norm(HcrossT)
    N0 = HcrossT/np.linalg.norm(HcrossT)

    gamma = T0[2]
    Q = -3.8*2.99792458e-3*q/np.linalg.norm(P0)

    points = np.zeros(shape=(100,3))
    for i in range(100):
        step = s/100*i
        theta = Q*step
        M = M0 + gamma*(theta-math.sin(theta))*H/Q + math.sin(theta)*T0/Q + alpha*(1.-math.cos(theta))*N0/Q
        points[i,:] = M
    return points

def neutralTrajectory(initPos, initMom, endz):
    M0 = initPos
    P0 = initMom
    
    points = np.zeros(shape=(100,3))
    points[:,0] = np.linspace(M0[0], P0[0]/P0[2]*endz, 100)
    points[:,1] = np.linspace(M0[1], P0[1]/P0[2]*endz, 100)
    points[:,2] = np.linspace(M0[2], endz, 100)
    return points

def trajectory(initPos, initMom, endz, q):
    return neutralTrajectory(initPos, initMom, endz) if q == 0 else \
                chargedTrajectory(initPos, initMom, endz, q) 

def labelsFromSimCluster(simHits, hitType, simClusters, evt, label='pdgId', endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    hitsdf = simHits[simHits.index.get_level_values(0) == evt]
    hitsdf.reset_index(inplace=True)

    clusterIds = simClusters.SimCluster_pdgId[evt].to_numpy()
    clusterMatch = getattr(hitsdf, hitType+"_SimClusterIdx")
    nhits = len(clusterMatch)
    nsc = len(simClusters.SimCluster_pdgId[evt])
    ids = np.zeros(nhits)
    for i in range(-1, nsc):
        match = hitsdf.index[clusterMatch == i].to_numpy()
        ids[match] = i+i if i < 0 or label != 'pdgId' else clusterIds[i]
    return ids

def momentumVector(obj, objName, evt, part):
    pt = getattr(obj, objName+"_pt")[evt].to_numpy()[part]
    eta = getattr(obj, objName+"_eta")[evt].to_numpy()[part]
    phi = getattr(obj, objName+"_phi")[evt].to_numpy()[part]

    return np.array([pt*math.cos(phi), pt*math.sin(phi), pt*math.sinh(eta)])

def idsFromHit(simHits, hitType, evt, endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    try:
        return getattr(simHits, hitType+"_pdgId")[evt].to_numpy()
    except KeyError:
        return np.zeros(0)

def hitPositionArray(simHits, hitType, evt, endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    xhits = getattr(simHits, hitType+"_x")
    try:
        xhits = xhits[evt].to_numpy()
    except KeyError:
        return np.zeros(shape=(1,3))
    position = np.zeros(shape=(len(xhits), 3))
    position[:,0] = xhits
    position[:,1] = getattr(simHits, hitType+"_y")[evt].to_numpy()
    position[:,2] = getattr(simHits, hitType+"_z")[evt].to_numpy()
    return position

def colorsFromIds(ids):
    colormap = {-1 : 'grey', 111 : "red", 211 : 'blue', 11 : 'green', 13 : 'orange', 22 : "lightblue", 
                    2112 : "pink", 2212 : "pink"}
    return [colormap[abs(i)] if abs(i) in colormap else 'black' for i in ids]

def drawHits(ax, df, label, evt, endcap, simClusters=None, colorby='pdgId'):
    hits = hitPositionArray(df, label, evt, endcap)
    if simClusters is None:
        ids = idsFromHit(df, label, evt, endcap)
    else:
        ids = labelsFromSimCluster(df, label, simClusters, evt, colorby, endcap)

    if colorby == 'pdgId':
        colors = colorsFromIds(ids)
    else:
        colormap = cm.get_cmap("tab20")
        colors = [random.uniform(0, 1) for i in ids]
    ax.scatter(hits[:,2], hits[:,0], hits[:,1], marker='o', c=colors, s=1)    

def drawGenParts(ax, df, label, vertex, evt, endcap):
    pdgids = getattr(df, label+"_pdgId")[evt]
    for i, pdgid in enumerate(pdgids):
        eta = getattr(df, label+"_eta")[evt][i]
        if (eta > 0 and endcap == "-") or (eta < 0 and endcap == "+"):
            continue
        charge = getattr(df, label+"_charge")[evt][i]
        momentum = momentumVector(df, label, evt, i)
        endz = 500 if momentum[2] > 0 else -500
        points = trajectory(vertex, momentum, endz, charge)
        color = colorsFromIds([pdgids[i]])[0]
        ax.plot(points[:,2], points[:,0], points[:,1], c=color)

def drawTrackingParts(ax, df, label, evt, endcap, ptcut=1):
    pts = getattr(df, label+"_pt")[evt]
    etas = getattr(df, label+"_eta")[evt]
    for i, (pt, eta) in enumerate(zip(pts, etas)):
        if (eta > 0 and endcap == "-") or (eta < 0 and endcap == "+") or pt < ptcut:
            continue
        vtx = vertex(df, label+"_Vtx", evt, i)
        decayvtx = vertex(df, label+"_DecayVtx", evt, i)
        mom = momentumVector(df, label, evt, i)
        pdgid = getattr(df, label+"_pdgId")[evt].to_numpy()[i]
        charge = getattr(df, label+"_charge")[evt].to_numpy()[i]
        maxend = 700 if abs(pdgid) == 13 else 400
        end = decayvtx[2] if decayvtx[2] < 10000 else (maxend if mom[2] > 0 else -1*maxend)
        points = trajectory(vtx, mom, end, charge)
        ax.plot(points[:,2], points[:,0], points[:,1], color=colorsFromIds([pdgid])[0])
