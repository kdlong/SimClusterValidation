import numpy as np
import math

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
    for i, s in enumerate(np.linspace(M0[2], endz, 100)):
        theta = Q*s
        M = M0 + gamma*(theta-math.sin(theta))*H/Q + math.sin(theta)*T0/Q + alpha*(1.-math.cos(theta))*N0/Q
        points[i,:] = M
    return points

#def hitIdsFromSimClusAssoc(

def neutralTrajectory(initPos, initMom, endz, q=0):
    M0 = initPos
    P0 = initMom
    
    points = np.zeros(shape=(100,3))
    points[:,0] = np.linspace(M0[0], P0[0]/P0[2]*endz, 100)
    points[:,1] = np.linspace(M0[1], P0[1]/P0[2]*endz, 100)
    points[:,2] = np.linspace(M0[2], endz, 100)
    return points

def simHitsAndClusters(simHits, hitType, simClusters, evt):
    nsc = len(simClusters.SimCluster_pdgId[evt])
    nhits = len(getattr(simHits, hitType+"_x")[evt])
    
    hits = np.zeros(shape=(nsc+1, nhits, 3))
    print(nhits, nsc)
    
    for i in range(-1, nsc):
        idx = i if i > 0 else nsc
        hitsPerSC = simHits[getattr(simHits, hitType+"_SimClusterIdx") == i]
        if not len(hitsPerSC) > evt:
            continue

        x = getattr(hitsPerSC, hitType+"_x")[evt]
        y = getattr(hitsPerSC, hitType+"_y")[evt]
        z = getattr(hitsPerSC, hitType+"_z")[evt]
        hitsClus = len(x.to_numpy()) if type(x) != np.float32 else 1
        hits[idx,:hitsClus,0] = x.to_numpy() if hitsClus > 1 else x
        hits[idx,:hitsClus,1] = y.to_numpy() if hitsClus > 1 else y
        hits[idx,:hitsClus,2] = z.to_numpy() if hitsClus > 1 else z
    return hits

def idFromHit(simHits, hitType, evt, endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    return getattr(simHits, hitType+"_pdgId")[evt].to_numpy()

def hitPositionArray(simHits, hitType, evt, endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    xhits = getattr(simHits, hitType+"_x")[evt].to_numpy()
    position = np.zeros(shape=(len(xhits), 3))
    position[:,0] = xhits
    position[:,1] = getattr(simHits, hitType+"_y")[evt].to_numpy()
    position[:,2] = getattr(simHits, hitType+"_z")[evt].to_numpy()
    return position

def colorsFromIds(ids):
    colormap = {-1 : 'grey', 111 : "red", 211 : 'blue', 11 : 'green', 13 : 'orange', 22 : "lightblue", 
                    2112 : "pink", 2212 : "pink"}
    return [colormap[i] if i in colormap else 'black' for i in ids]
