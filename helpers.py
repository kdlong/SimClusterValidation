import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import random
import logging

def vertex(obj, name, i=None):
    x = getattr(obj, "_".join([name, "x"]))
    y = getattr(obj, "_".join([name, "y"]))
    z = getattr(obj, "_".join([name, "z"]))

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

def labelsFromSimCluster(simHits, hitType, simClusters, label='pdgId', endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    clusterIds = simClusters.SimCluster_pdgId.to_numpy()
    clusterMatch = getattr(simHits, hitType+"_SimClusterIdx")
    caloMatches = simClusters.SimCluster_CaloPartIdx.to_numpy()
    nhits = len(clusterMatch)
    nsc = len(simClusters.SimCluster_pdgId)
    ids = np.zeros(nhits)
    logging.debug("Found %i SimClusters" % nsc)
    nfilled = 0
    for i in range(-1, nsc):
        match = simHits.index[clusterMatch == i].to_numpy()
        if i < 0:
            ids[match] = nsc+1 if i < 0 else i
        elif 'simclus' in label:
            ids[match] = i
            if len(match):
                nfilled += 1
        elif 'calo' in label:
            ids[match] = caloMatches[i]
        elif label == 'pdgId':
            ids[match] = clusterIds[i]
    return ids

def momentumVector(obj, objName, part):
    pt = getattr(obj, objName+"_pt").to_numpy()[part]
    eta = getattr(obj, objName+"_eta").to_numpy()[part]
    phi = getattr(obj, objName+"_phi").to_numpy()[part]

    return np.array([pt*math.cos(phi), pt*math.sin(phi), pt*math.sinh(eta)])

def idsFromHit(simHits, hitType, endcap=""):
    if endcap == "+":
        simHits = simHits[getattr(simHits, hitType+"_z") > 0]
    elif endcap == "-":
        simHits = simHits[getattr(simHits, hitType+"_z") < 0]

    return getattr(simHits, hitType+"_pdgId").to_numpy()

def hitPositionArray(simHits, hitType):
    xhits = getattr(simHits, hitType+"_x")
    position = np.zeros(shape=(len(xhits), 3))
    position[:,0] = xhits
    position[:,1] = getattr(simHits, hitType+"_y").to_numpy()
    position[:,2] = getattr(simHits, hitType+"_z").to_numpy()
    return position

def colorsFromIds(ids):
    colormap = {-1 : 'grey', 111 : "red", 211 : 'blue', 11 : 'green', 13 : 'orange', 22 : "lightblue", 
                    2112 : "pink", 2212 : "purple"}
    return [colormap[abs(i)] if abs(i) in colormap else 'black' for i in ids]

def drawSimClusters(ax, df, endcap=""):
    label = "SimCluster_impactPoint" 
    if not hasattr(df, label):
        label = "SimCluster_lastPos"
    points = hitPositionArray(df, label)
    nsc = len(df)
    norm = matplotlib.colors.Normalize(vmin=-1, vmax=nsc)
    colormap = matplotlib.cm.get_cmap("gist_rainbow")
    #colormap = matplotlib.cm.get_cmap("tab20")
    plt.set_cmap(colormap)
    print("Number of points", len(points))
    #colors = np.array([i if i % 2 else nsc-i for i in range(nsc)])
    colors = 'black'
    ax.scatter(points[:,2], points[:,0], points[:,1], marker='x', c=colors, norm=norm, s=100)

def drawHits(ax, df, label, endcap, simClusters=None, colorby='pdgId'):
    if endcap == "+":
        df = df[getattr(df, label+"_z") > 0]
    elif endcap == "-":
        df = df[getattr(df, label+"_z") < 0]
    df.reset_index(inplace=True)
    hits = hitPositionArray(df, label)
    nhits = len(getattr(df, label+"_x").to_numpy())

    print("Number of hits is", nhits)
    if simClusters is None:
        ids = idsFromHit(df, label, endcap)
    else:
        ids = labelsFromSimCluster(df, label, simClusters, colorby, endcap)

    if colorby == 'pdgId':
        colors = colorsFromIds(ids)
        norm = None
    else:
        colormap = matplotlib.cm.get_cmap("gist_rainbow")
        #colormap = matplotlib.cm.get_cmap("tab20")
        plt.set_cmap(colormap)
        nsc = len(simClusters)
        norm = matplotlib.colors.Normalize(vmin=-1, vmax=nsc)
        print("The number of unique IDs is", len(np.unique(ids)))
        print("The number of unique colors is", len(np.unique(norm(ids))))
        print("Number of SCs is", len(simClusters))
        colors = [i if i % 2 else nsc-i for i in ids]

    ax.scatter(hits[:,2], hits[:,0], hits[:,1], marker='o', c=colors, norm=norm, s=1)

def drawGenParts(ax, df, label, vertex, endcap):
    pdgids = getattr(df, label+"_pdgId")
    for i, pdgid in enumerate(pdgids):
        eta = getattr(df, label+"_eta")[i]
        pt = getattr(df, label+"_pt")[i]
        print("pt is", pt)
        if (eta > 0 and endcap == "-") or (eta < 0 and endcap == "+"):
            continue
        charge = getattr(df, label+"_charge")[i]
        momentum = momentumVector(df, label, i)
        endz = 500 if momentum[2] > 0 else -500
        points = trajectory(vertex, momentum, endz, charge)
        color = colorsFromIds([pdgids[i]])[0]
        ax.plot(points[:,2], points[:,0], points[:,1], c=color)

def drawTrackingParts(ax, df, label, endcap, ptcut=1, hasDecay=True):
    pts = getattr(df, label+"_pt")
    etas = getattr(df, label+"_eta")
    tot = 0 
    for i, (pt, eta) in enumerate(zip(pts, etas)):
        if (eta > 0 and endcap == "-") or (eta < 0 and endcap == "+") or pt < ptcut:
            continue
        tot += 1
        vtx = vertex(df, label+"_Vtx", i)
        decayvtx = vertex(df, label+"_DecayVtx", i) if hasDecay else np.array([10000]*3)
        mom = momentumVector(df, label, i)
        pdgid = getattr(df, label+"_pdgId").to_numpy()[i]
        charge = getattr(df, label+"_charge").to_numpy()[i]
        maxend = 700 if abs(pdgid) == 13 else 400
        end = decayvtx[2] if decayvtx[2] < 10000 else (maxend if mom[2] > 0 else -1*maxend)
        points = trajectory(vtx, mom, end, charge)
        ax.plot(points[:,2], points[:,0], points[:,1], color=colorsFromIds([pdgid])[0])
        print("Particle ID is", pdgid, "pt is", pt, "eta is", eta)
    print("Number of particles plotted was ", tot)
