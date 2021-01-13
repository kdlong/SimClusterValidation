import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import plotly.graph_objects as go
import random
import logging
logging.basicConfig(level=logging.INFO)

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

def labelsFromSimCluster(simHits, hitType, simClusters, label='pdgId'):
    clusterIds = simClusters.SimCluster_pdgId.to_numpy()
    clusterMatch = simHits[hitType+"_SimClusterIdx"].to_numpy()
    caloMatches = simClusters.SimCluster_CaloPartIdx.to_numpy()
    nhits = len(clusterMatch)
    nsc = len(simClusters.SimCluster_pdgId)
    ids = np.zeros(nhits)
    logging.debug("Found %i SimClusters" % nsc)
    nfilled = 0
    for i in range(-1, nsc):
        match = simHits.index[clusterMatch == i].to_numpy()
        if 'simclus' in label:
            ids[match] = i
            if len(match):
                nfilled += 1
        elif 'calo' in label:
            ids[match] = -1 if i < 0 else caloMatches[i]
        elif label == 'pdgId':
            ids[match] = -1 if i < 0 else clusterIds[i]
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

def drawHits(df, label, endcap, simClusters=None, colorby='pdgId'):
    df.reset_index(inplace=True)
    hits = hitPositionArray(df, label)
    nhits = len(getattr(df, label+"_x").to_numpy())

    labels = np.zeros(0)
    if simClusters is None:
        ids = idsFromHit(df, label, endcap)
    else:
        ids = labelsFromSimCluster(df, label, simClusters, 'pdgId')
        labels = labelsFromSimCluster(df, label, simClusters, 'simclus')
        caloparts = labelsFromSimCluster(df, label, simClusters, 'calo')

    if colorby == 'pdgId':
        colors = colorsFromIds(ids)
    else:
        nsc = len(simClusters)
        colors = [color_for_id(i) for i in (labels if colorby == 'simclus' else caloparts)]

    minsize = 2
    maxsize = 10
    energy = df[label+'_energy']
    energyNorm = [x if x > minsize else minsize for x in energy/energy.max()*maxsize]
    normE = True

    text = ['pdgid: %i<br>energy: %0.2e' % (i,e) for i,e in zip(ids, energy)]
    if labels.size:
        text = [text[i]+"<br>SimClusIdx: %i<br>CaloPartIdx %i" % (l,c) for i,(l,c) in enumerate(zip(labels, caloparts))]

    return go.Scatter3d(x = df[label+'_z'], y = df[label+'_x'], z = df[label+'_y'], 
                mode='markers', 
                marker=dict(line=dict(width=0), size=energyNorm, 
                    color=colors, 
                ),
                name=label, text=text,
                )

# Modified from Thomas
_all_colors = list(mcd.XKCD_COLORS.values())
_assigned_colors = {}
def color_for_id(i):
    i = int(i)
    global _assigned_colors, _all_colors
    if i < 0:
        _assigned_colors[i] = "#c8cbcc"
    elif not i in _assigned_colors:
        if i > len(_all_colors):
            i = np.random(len(_all_colors))
        i_picked_color = i if i % 2 else len(_all_colors)-i-1
        _assigned_colors[i] = _all_colors[i_picked_color]
    return _assigned_colors[i]

def drawGenParts(ax, df, label, vertex, endcap):
    pdgids = df[label+"_pdgId"]
    for i, pdgid in enumerate(pdgids):
        eta = getattr(df, label+"_eta")[i]
        pt = getattr(df, label+"_pt")[i]
        if (eta > 0 and endcap == "-") or (eta < 0 and endcap == "+"):
            continue
        charge = getattr(df, label+"_charge")[i]
        momentum = momentumVector(df, label, i)
        endz = 500 if momentum[2] > 0 else -500
        points = trajectory(vertex, momentum, endz, charge)
        color = colorsFromIds([pdgids[i]])[0]
        ax.plot(points[:,2], points[:,0], points[:,1], c=color)

def drawParticles(df, label, vert=None, ptcut=1, decay=False):
    pts = df[label+"_pt"]
    etas = df[label+"_eta"]
    pdgids = df[label+"_pdgId"]
    charges = df[label+"_charge"]
    paths = []
    for i, (pt, eta, pdgid, charge) in enumerate(zip(pts, etas, pdgids, charges)):
        if pt < ptcut:
            continue
        vtx = vertex(df, label+"_Vtx", i) if vert is None else vert
        decayvtx = vertex(df, label+"_DecayVtx", i) if decay else np.array([10000]*3)
        mom = momentumVector(df, label, i)
        maxend = 700 if abs(pdgid) == 13 else 400
        end = decayvtx[2] if decayvtx[2] < 10000 else (maxend if mom[2] > 0 else -1*maxend)
        points = trajectory(vtx, mom, end, charge)
        paths.append(go.Scatter3d(x = points[:,2], y = points[:,0], z = points[:,1],
                mode='lines', name="%s (pdgId=%i)" % (label, pdgid), 
                text="pdgid: %i<br>pt: %f<br>eta: %f" % (pdgid, pt, eta),
                line=dict(color=colorsFromIds([pdgid])[0])))
    return paths

def drawTracker():
    x, y, z = cylinder(113.5, 282*2, a=-282)
    return go.Surface(x=z, y=x, z=y,
                colorscale = [[0, '#d7dff5'], [1, '#d7dff5']],
                showscale=False,
                name='Tracker',
                hoverinfo='skip',
                opacity=0.25)

def drawCSCME1():
    x, y, z = boundary_circle(275, 580)
    return [go.Scatter3d(x=z, y=x, z=y,
                mode='lines',
                surfaceaxis=0,
                line=dict(color='#f5ebd7'),
                opacity=0.25,
                hoverinfo='skip',
                name='CSC ME1/1'),
            go.Scatter3d(x=[-i for i in z], y=x, z=y,
                mode='lines',
                surfaceaxis=0,
                line=dict(color='#f5ebd7'),
                opacity=0.25,
                hoverinfo='skip',
                name='CSC ME-1/1'),
            ]

def drawHGCFront():
    x, y, z = boundary_circle(125, 315)
    return [go.Scatter3d(x=z, y=x, z=y,
                mode='lines',
                surfaceaxis=0,
                line=dict(color='#bacfbe'),
                opacity=0.25,
                name='HGCAL front',
                hoverinfo='skip',
                ),
            go.Scatter3d(x=[-i for i in z], y=x, z=y,
                mode='lines',
                surfaceaxis=0,
                line=dict(color='#bacfbe'),
                opacity=0.25,
                hoverinfo='skip',
                name='HGCAL front'),
            ]

# From https://community.plotly.com/t/basic-3d-cylinders/27990
def cylinder(r, h, a =0, nt=100, nv =50):
    """
    parametrize the cylinder of radius r, height h, base point a
    """
    theta = np.linspace(0, 2*np.pi, nt)
    v = np.linspace(a, a+h, nv )
    theta, v = np.meshgrid(theta, v)
    x = r*np.cos(theta)
    y = r*np.sin(theta)
    z = v
    return x, y, z

def boundary_circle(r, h, nt=100):
    """
    r - boundary circle radius
    h - height above xOy-plane where the circle is included
    returns the circle parameterization
    """
    theta = np.linspace(0, 2*np.pi, nt)
    x= r*np.cos(theta)
    y = r*np.sin(theta)
    z = h*np.ones(theta.shape)
    return x, y, z
