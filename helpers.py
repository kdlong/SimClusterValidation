import numpy as np
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib._color_data as mcd
import plotly.graph_objects as go
import random
import pandas
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
        # Don't propogate central particles forever
        if abs(M[2]) < 400 and M[0]**2 + M[1]**2 > 113.5**2:
            break
        points[i,:] = M
    return points

def neutralTrajectory(initPos, initMom, endz):
    M0 = initPos
    P0 = initMom
    
    points = np.zeros(shape=(100,3))
    points[:,0] = np.linspace(M0[0], P0[0]/P0[2]*endz, 100)
    points[:,1] = np.linspace(M0[1], P0[1]/P0[2]*endz, 100)
    points[:,2] = np.linspace(M0[2], endz, 100)
    #Definitely not the most efficient way...
    for i, point in enumerate(points):
        if point[0]**2+point[1]**2 > 113.5**2:
            break
    filtpoints = np.zeros(shape=(i, 3))
    filtpoints = points[:i,:]
    return filtpoints

def trajectory(initPos, initMom, endz, q):
    return neutralTrajectory(initPos, initMom, endz) if q == 0 else \
                chargedTrajectory(initPos, initMom, endz, q) 

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
    xhits = simHits[hitType+"_x"]
    position = np.zeros(shape=(len(xhits), 3))
    position[:,0] = xhits
    position[:,1] = simHits[hitType+"_y"]
    position[:,2] = simHits[hitType+"_z"]
    return position

def colorsFromIds(ids):
    colormap = {1 : "#c8cbcc", 111 : "red", 211 : 'blue', 11 : 'green', 13 : 'orange', 22 : "lightblue", 
                    2112 : "pink", 2212 : "purple"}
    return [colormap[abs(i)] if abs(i) in colormap else 'black' for i in ids]

def makeRanges(seq):
    if len(seq) < 2:
        return [seq]
    first = seq[0]
    result = []
    for i in range(1, len(seq)+1):
        if i == len(seq) or seq[i] != seq[i-1]+1:
            ins = [first, seq[i-1]] if seq[i-1] != first else [first]
            result.append(ins)
            if i < len(seq):
                first = seq[i]
    return result

def drawSimClusters(df, label, unmerged=None):
    pos = label+"_impactPoint" 
    if not hasattr(df, pos+"_x"):
        pos = label+"_lastPos"

    scidx = df.index
    text = ["Idx: %i<br>nHits: %i<br>pdgId: %i<br>energy: %0.2f" % (i,n,p, e) for (i,n,p,e) \
                in zip(scidx, df[label+"_nHits"], df[label+"_pdgId"], df[label+"_boundaryEnergy"])]

    unmergedLabel = []
    if unmerged is not None:
        unmergedIdx = []
        for i in scidx:
            entry = unmerged[unmerged["SimCluster_MergedSimClusterIdx"] == i].index
            unmergedLabel.append("; ".join(["-".join([str(j) for j in i]) for i in makeRanges(entry)]))
        text = ["%s<br>Unmerged Idxs: %s" % (t,u) for t,u in zip(text, unmergedLabel)]

    return go.Scatter3d(x = df[pos+'_z'], y = df[pos+'_x'], z = df[pos+'_y'],
                mode='markers',
                marker=dict(line=dict(width=1, color='DarkSlateGrey', ),
                    symbol='x', 
                    size=2, 
                    color=[color_for_id(i) for i in scidx], 
                ),
                hovertemplate="x: %{y}<br>y: %{z}<br>z: %{x}<br>%{text}<br>",
                name=label, text=text,
            )

def drawSimTracks(df, label):
    pos = label+"_impactPoint" 
    if not hasattr(df, pos+"_x"):
        pos = label+"_lastPos"

    df = df[df[label+"_crossedBoundary"]]

    text = ["pdgId: %i" % i for i in df[label+"_pdgId"]]

    return go.Scatter3d(x = df[pos+'_z'], y = df[pos+'_x'], z = df[pos+'_y'],
                mode='markers',
                marker=dict(line=dict(width=1, color='DarkSlateGrey', ),
                    symbol='x', 
                    size=2, 
                    color=colorsFromIds(df[label+"_pdgId"]), 
                ),
                hovertemplate="x: %{y}<br>y: %{z}<br>z: %{x}<br>%{text}<br>",
                name=label, text=text,
            )


def drawHits(df, label, endcap, simClusters=None, colorby='pdgId'):
    df.reset_index(inplace=True)
    hits = hitPositionArray(df, label)
    nhits = len(getattr(df, label+"_x").to_numpy())

    pfidx = np.zeros(0)
    scidx = np.zeros(0)
    mergedScidx = np.zeros(0)
    if simClusters is None or simClusters.empty:
        ids = df[label+"_pdgId"] if (label+"_pdgId") in df else [-1]*len(df)
    else:
        scidx = df[label+"_SimClusterIdx"].to_numpy()
        pfidx = df[label+"_PFCandIdx"] if label+"_PFCandIdx" in df else pandas.DataFrame()

        caloidx = np.where(scidx > 0, simClusters["SimCluster_CaloPartIdx"].to_numpy()[scidx], scidx)
        mergedScidx = np.where(scidx > 0, simClusters["SimCluster_MergedSimClusterIdx"].to_numpy()[scidx], scidx)
        ids = np.where(scidx > 0, simClusters["SimCluster_pdgId"].to_numpy()[scidx], scidx)

    if colorby == 'pdgId':
        colors = colorsFromIds(ids)
    else:
        nsc = len(simClusters)
        indices = scidx
        if "simclus" not in colorby:
            indices = caloidx if "calo" in colorby else pfidx
        elif "merged" in colorby:
            indices = mergedScidx
        colors = [color_for_id(i) for i in indices]

    minsize = 2
    maxsize = 5
    energy = df[label+'_energy']
    energyNorm = [x if x > minsize else minsize for x in energy/energy.max()*maxsize]
    normE = True

    text = ['pdgid: %i<br>energy: %0.2e' % (i,e) for i,e in zip(ids, energy)]
    if scidx.size:
        text = [text[i]+"<br>SimClusIdx: %i<br>CaloPartIdx %i" % (l,c) \
                for i,(l,c) in enumerate(zip(scidx, caloidx))]
    if mergedScidx.size:
        text = [text[i]+"<br>MergedSimClusIdx: %i" % m for i,m in enumerate(mergedScidx)]
    if pfidx.size:
        text = [text[i]+"<br>PFCandIdx: %i" % p for i,p in enumerate(pfidx)]

    return go.Scatter3d(x = df[label+'_z'], y = df[label+'_x'], z = df[label+'_y'], 
                mode='markers', 
                marker=dict(line=dict(width=0), size=energyNorm, 
                    color=colors, 
                ),
                hovertemplate="x: %{y}<br>y: %{z}<br>z: %{x}<br>%{text}<br>",
                name=label, text=text,
                )

# Modified from Thomas
_assigned_colors = {}
_all_colors = []
cmap = matplotlib.cm.get_cmap('tab20b')    
for i in range(cmap.N):
    _all_colors.append(matplotlib.colors.rgb2hex(cmap(i)))
_all_colors.extend(list(mcd.XKCD_COLORS.values()))
# TODO: clean up
# This is convoluted since I go back and forth on how complicated I
# want it to be
def color_for_id(i):
    i = int(i)
    global _assigned_colors, _all_colors
    if i < 0:
        _assigned_colors[i] = "#c8cbcc"
    elif not i in _assigned_colors:
        if i >= len(_all_colors):
            i = np.random.randint(0, len(_all_colors))
        i_picked_color = i if i % 2 else (20 if i < 20 else len(_all_colors))-i-1
        _assigned_colors[i] = _all_colors[i_picked_color]
    return _assigned_colors[i]

def drawParticles(df, label, vert=None, ptcut=1, colorbyIdx=False, decay=False):
    pts = df[label+"_pt"]
    etas = df[label+"_eta"]
    pdgids = df[label+"_pdgId"] if label+"_pdgId" in df else np.zeros(len(pts))
    charges = df[label+"_charge"]
    paths = []
    for i, (pt, eta, pdgid, charge) in enumerate(zip(pts, etas, pdgids, charges)):
        if pt < ptcut:
            continue
        vtx = vertex(df, label+"_Vtx", i) if vert is None else vert
        decayvtx = vertex(df, label+"_DecayVtx", i) if decay else np.array([10000]*3)
        mom = momentumVector(df, label, i)
        maxend = 1000 if abs(pdgid) == 13 else 350
        end = decayvtx[2] if abs(decayvtx[2]) < 10000 else (maxend if mom[2] > 0 else -1*maxend)
        # Extend tracks that go to the last layer of the tracker out to HGCal
        if abs(end) > 260:
            end = maxend if end > 0 else -1*maxend
        points = trajectory(vtx, mom, end, charge)
        idx = df.index[i]
        colors = colorsFromIds([pdgid])[0] if not colorbyIdx else color_for_id(idx)
        paths.append(go.Scatter3d(x = points[:,2], y = points[:,0], z = points[:,1],
                mode='lines', name="%sIdx%i (pdgId=%i)" % (label, idx, pdgid), 
                text="pdgid: %i<br>pt: %f<br>eta: %f<br>Idx: %i" % (pdgid, pt, eta, idx),
                hovertemplate="x: %{y}<br>y: %{z}<br>z: %{x}<br>%{text}<br>",
                line=dict(color=colors)
            )
        )
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
