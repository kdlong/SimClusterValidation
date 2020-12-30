#!/usr/bin/env python3
import uproot
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import helpers
import numpy as np
import math
import multiprocessing
import argparse
import os
import logging
import shutil
import plotly.graph_objects as go

class HitsAndTracksPlotter(object):
    def __init__(self, rtfile, outpath):
        self.outpath = outpath
        if not os.path.isdir(self.outpath):
            logging.info("Creating folder %s" % self.outpath)
            os.makedirs(self.outpath)
            shutil.copy("/afs/cern.ch/user/k/kelong/index.php", self.outpath)
        rtfile = uproot.open(rtfile)

        self.showPixelHits = True
        self.showMuonHits = True
        self.showGen = True
        self.showCaloPart = True
        self.showTracking  = True
        self.colorBy = 'pdgId'
        self.trackingCut = 1
        self.nThreads = 0
        self.endcap = ""
        self.showTICL = False
        
        self.allGen = rtfile["Events"].pandas.df(["GenPart*"], flatten=True)
        self.allGenVtx = rtfile["Events"].pandas.df(["GenVtx*"], flatten=True)
        self.allTracking = rtfile["Events"].pandas.df(["Tracking*"], flatten=True)
        self.allCalo = rtfile["Events"].pandas.df(["CaloPart*"], flatten=True)
        self.allTicl = rtfile["Events"].pandas.df(["PFTICLCand*"], flatten=True)
        self.allPf = rtfile["Events"].pandas.df(["PFCand*"], flatten=True)
        if not self.allPf.empty:
            self.allPf = self.allPf[(self.allPf.PFCand_eta < -1.6) | (self.allPf.PFCand_eta > 1.6)]

        self.allSimHitsHGCEE = rtfile["Events"].pandas.df(["SimHitHGCEE*"], flatten=True)
        self.allSimHitsHGCHEfront = rtfile["Events"].pandas.df(["SimHitHGCHEfront*"], flatten=True)
        self.allSimHitsHGCHEback = rtfile["Events"].pandas.df(["SimHitHGCHEback*"], flatten=True)
        self.allSimClusters = rtfile["Events"].pandas.df(["SimCluster*"], flatten=True)
        self.allSimHitsPixelEC = rtfile["Events"].pandas.df(["SimHitPixelEC*"], flatten=True)
        self.allSimHitsPixel = rtfile["Events"].pandas.df(["SimHitPixelLowTof*"], flatten=True)
        self.allSimHitsCSC = rtfile["Events"].pandas.df(["SimHitMuon*"], flatten=True)

        self.gen = self.allGen
        self.tracking = self.allTracking
        self.calo = self.allCalo
        self.ticl = self.allTicl
        self.pf = self.allPf

        self.simHitsHGCEE = self.allSimHitsHGCEE 
        self.simHitsHGCHEfront = self.allSimHitsHGCHEfront 
        self.simHitsHGCHEback = self.allSimHitsHGCHEback 
        self.simClusters = self.allSimClusters 
        self.simHitsPixelEC = self.allSimHitsPixelEC 
        self.simHitsPixel = self.allSimHitsPixel 
        self.simHitsCSC = self.allSimHitsCSC 

    def setShowTICL(self, show):
        self.showTICL = show

    def setShowPF(self, show):
        self.showPF = show

    def setShowGenPart(self, show):
        self.showGen = show

    def setShowCaloPart(self, show):
        self.showCaloPart = show

    def setShowTrackHits(self, show):
        print("Show track hits!")
        self.showPixelHits = show
        self.showMuonHits = show

    def setColorBy(self, by):
        self.colorBy = by

    def setShowCaloPart(self, show):
        self.showCaloPart = show

    def setShowTrackingPart(self, show):
        self.showTracking = show

    def setNumThreads(self, threads):
        self.nThreads = threads

    def setPath(self, path):
        self.path = path

    def setEndcap(self, endcap):
        self.endcap = endcap

    #def filterEndcap(self):

    def setTrackingCut(self, cut):
        self.trackingCut = cut
    
    def filterEvent(self, evt):
        self.gen = self.allGen.xs(evt, level="entry")
        self.genVtx = helpers.vertex(self.allGenVtx, "GenVtx", evt)
        self.tracking = self.allTracking.xs(evt, level="entry")
        self.calo = self.allCalo.xs(evt, level="entry")
        if not self.allPf.empty:
            self.pf = self.allPf.xs(evt, level="entry")
        if not self.allTicl.empty:
            self.ticl = self.allTicl.xs(evt, level="entry")

        self.simHitsHGCEE = self.allSimHitsHGCEE.xs(evt, level="entry")
        self.simHitsHGCHEfront = self.allSimHitsHGCHEfront.xs(evt, level="entry")
        self.simHitsHGCHEback = self.allSimHitsHGCHEback.xs(evt, level="entry")
        self.simClusters = self.allSimClusters.xs(evt, level="entry")
        self.simHitsPixelEC = self.allSimHitsPixelEC.xs(evt, level="entry")
        self.simHitsPixel = self.allSimHitsPixel.xs(evt, level="entry")
        self.simHitsCSC = self.allSimHitsCSC.xs(evt, level="entry")

    def __call__(self, args):
        self.makePlot(*args)

    def runPlots(self, inputs):
        if self.nThreads > 0:
            p = multiprocessing.Pool(processes=min(self.nThreads, len(inputs)))
            p.map(self, inputs)
        else:
            for i in inputs:
                self.makePlot(*i)

    def makeMoviePlot(self, events, basename):
        for e in events:
            angles = range(360)
            name = '_'.join([basename, "evt%i" % e])
            inputs = [[e, name, a] for a in angles]
            self.runPlots(inputs)
        with open("/".join([self.outpath, "makeMovie.sh"]), "w") as f:
            f.write("ffmpeg -f image2 -r 20 -i ./{name}_angle%d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 {name}.mp4".format(name=name))

    def makePlots(self, events, basename):
        self.runPlots([[e, '_'.join([basename, "evt%i" % e])] for e in  events])

    def makePlot(self, evt, name, angle=None):
        print(evt, name, angle)
        fig = plt.figure(figsize=[10, 10])
        ax = fig.add_subplot(111, projection='3d')

        self.filterEvent(evt)

        df = self.simClusters
        label = "SimCluster_lastPos"
        if self.endcap == "+":
            df = [getattr(df, label+"_z") > 0]
        elif self.endcap == "-":
            df = df[getattr(df, label+"_z") < 0]

        helpers.drawHits(ax, self.simHitsHGCEE, "SimHitHGCEE", self.endcap, df, self.colorBy)
        helpers.drawHits(ax, self.simHitsHGCHEfront, "SimHitHGCHEfront", self.endcap, df, self.colorBy)
        helpers.drawHits(ax, self.simHitsHGCHEback, "SimHitHGCHEback", self.endcap, df, self.colorBy)

        if self.showPixelHits:
            print("Drawing pixel")
            helpers.drawHits(ax, self.simHitsPixel, "SimHitPixelLowTof", self.endcap)
            helpers.drawHits(ax, self.simHitsPixelEC, "SimHitPixelECLowTof", self.endcap)
        if self.showMuonHits:
            helpers.drawHits(ax, self.simHitsCSC, "SimHitMuonCSC", self.endcap)

        if self.showPF:
            helpers.drawTrackingParts(ax, self.pf, "PFCand", self.endcap, 0, False)
        if self.showTICL:
            helpers.drawTrackingParts(ax, self.ticl, "PFTICLCand", self.endcap, 0, False)
        if self.showCaloPart:
            helpers.drawGenParts(ax, self.calo, "CaloPart", self.genVtx, self.endcap)
        if self.showGen:
            helpers.drawGenParts(ax, self.gen, "GenPart", self.genVtx, self.endcap)
        if self.showTracking:
            helpers.drawTrackingParts(ax, self.tracking, "TrackingPart", self.endcap, self.trackingCut)
        #helpers.drawSimClusters(ax, df, self.endcap)

        zlim = [-600,600]
        #zlim = [0,500]
        if self.endcap == "+":
            zlim = [0, 500]
        elif self.endcap == "-":
            zlim = [-600, 0]

        ax.set_xlim3d(*zlim)
        ax.set_ylim3d(-150,150)
        ax.set_zlim3d(-150,150)
        ax.set_xlabel('z pos')
        ax.set_ylabel('x pos')
        ax.set_zlabel('y pos')
        dpi=300
        if angle is not None:
            ax.view_init(30, angle)
            print(name)
            name += "_angle%i.png" % angle
            print(name)
            dpi = 100
        fig.savefig("/".join([self.outpath, name]), dpi=dpi)
        plt.close(fig)

parser = argparse.ArgumentParser()
parser.add_argument("--endcap", type=str, choices=["+","-"], default="", help="plot only one endcap")
parser.add_argument("-e", "--events", type=str, nargs='*', required=True, help="Which event to plot")
parser.add_argument("-o", "--outfile", type=str, required=True, help="Name of output file")
parser.add_argument("-p", "--path", type=str, default="/eos/user/k/kelong/www/ML4Reco/PFParticleValidation", help="output path")
parser.add_argument("-s", "--subfolder", type=str, default="", help="append folder to output path")
parser.add_argument("-m", "--movie", action='store_true', help="Produce scan in angle that can be made into a movie")
parser.add_argument("--noTrackHits", action='store_true', help="Don't show hits in tracker")
parser.add_argument("--tracking", action='store_true', help="Don't show tracking particles")
parser.add_argument("--caloPart", action='store_true', help="Show calo particles")
parser.add_argument("--ticl", action='store_true', help="Show TICL cands")
parser.add_argument("--pf", action='store_true', help="Show PF cands")
parser.add_argument("--gen", action='store_true', help="Don't show gen particles")
parser.add_argument("--colorBy", choices=['pdgId', 'simclus', 'calopart'], default="pdgId", help="Color calo hits by pdgId or SimCluster association")
parser.add_argument("--default", action='store_true', help="Use default (not fine calo) input")
parser.add_argument("-j", "--nCores", type=int, default=0, help="Number of cores, currently only for movie making")
parser.add_argument("-c", "--cut", type=float, default=1, help="Tracking particle pt cut")

args = parser.parse_args()
if len(args.events) == 1 and ":" in args.events[0]:
    args.events = range(*[int(i) for i in args.events[0].split(":")])
else:
    args.events = [int(i) for i in args.events]

if args.movie and args.path == parser.get_default('path'):
    args.path = args.path.replace("www/", "")

plotter = HitsAndTracksPlotter(
    #"/eos/cms/store/cmst3/group/hgcal/CMG_studies/kelong/GeantTruthStudy/SimClusterNtuples/test_ExactShoot5Part_seed0_nano.root",
    "/home/kelong/work/ML4Reco/defaultCMSSW/CMSSW_11_2_0_pre9/src/production_tests/Nano_SinglePion150To200_seed0.root" if args.default else \
        "/home/kelong/work/ML4Reco/CMSSW_11_2_0_pre9/src/production_tests/Nano_SinglePion150To200_seed0_fineCaloSC.root",
        #"/home/kelong/work/ML4Reco/CMSSW_11_2_0_pre9/src/production_tests/Nano_SinglePion150To200_seed0.root",
    args.path if not args.subfolder else "/".join([args.path, args.subfolder]))

plotter.setEndcap(args.endcap)
plotter.setNumThreads(args.nCores)
plotter.setTrackingCut(args.cut)
plotter.setShowGenPart(args.gen)
plotter.setShowCaloPart(args.caloPart)
plotter.setShowTICL(args.ticl)
plotter.setShowPF(args.pf)
plotter.setShowTrackingPart(args.tracking)
plotter.setShowTrackHits(not args.noTrackHits)
plotter.setColorBy(args.colorBy)

if not args.movie:
    plotter.makePlots(args.events, args.outfile)
else:
    plotter.makeMoviePlot(args.events, args.outfile)
