#!/usr/bin/env python3
#import uproot
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import uproot3 as uproot
import helpers
import numpy as np
import math
import multiprocessing
import argparse
import os
import logging
import shutil
import plotly.graph_objects as go
import logging
import pandas

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
        self.showGen = False
        self.showSimTracks = False
        self.showCaloPart = False
        self.showTracking  = False
        self.colorBy = 'pdgId'
        self.trackingCut = 1
        self.nThreads = 1
        self.endcap = ""
        self.showTICL = False
        self.showTracks = False
        self.recHits = False
        self.showSimClusters = "none"
        
        self.allGen = rtfile["Events"].pandas.df(["GenPart*"], flatten=True)
        self.allGen = self.allGen[self.allGen["GenPart_status"] == 1]
        self.allGenVtx = rtfile["Events"].pandas.df(["GenVtx*"], flatten=True)
        self.allTracking = rtfile["Events"].pandas.df(["Tracking*"], flatten=True)
        self.allCalo = rtfile["Events"].pandas.df(["CaloPart*"], flatten=True)
        self.allTicl = rtfile["Events"].pandas.df(["PFTICLCand*"], flatten=True)
        self.allPf = rtfile["Events"].pandas.df(["PFCand*"], flatten=True)
        self.allTracks = rtfile["Events"].pandas.df(["Track_*"], flatten=True)
        self.allConvTracks = rtfile["Events"].pandas.df(["TrackConv*"], flatten=True)

        self.allRecHitsHGCEE = rtfile["Events"].pandas.df(["RecHitHGCEE*"], flatten=True)
        self.allRecHitsHGCHEF = rtfile["Events"].pandas.df(["RecHitHGCHEF*"], flatten=True)
        self.allRecHitsHGCHEB = rtfile["Events"].pandas.df(["RecHitHGCHEB*"], flatten=True)

        self.allSimClusters = rtfile["Events"].pandas.df(["SimCluster*"], flatten=True)
        self.allSimTracks = rtfile["Events"].pandas.df(["SimTrack*"], flatten=True)
        self.allMergedSimClusters = rtfile["Events"].pandas.df(["MergedSimCluster*"], flatten=True)
        self.allSimHitsHGCEE = rtfile["Events"].pandas.df(["SimHitHGCEE*"], flatten=True)
        self.allSimHitsHGCHEF = rtfile["Events"].pandas.df(["SimHitHGCHEF*"], flatten=True)
        self.allSimHitsHGCHEB = rtfile["Events"].pandas.df(["SimHitHGCHEB*"], flatten=True)

        self.allSimHitsPixelEC = rtfile["Events"].pandas.df(["SimHitPixelEC*"], flatten=True)
        self.allSimHitsPixel = rtfile["Events"].pandas.df(["SimHitPixelLowTof*"], flatten=True)
        self.allSimHitsCSC = rtfile["Events"].pandas.df(["SimHitMuon*"], flatten=True)

        self.gen = self.allGen
        self.tracking = self.allTracking
        self.calo = self.allCalo
        self.ticl = self.allTicl
        self.pf = self.allPf
        self.tracks = self.allTracks
        self.convTracks = self.allConvTracks

        self.simHitsHGCEE = self.allSimHitsHGCEE 
        self.simHitsHGCHEF = self.allSimHitsHGCHEF 
        self.simHitsHGCHEB = self.allSimHitsHGCHEB 
        self.simClusters = self.allSimClusters 
        self.simTracks = self.allSimTracks
        self.mergedSimClusters = self.allMergedSimClusters 
        self.simHitsPixelEC = self.allSimHitsPixelEC 
        self.simHitsPixel = self.allSimHitsPixel 
        self.simHitsCSC = self.allSimHitsCSC 

    def setRecHits(self, rechits):
        self.recHits = rechits

    def setShowTICL(self, show):
        self.showTICL = show

    def setShowPF(self, show):
        self.showPF = show

    def setShowGenPart(self, show):
        self.showGen = show

    def setShowSimTracks(self, show):
        self.showSimTracks = show

    def setShowCaloPart(self, show):
        self.showCaloPart = show

    def setShowTrackHits(self, show):
        self.showPixelHits = show
        self.showMuonHits = show

    def setColorBy(self, by):
        self.colorBy = by

    def setShowCaloPart(self, show):
        self.showCaloPart = show

    def setShowTrackingPart(self, show):
        self.showTracking = show

    def setShowTracks(self, show):
        self.showTracks = show

    def setShowSimClusters(self, drawType):
        self.showSimClusters = drawType

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
        if not self.tracking.empty:
            self.tracking = self.allTracking.xs(evt, level="entry")
        if not self.calo.empty:
            self.calo = self.allCalo.xs(evt, level="entry")
        if not self.allPf.empty:
            self.pf = self.allPf.xs(evt, level="entry")
        if not self.allTicl.empty:
            self.ticl = self.allTicl.xs(evt, level="entry")
        if not self.allTracks.empty:
            self.tracks = self.allTracks.xs(evt, level="entry")
        if not self.allConvTracks.empty:
            # TODO: Learn the syntax to avoid this
            try:
                self.convTracks = self.allConvTracks.xs(evt, level="entry")
            except KeyError:
                self.convTracks = pandas.DataFrame()

        if self.recHits:
            self.recHitsHGCEE = self.allRecHitsHGCEE.xs(evt, level="entry")
            self.recHitsHGCHEF = self.allRecHitsHGCHEF.xs(evt, level="entry")
            self.recHitsHGCHEB = self.allRecHitsHGCHEB.xs(evt, level="entry")

            self.recHitsHGCEE = self.recHitsHGCEE[~((self.recHitsHGCEE["RecHitHGCEE_energy"] < 0.01) &
                                        (self.recHitsHGCEE["RecHitHGCEE_SimClusterIdx"] < 0))]
            self.recHitsHGCHEF = self.recHitsHGCHEF[~((self.recHitsHGCHEF["RecHitHGCHEF_energy"] < 0.1) &
                                        (self.recHitsHGCHEF["RecHitHGCHEF_SimClusterIdx"] < 0))]
            self.recHitsHGCHEB = self.recHitsHGCHEB[~((self.recHitsHGCHEB["RecHitHGCHEB_energy"] < 0.01) &
                                        (self.recHitsHGCHEB["RecHitHGCHEB_SimClusterIdx"] < 0))]

        # Not really necessary to copy, just doing this to avoid warnings
        # Can change if the memory usage is excessive
        if not self.allSimClusters.empty:
            self.simClusters = self.allSimClusters.xs(evt, level="entry").copy()
        if not self.allMergedSimClusters.empty:
            self.mergedSimClusters = self.allMergedSimClusters.xs(evt, level="entry").copy()
        self.simTracks = self.allSimTracks.xs(evt, level="entry")
        self.simHitsHGCEE = self.allSimHitsHGCEE.xs(evt, level="entry").copy()
        self.simHitsHGCHEF = self.allSimHitsHGCHEF.xs(evt, level="entry").copy()
        self.simHitsHGCHEB = self.allSimHitsHGCHEB.xs(evt, level="entry").copy()
        self.simHitsPixelEC = self.allSimHitsPixelEC.xs(evt, level="entry")
        self.simHitsPixel = self.allSimHitsPixel.xs(evt, level="entry")
        if not self.allSimHitsCSC.empty:
            try:
                self.simHitsCSC = self.allSimHitsCSC.xs(evt, level="entry")
            except KeyError:
                logging.warning("No CSC hits found for event")
                self.simHitsCSC = pandas.DataFrame()

        # TODO: Make configurable
        # Remove SCs off face
        if not True:
            zpos = self.simClusters.SimCluster_impactPoint_z.to_numpy()
            nhits = self.simClusters.SimCluster_nHits.to_numpy()
            face = 320
            tol = 5
            minhits = 10

            for df, label in [(self.simHitsHGCEE, "SimHitHGCEE"), (self.simHitsHGCHEF, "SimHitHGCHEF"), 
                                                                    (self.simHitsHGCHEB, "SimHitHGCHEB")]:
                idx = df[label+"_SimClusterIdx"]
                zposfilt = zpos[idx]
                nhitsfilt = nhits[idx]
                df.loc[((abs(zposfilt) - face) > tol) | (nhitsfilt < minhits), label+"_SimClusterIdx"] = -1

            zpos = self.mergedSimClusters.MergedSimCluster_impactPoint_z.to_numpy()
            nhits = self.mergedSimClusters.MergedSimCluster_nHits.to_numpy()
            self.mergedSimClusters = self.mergedSimClusters[((abs(zpos) - face) < tol) & (nhits > minhits)]

    def __call__(self, args):
        return self.makePlot(*args)

    def runPlots(self, inputs):
        if self.nThreads > 1:
            p = multiprocessing.Pool(processes=min(self.nThreads, len(inputs)))
            return p.map(self, inputs)
        else:
            return [x for x in map(self, inputs)]

    def makeMoviePlot(self, events, basename):
        for e in events:
            angles = range(360)
            name = '_'.join([basename, "evt%i" % e])
            inputs = [[e, name, a] for a in angles]
            self.runPlots(inputs)
        with open("/".join([self.outpath, "makeMovie.sh"]), "w") as f:
            f.write("ffmpeg -f image2 -r 20 -i ./{name}_angle%d.png -f mp4 -q:v 0 -vcodec mpeg4 -r 20 {name}.mp4".format(name=name))

    def makePlots(self, events, basename):
        return self.runPlots([[e, '_'.join([basename, "evt%i" % e])] for e in  events])

    def makePlot(self, evt, name, angle=None):
        self.filterEvent(evt)
        df = self.simClusters
        data = []

        hits = [(self.simHitsHGCEE, "SimHitHGCEE"), (self.simHitsHGCHEF, "SimHitHGCHEF"), \
                        (self.simHitsHGCHEB, "SimHitHGCHEB")] if not self.recHits else \
                    [(self.recHitsHGCEE, "RecHitHGCEE"), (self.recHitsHGCHEF, "RecHitHGCHEF"), \
                        (self.recHitsHGCHEB, "RecHitHGCHEB")] 
        for hitpair in hits:
            data.append(helpers.drawHits(*hitpair, self.endcap, df, self.colorBy))

        data.append(helpers.drawTracker())
        data.extend(helpers.drawCSCME1())
        data.extend(helpers.drawHGCFront())
        layout = go.Layout(title="5 particle gun, Color by %s" % self.colorBy,
                            scene = dict(
                                aspectmode='data',
                                xaxis=dict(range=[-1500, 1500], title="z (beamline)",
                                    showgrid=True, gridcolor='#aebacf', 
                                    showbackground=True, backgroundcolor='#fafcff'),
                                yaxis=dict(range=[-400, 400], title="x",
                                    showgrid=True, gridcolor='white', 
                                    showbackground=True, backgroundcolor='#fafcff'),
                                zaxis=dict(range=[-400, 400], title="y", 
                                    showgrid=True, gridcolor='white', 
                                    showbackground=True, backgroundcolor='#f7faff'),
                                ),
                            )
        if self.showPixelHits:
            data.append(helpers.drawHits(self.simHitsPixel, "SimHitPixelLowTof", self.endcap))
            data.append(helpers.drawHits(self.simHitsPixelEC, "SimHitPixelECLowTof", self.endcap))
        if self.showMuonHits and not self.simHitsCSC.empty:
            data.append(helpers.drawHits(self.simHitsCSC, "SimHitMuonCSC", self.endcap))

        if self.showPF:
            data.extend(helpers.drawParticles(self.pf, "PFCand", ptcut=self.trackingCut,
               colorbyIdx=(self.colorBy == "pfcand")))
        if self.showTICL:
            data.extend(helpers.drawParticles(self.ticl, "PFTICLCand", ptcut=self.trackingCut))
        if self.showCaloPart:
            data.extend(helpers.drawParticles(self.calo, "CaloPart", self.genVtx, self.trackingCut,
               colorbyIdx=(self.colorBy == "calopart")))
        if self.showGen:
            data.extend(helpers.drawParticles(self.gen, "GenPart", self.genVtx, self.trackingCut))
        if self.showTracking:
            data.extend(helpers.drawParticles(self.tracking, "TrackingPart", ptcut=self.trackingCut, decay=True))
        if self.showTracks:
            data.extend(helpers.drawParticles(self.tracks, "Track", ptcut=self.trackingCut, decay=True))
            if not self.convTracks.empty:
                data.extend(helpers.drawParticles(self.convTracks, "TrackConv", ptcut=self.trackingCut, decay=True))

        if self.showSimTracks:
            data.append(helpers.drawSimTracks(self.simTracks, "SimTrack"))

        if self.showSimClusters:
            if self.showSimClusters == "default":
                data.append(helpers.drawSimClusters(self.simClusters, "SimCluster"))
            elif self.showSimClusters == "merged":
                data.append(helpers.drawSimClusters(self.mergedSimClusters, "MergedSimCluster", self.simClusters))

        fig = go.Figure(data = data, layout = layout)

        if angle is not None:
            ax.view_init(30, angle)
            name += "_angle%i.png" % angle
            dpi = 100
        outfile = "/".join([self.outpath, name])+".html"
        fig.write_html(outfile)
        logging.info("Wrote file %s" % outfile)
        out = fig.to_html(full_html=True, include_plotlyjs='cdn')
        if False:
            with open("test.html", "w") as f:
                f.write(out)
        return out

def parseArgs():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--events", type=str, nargs='*', required=True, help="Which event to plot")
    parser.add_argument("-o", "--outfile", type=str, required=True, help="Name of output file")
    parser.add_argument("-p", "--path", type=str, default="/eos/user/k/kelong/www/ML4Reco/PFParticleValidation", help="output path")
    parser.add_argument("-s", "--subfolder", type=str, default="", help="append folder to output path")
    parser.add_argument("-m", "--movie", action='store_true', help="Produce scan in angle that can be made into a movie")
    parser.add_argument("--noTrackHits", action='store_true', help="Don't show hits in tracker")
    parser.add_argument("--tracking", action='store_true', help="Don't show tracking particles")
    parser.add_argument("--tracks", action='store_true', help="Don't show reco tracks")
    parser.add_argument("--caloPart", action='store_true', help="Show calo particles")
    parser.add_argument("--ticl", action='store_true', help="Show TICL cands")
    parser.add_argument("--pf", action='store_true', help="Show PF cands")
    parser.add_argument("--gen", action='store_true', help="Don't show gen particles")
    parser.add_argument("--rechits", action='store_true', help="Show RecHits instead of SimHits")
    parser.add_argument("--colorBy", choices=['pdgId', 'pfcand', 'mergedsimclus', 'simclus', 'calopart'], 
                        default="pdgId", help="Color calo hits by pdgId or SimCluster association")
    parser.add_argument("--simclus", choices=['none', 'merged', 'default', ], 
                        default="none", help="Draw (or not) merged or default simClusters")
    parser.add_argument("--simtracks", action='store_true', help="show simtracks")
    parser.add_argument("--default", action='store_true', help="Use default (not fine calo) input")
    parser.add_argument("-j", "--nCores", type=int, default=0, help="Number of cores, currently only for movie making")
    parser.add_argument("-c", "--cut", type=float, default=1, help="Tracking particle pt cut")
    parser.add_argument("-f", "--inputFile", type=str, help="Input ROOT file, in NanoML format",
        default="/home/kelong/work/ML4Reco/CMSSW_11_2_0_pre9/src/production_tests/Gun5Part_E15To500GeV_seed0_nano.root")

    args = parser.parse_args()
    if len(args.events) == 1 and ":" in args.events[0]:
        args.events = range(*[int(i) for i in args.events[0].split(":")])
    else:
        args.events = [int(i) for i in args.events]

    if args.movie and args.path == parser.get_default('path'):
        args.path = args.path.replace("www/", "")

    return vars(args)

def configureAndPlot(**kwargs):
    subfolder = kwargs.get("subfolder", "")
    path = kwargs.get("path", ".")
    plotter = HitsAndTracksPlotter(
        kwargs.get("inputFile"),
        path if not subfolder else "/".join([path, subfolder]))

    plotter.setNumThreads(kwargs.get("nCores", 1))
    plotter.setTrackingCut(kwargs.get("cut", 1))
    plotter.setShowGenPart(kwargs.get("gen", False))
    plotter.setRecHits(kwargs.get("rechits", False))
    plotter.setShowCaloPart(kwargs.get("caloPart", False))
    plotter.setShowTICL(kwargs.get("ticl", False))
    plotter.setShowPF(kwargs.get("pf", False))
    plotter.setShowTrackingPart(kwargs.get("tracking", False))
    plotter.setShowTracks(kwargs.get("tracks", False))
    plotter.setShowTrackHits(not kwargs.get("noTrackHits", False))
    plotter.setShowSimClusters(kwargs.get("simclus", "none"))
    plotter.setShowSimTracks(kwargs.get("simtracks", False))
    plotter.setColorBy(kwargs.get("colorBy", "pdgId"))

    outfile = kwargs.get("outfile", "temp")
    events = kwargs.get("events", [0])
    if not kwargs.get("movie", False):
        return plotter.makePlots(events, outfile)
    else:
        return plotter.makeMoviePlot(events, outfile)

def main():
    args = parseArgs()
    configureAndPlot(**args)

if __name__ == "__main__":
    main()
