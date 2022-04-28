import numpy as num
import pickle
import time

from pyrocko import gf
from pyrocko import moment_tensor as pmt

import matplotlib.backends.backend_pdf
import matplotlib.pyplot as plt

from gmacc import config
from gmacc.nnsynth import neural_networks as GMnn

from gmacc.gmeval import observation as GMEobs
from gmacc.gmeval import plot as GMEplt

import ewrica.gm.sources as GMs
import ewrica.gm.util as GMu

nnargs = config.NeuralNetwork(config_path='newsamples_nn_config.yaml').get_config()
sdargs = config.SyntheticDatabase(config_path='newsamples_sd_config.yaml').get_config()

spm = 1  # sources per magnitudes
nsources = 2
imags = num.linspace(5, 7.5, nsources)
# imags = [4.5, 4.75, 5.0, 7.5, 7.75, 8.0]


allrefsourcetime = 0.
allpysourcetime = 0.
allnntime = 0.
nnDiff = {}
diffs = {}
dists = {}
timemags = []
mags = []
nntimes = []
pytimes = []

oneloctimesdict = {}
oneloctimesnndict = {}

# print(imags)
# exit()
for ii in range(nsources):

    spms = {}
    for mm in range(spm):
        print(ii)

        # num.random.seed(seed=ii * 100)

        mag = float(imags[ii])
        strike = float(num.random.randint(0., 360.))
        dip = float(num.random.randint(1, 89))
        rake = float(num.random.randint(-180, 180))
        depth = num.random.uniform(0.1, 10.)

        lat = 0.
        lon = 0.

        if sdargs.sourcemode == 'RS':
            nucfac = 9
            nucx = float(num.random.choice(num.linspace(-1, 1, nucfac)))  # (-1 = left edge, +1 = right edge)
            nucy = float(num.random.choice(num.linspace(-1, 1, nucfac)))  # (-1 = upper edge, +1 = lower edge)
            width, length = GMu.calc_source_width_length(mag, mode='Blaser', rake=rake)
            
            ### point of top_left
            anchor = 'top_left'

            src = gf.RectangularSource(time=0.,
                                lat=lat, lon=lon, depth=depth * 1000., anchor=anchor,
                                strike=strike, dip=dip, rake=rake,
                                width=width * 1000., length=length * 1000.,
                                nucleation_x=nucx,
                                nucleation_y=nucy,
                                # decimation_factor=1,
                                magnitude=mag)
            src.validate()
            source = GMs.from_rectsource_to_own_source(src)
            source.create_rupture_surface()
            
        elif sdargs.sourcemode == 'MT':
            mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mag)

            src = gf.MTSource(lat=lat, lon=lon,
                        depth=depth * 1000.,
                        mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
                        mne=mt.mne, mnd=mt.mnd, med=mt.med)
            src.validate()

            source = GMs.SourceClass(
                    name='Synthetic_%s' %(ii),
                    form='point',
                    time=src.time,
                    lon=float(lon),  # hypo
                    lat=float(lat),  # hypo
                    depth=float(depth),  # hypo
                    magnitude=float(mag),
                    strike=float(strike),
                    dip=float(dip),
                    rake=float(rake),
                    tensor=dict(mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
                        mne=mt.mne, mnd=mt.mnd, med=mt.med))

        source.update(name='%s_%s' % (source.name, ii))
        source.validate()

        if sdargs.mapmode == 'random':
            coords = GMu.quasirandom_mapping(source, sdargs.mapextent, sdargs.mappoints, rmin=0.)
        elif sdargs.mapmode == 'rectangular':
            coords = GMu.rectangular_mapping(source, sdargs.mapextent, sdargs.mappoints, rmin=0.)
        elif sdargs.mapmode == 'circular':
            coords = GMu.circular_mapping(source, sdargs.mapextent, sdargs.mappoints, rmin=0.1)

        nc = len(coords)

        ###############
        ## PWS
        ###############
        t1 = time.time()
        pyCont = GMEobs.get_pyrocko_container(source, coords, sdargs.comps,
                    sdargs.imts, sdargs.freqs, resample_f=20,
                    deleteWvData=True, H2=sdargs.rotd100, gfpath=sdargs.gf)

        pytime = time.time() - t1
        pytimes.append(pytime)

        tpo = pytime / (nc**2)
        print('Time for %s location:' % nc**2, pytime)
        print('Time per one location:', tpo)

        if mag not in oneloctimesdict:
            oneloctimesdict[mag] = []
        oneloctimesdict[mag].append(tpo)

        ###############
        ## NN
        ###############
        print('\n\n')

        t2 = time.time()
        lons, lats = coords.T
        NNCont = GMnn.get_NNcontainer(source, nnargs.modeldir + '/MODEL.h5',
            '%s/%s_scalingdict.bin' % (nnargs.indir, nnargs.filecore), coords)
        nntime = time.time() - t2
        nntimes.append(nntime)

        tpo = nntime / (nc**2)
        print('Time for %s location:' % nc**2, nntime)
        print('Time per one location:', tpo)

        ###
        ## Analysis
        ###

        if mag not in oneloctimesnndict:
            oneloctimesnndict[mag] = []
        oneloctimesnndict[mag].append(tpo)

        # print(staDict)
        # for sta in staDict.keys():
        #     print(staDict[sta])

        # print(NNCont)
        # target = targets[0]
        pyVals = pyCont.get_gm_values()
        NNVals = NNCont.get_gm_values()
        shmmin = min(min(min(pyVals.values())), min(min(NNVals.values())))

        shmmax = max(max(max(pyVals.values())), max(max(NNVals.values())))

        print(shmmin, shmmax)
        # shmLevels = num.linspace(shmmin, shmmax, 25)
        dnum = 0.1
        shmLevels = num.arange(num.floor(shmmin * 100) / 100., dnum + num.ceil(shmmax * 100) / 100., dnum)
        # print(shmLevels)
        # exit()
        GMEplt.plot_gm_map(pyCont, mapextent=sdargs.mapextent,
                        savename='%s/nn_py_test_%s_%s_M%s_pyrocko' % (nnargs.modeldir, ii, mm, mag),
                        # minmax=True,
                        shmLevels=shmLevels,
                        # figtitle='Pyrocko\nTime: %.3f\n%.1f, %.1f, %.1f, %.2f' % (refsourcetime,
                        # source.strike, source.dip, source.rake, source.depth),
                        figtitle='PWS', figtitlesize=40.,
                        predPlotMode='area', smoothfac=0)

        GMEplt.plot_gm_map(NNCont, mapextent=sdargs.mapextent,
                        # minmax=True,
                        shmLevels=shmLevels,
                        savename='%s/nn_py_test_%s_%s_M%s_nn' % (nnargs.modeldir, ii, mm, mag),
                        # figtitle='NN\nTime: %.3f' % nntime,
                        figtitle='NN', figtitlesize=40.,
                        predPlotMode='area', smoothfac=0)

        GMEplt.plot_gm_map(NNCont, obsCont=pyCont, mapextent=sdargs.mapextent,
                        # minmax=True,
                        shmLevels=shmLevels,
                        savename='%s/nn_py_test_%s_%s_M%s_pyrocko_nn' % (nnargs.modeldir, ii, mm, mag),
                        # figtitle='NN\nTime: %.3f' % nntime,
                        figtitle='NN vs. PWS', figtitlesize=40.,
                        predPlotMode='area', smoothfac=0)

        GMEplt.plot_gm_map(NNCont, obsCont=pyCont, resCont=NNCont,
                        mapextent=sdargs.mapextent,
                        minmax=True,
                        savename='%s/nn_py_test_%s_%s_M%s_pyrocko_nn_residual' % (nnargs.modeldir, ii, mm, mag),
                        figtitle='NN - Pyrocko\nTimefac: %.3f' % (pytime / nntime),
                        predPlotMode='resArea', smoothfac=0)

        ## Putting everything in one PDF

        outfile = '%s/nn_py_test_%s_%s_M%s' % (nnargs.modeldir, ii, mm, mag)
        pdf = matplotlib.backends.backend_pdf.PdfPages("%s_full.pdf" % (outfile))
        for fig in range(1, plt.gcf().number + 1):
            pdf.savefig(fig)
        pdf.close()
        plt.close('all')

        refDict = pyCont.to_dictionary()
        nnDict = NNCont.to_dictionary()

        print(ii, mag)
        timemags.append('M%s' % (mag))
        if mm == 0:
            mags.append('M%s' % (mag))

        distTypes = ['rhypo']
        NNCont.calc_distances(distTypes=distTypes)
        distdict = NNCont.get_distances(distTypes=distTypes)
        for disttyp, vals in distdict.items():
            if disttyp not in dists:
                dists[disttyp] = []
            dists[disttyp].append(vals)

        chagms = []
        for refgm in refDict.keys():
            for refcomp in refDict[refgm].keys():

                refvals = num.array(refDict[refgm][refcomp]['vals'])
                nnvals = num.array(nnDict[refgm][refcomp]['vals'])
                nnd = nnvals - refvals

                chagm = '%s_%s' % (refcomp, refgm)
                if chagm not in diffs:
                    diffs[chagm] = []
                diffs[chagm].append(nnd)

                chagms.append(chagm)

                if mm not in spms:
                    spms[mm] = {}
                spms[mm][chagm] = list(nnd)

    for chagm in chagms:
        if chagm not in nnDiff:
            nnDiff[chagm] = []

        alllist = []
        for mm in range(spm):
            alllist = alllist + spms[mm][chagm]    
        
        nnDiff[chagm].append(list(alllist))

print('Ref    time:', allrefsourcetime, pytimes)
print('NN     time:', allnntime, nntimes)

refsourcetimes = num.array(pytimes)
nntimes = num.array(nntimes)

colorpws = 'black'
colornn = 'grey'


####
## Points per minute Plot
####
pymean = []
nnmean = []
for nn, key in enumerate(oneloctimesdict.keys()):
    # minnnvalidx = num.argmax(oneloctimesnndict[key])
    for ii in range(len(oneloctimesdict[key])):
        # if ii == minnnvalidx:
        #     continue
        plt.plot(key, 60 / oneloctimesdict[key][ii], color=colorpws, marker='x', label='PWS' if (ii == 0) and (nn == 0) else "")
        plt.plot(key, 60 / oneloctimesnndict[key][ii], color=colornn, marker='o', label='NN' if (ii == 0) and (nn == 0) else "")
    
    pymean.append(num.median(oneloctimesdict[key]))
    nnmean.append(num.median(oneloctimesnndict[key]))

pymean = num.array(pymean)
nnmean = num.array(nnmean)
plt.plot(list(oneloctimesdict.keys()), 60 / pymean, color=colorpws, linestyle=':')
plt.plot(list(oneloctimesnndict.keys()), 60 / nnmean, color=colornn, linestyle=':')

# 50/100 quellen a 50x50 punkte, ein beispiel als horizontale linie einfügen (Quellschätzung 4min, GMs eine Minute)
testcase = 50 * (50 * 50)
plt.axhline(testcase, linestyle='--', color='black', label='Test case: %1.0f' % testcase)

plt.ylabel('Number of points')
plt.xlabel('Magnitude')
plt.legend()
plt.yscale('log')
plt.tight_layout()
print('%s/locations_per_min.png' % nnargs.modeldir)
plt.savefig('%s/locations_per_min.png' % nnargs.modeldir)

####
## Absolute run time Plot
####
plt.figure(figsize=(12, 3))
plt.plot(timemags, refsourcetimes, 'r', marker='o', markersize=8, linestyle='', label='PWS')
plt.plot(timemags, nntimes, 'g', marker='*', markersize=8, linestyle='', label='NN')
plt.ylabel('Computational time [s]')
plt.xlabel('Magnitude')
plt.legend()
plt.yscale('log')
plt.tight_layout()
plt.savefig('%s/abs_time.png' % nnargs.modeldir)

plt.close('all')


####
## Save time values
####
import pickle
timedict = dict(mags=timemags, nn=nntimes, ref=refsourcetimes)
print(timedict)
pickle.dump(timedict, open('%s/timesdict.bin' % (nnargs.modeldir), 'wb'))

####
## Violinplot
####
for chagm in chagms:

    GMnn.violinplot(nnDiff[chagm], imags, imags, nnargs.modeldir, 
        fileprefix='%s2_' % chagm, xlabel='Magnitude', predirectory=False,
        points=20)

    fig = plt.figure()
    plt.plot(dists['rhypo'], diffs[chagm], 'ko', markersize=1)
    plt.axhline(y=0, color='black', linewidth=3, zorder=-1)
    plt.xlabel('rHypo [km]')
    plt.ylabel('Difference [log10]')
    plt.xscale('log')
    plt.grid(which='both', axis='y', color='black', linestyle=':', zorder=-1, alpha=0.5)
    fig.savefig('%s/nn_%s_dist.png' % (nnargs.modeldir, chagm))