import os
import time
import numpy as num

import matplotlib.pyplot as plt
from matplotlib import rc

from pyrocko import io
from pyrocko import moment_tensor as pmt

import ewrica.gm.sources as GMs
import ewrica.gm.gmpe as GMgmpe
import ewrica.gm.util as GMu

# import ewrica.gm.misc as GMm

import plot as GMTp
import observation as GMTobs
import inout as GMTio

#############################
### General Settings
#############################

# eq = 'Norcia'
# eq = 'Kumamoto'
# eq = 'Ridgecrest'
# eq = 'Iwate'
eq = 'Napa'
# eq = 'Ussita'
# eq = 'Miyagi'
# eq = 'Amatrice'
# eq = 'misc'

font = {
        # 'weight' : 'bold',
        'size': 14}
rc('font', **font)

appendix = ''

# filterfreq = None
filterfreq = 0.5
# filterfreq = [0.2, 0.5]

# mapextend = [2., 2.]
# mapextend = [1.75, 1.75]
# np = 30
mapextend = [.2, .2]
np = 3
imts = ['pga', 'pgv', 'pgd']
# imts = []
# imts = ['pgd']
# imts = ['pgv']
# imts = ['pga']
# imts = ['pgv', 'pgd']
freqs = []
# imts = ['SA_0.1', 'SA_1.0', 'SA_3.0', 'SA_5.0']
# freqs = [0.01, 0.1, 0.2, 0.5, 1.0]
pyrockoChas = ['Z', 'E', 'N']
# pyrockoChas = ['E', 'N']
# pyrockoChas = ['Z']
h2 = True
# h2 = False


# pdr = True
pdr = False

# mt = True
mt = False

delete = True
# delete = False
ratios = False

if filterfreq is None:
    filterfreq = ''

############################
### Norcia
############################
if eq == 'Norcia':
    datadir = '/home/lehmann/dr/example_eq/Norcia2016'

    # ending = 'fault_esm2.txt'
    ending = 'srcmod.fsp'

    # ngmpes = ['KothaEtAl2020', 'ChiouYoungs2014Italy', 'ChiouYoungs2014NearFaultEffect']
    # ngmpes = ['KothaEtAl2020']
    ngmpes = []

############################
### Kumamoto
############################
elif eq == 'Kumamoto':
    datadir = '/home/lehmann/dr/example_eq/Kumamoto2016'

    # ending = 'fault_usgs.txt'
    # ending = 'finit_fault_usgs.fsp'
    # ending = 'hayes.fsp'
    ending = 'yagi.fsp'

    ngmpes = []
    # ngmpes = ['ChiouYoungs2014Japan']
    # ngmpes = ['ChiouYoungs2014Japan', 'BooreEtAl2014JapanBasinNoSOF'] #, 'ChiouYoungs2014NearFaultEffect']

############################
### Ridgecrest
############################
elif eq == 'Ridgecrest':

    datadir = '/home/lehmann/dr/example_eq/Ridgecrest2019'

    # ending = 'fault_usgs.txt'
    ending = 'finite_fault_usgs.fsp'

    ngmpes = []

############################
### Iwate
############################
elif eq == 'Iwate':

    datadir = '/home/lehmann/dr/example_eq/Iwate2008/'

    ending = 'srcmod_tr.fsp'
    # ending = 'srcmod.fsp'

    ngmpes = []

############################
### Napa
############################
elif eq == 'Napa':

    # datadir = '/home/lehmann/dr/CESMD_wv/NC72282711'
    datadir = '/home/lehmann/dr/example_eq/Napa2014'

    ending = 'srcmod.fsp'

    ngmpes = []

############################
### Ussita
############################
elif eq == 'Ussita':

    # datadir = '/home/lehmann/dr/ESM_wv/EMSC-20161026_0000095'
    datadir = '/home/lehmann/dr/example_eq/Ussita2016'

    ending = 'srcmod.fsp'

    ngmpes = []

############################
### Miyagi
############################
elif eq == 'Miyagi':

    # datadir = '/home/lehmann/dr/gmprocess_jpn_wv/usp000c3bu'
    datadir = '/home/lehmann/dr/example_eq/Miyagi2003'

    ending = 'srcmod.fsp'

    ngmpes = []

############################
### Amatrice
############################
elif eq == 'Amatrice':

    datadir = '/home/lehmann/dr/example_eq/Amatrice2016'

    ending = 'srcmod.fsp'

    ngmpes = []

############################
### Misc
############################
elif eq == 'misc':

    # datadir = '/home/lehmann/dr/gmprocess_jpn_wv/official20030925195006360_27'
    datadir = '/home/lehmann/dr/sm_data/CESMD_wv/ci38457487'
    # datadir = '/home/lehmann/dr/gmprocess_jpn_wv/usp0007zev'
    # datadir = '/home/lehmann/dr/ESM_wv/EMSC-20161026_0000095'

    # ending = 'fault_usgs.txt'
    ending = 'srcmod.fsp'

    ngmpes = []


print('\n\n', eq, '\n\n')

#############################
#############################
#############################
### Processing
#############################

if type(filterfreq) == list:
    filterfreqstr = '%s_%s' % (filterfreq[0], filterfreq[1])
else:
    filterfreqstr = filterfreq

if mt:
    appendix = 'mt%s_%s' % (filterfreqstr, appendix) 
elif pdr:
    appendix = 'PDR%s_%s' % (filterfreqstr, appendix) 
else:
    appendix = 'rect%s_%s' % (filterfreqstr, appendix) 

# gmpes = []
# gmpe_list = get_available_gsims()
# for g in ngmpes:
#     gmpes.append(gmpe_list[g]())

gmpes = GMgmpe.get_gmpes_by_list(ngmpes)


## Waveform
if eq in ['Norcia', 'Amatrice', 'Ridgecrest']:
    fileMseed = '%s/wv_CV.mseed' % (datadir)
else:
    fileMseed = '%s/wv.mseed' % (datadir)

if os.path.exists(fileMseed) and not os.stat(fileMseed).st_size < 1:
    wvData = io.load(fileMseed)
else:
    print('No MSEED data available or empty')
    exit()

# eventFile = '%s/event.xml' % (datadir)
# source = GMTobs.convert_quakeml_to_source(eventFile)


for eventFile in ['event_ugsgs.json', 'event.xml', 'event_usgs.xml',
                  'event_esm.xml', 'event_small.xml']:
    if os.path.exists('%s/%s' % (datadir, eventFile)):
        print('Try %s' % eventFile)
        source = GMTio.convert_quakeml_to_source('%s/%s' % (datadir, eventFile))
        if source:
            if not source.strike and not source.tensor:
                with open('%s/%s' % (datadir, eventFile)) as f:
                    if 'strike' in f.read():
                        GMm.pwarning('Strike or MT in XML!!')
            else:
                break

if hasattr(source, 'moment'):
    if source.moment:
        newmag = float(pmt.moment_to_magnitude(source['moment']))
        if abs(source.magnitude - newmag) > 0.5:
            print('Moment wrong, newmag would be: %0.2f; oldmag %s' % (newmag, source.magnitude))
            print(source.moment)
            source.moment = None

        else:
            # print(source.magnitude, pmt.moment_to_magnitude(source['moment']))
            source.magnitude = newmag

if mt is False:
    faultFile = datadir + '/' + ending
    if os.path.exists(faultFile) and not os.stat(faultFile).st_size < 1:
        if 'srcmod.fsp' in ending:
            source = GMTio.read_fsp(faultFile, source)
            print('Fault from SRCMOD')

        elif 'fault_esm.txt' == ending or 'fault_esm2.txt' == ending:
            corners = GMTio.read_esm_faultfile(faultFile)
            source.rupture = corners
            surface = GMTio.corners2surface(source, corners)
            if surface:
                source = GMTio.convert_surface(source, surface)
            print('Fault from ESM')

        elif '.fsp' in ending:
            source = GMTio.read_fsp(faultFile, source)
            print('Fault from USGS finite fault')

        elif 'fault_usgs.txt' == ending:
            corners = GMTio.read_usgs_faultfile(faultFile)
            source.rupture = corners
            surface = GMTio.corners2surface(source, corners)
            if surface:
                source = GMTio.convert_surface(source, surface)
            print('Fault from USGS')

    source.create_rupture_surface()
    
    if source.risetime is None or source.risetime < 1.:
        source.risetime = 1.
print(source)
source.validate()


# print(source)

## Station info
locationFile = '%s/location.txt' % (datadir)
invFile = '%s/inv.xml' % (datadir)

if eq in ['Norcia', 'Amatrice', 'Ridgecrest']:
    invDir = '%s/inv_CV' % (datadir)
else:
    invDir = '%s/inv' % (datadir)

if os.path.exists(invDir):
    locationDict = GMTio.create_locationDict_from_invdir(invDir)
    print('Read from inventory directory')

elif os.path.exists(invFile):
    locationDict = GMTio.create_locationDict_from_xml(invFile)
    print('Read from inventory xml')

elif os.path.exists(locationFile):
    locationDict = GMTio.create_locationDict(locationFile)
    print('Read from location txt')

else:
    print('No file/directory with station/inventory locations is present')
    exit()


# print(source)
# exit()

### Station Container
stationCont = GMTobs.get_observation_container(source, wvData, locationDict,
                mapextend, pyrockoChas, filterfreq, imts, freqs, deleteWvData=False,
                savepath='%s/output/%swaveform' % (datadir, appendix),
                resample_f=100)

# for stations in stationCont.stations:
#   # print('\'%s\',' % stations)
#   print(stationCont.stations[stations].components['Z'].traces)
# exit()

# stationCont.calc_azimuths()
# stationCont.calc_rupture_azimuths()
# for sta in stationCont.stations:
#     print('%s: %3.f %3.f %3.f %3.f' %(sta,
#         stationCont.refSource.strike,
#         stationCont.stations[sta].azimuth,
#         stationCont.stations[sta].centre_azimuth,
#         stationCont.stations[sta].rup_azimuth))
# exit()
if ratios:
    stationCont.calc_peak_ratios(delete=False)
    stationCont.calc_freq_ratios(delete=False)

#############################
### Generate Map points/coords
#############################
coordinates = []
# GMm.circular_mapping(
mapCoords = GMu.rectangular_mapping(source, mapextend, np, rmin=0)
# mapCoords = GMm.circular_mapping(source, mapextend, np, rmin=0.1)
coordinates.append(mapCoords)

dataCoords = []
for sta in stationCont.stations:
    lat = stationCont.stations[sta].lat
    lon = stationCont.stations[sta].lon
    dataCoords.append([lon, lat])

dataCoords = num.array(dataCoords)
coordinates.append(dataCoords)

#############################
### Calculation of GM
#############################
for coords, mode in zip(coordinates, ['map', '1D']):
    print()
    print(mode, len(coords))

    #############################
    ### Calculation of Synthetic WV with Pyrocko
    #############################

    if pdr is True:
        source.form = 'pdr'

    if mt is True:
        source.form = 'point'

    t1 = time.time()
    pyrockoCont = GMTobs.get_pyrocko_container(source, coords,
                                pyrockoChas, imts, freqs, filterfreq,
                                delete=delete, deleteWvData=False,
                                resample_f=20)

    print(mode, len(coords), time.time() - t1, (time.time() - t1) / len(coords))

    if ratios:
        pyrockoCont.calc_peak_ratios(delete=False)
        pyrockoCont.calc_freq_ratios(delete=False)

    # print(pyrockoCont)
    #############################
    ### Calculation of GMPE with openquake
    #############################
    openquakeCont = GMgmpe.get_openquake_container(source, gmpes,
                                coords, imts, freqs)
    if ratios:
        openquakeCont.calc_peak_ratios(delete=False)
        openquakeCont.calc_freq_ratios(delete=False)

    #############################
    ### Merge StationContainers
    #############################
    # mergeCont = pyrockoCont
    mergeCont = GMs.merge_StationContainer(pyrockoCont, openquakeCont)
    # mergeCont = openquakeCont
    # mergeCont.validate()
    # print(mergeCont)
    # exit()

    outputdir = datadir + '/output/'

    if mode == 'map':
        mergeContMap = mergeCont

        # GMp.synth_peak_orientation_plot(staCont=pyrockoCont, imts=imts,
        #                               savepath=outputdir + 'synth_')

    elif mode == '1D':
        mergeContData = mergeCont

        if delete:
            pltcomps = ['Z']
        else:
            pltcomps = pyrockoChas
            
        # GMTp.waveform_comparison(outputdir, source,
        #             pyrockoCont, stationCont, pltcomps, appendix,
        #             mode='acc')
        # GMTp.waveform_comparison(outputdir, source,
        #             pyrockoCont, stationCont, pltcomps, appendix,
        #             mode='vel')
        # GMTp.waveform_comparison(outputdir, source,
        #             pyrockoCont, stationCont, pltcomps, appendix,
        #             mode='disp')

        # exit()

        # GMTp.waveform_comparison(outputdir, source,
        #           pyrockoCont, stationCont, pltcomps, appendix,
        #           mode='acc', spectrum=True)
        # GMTp.waveform_comparison(outputdir, source,
        #           pyrockoCont, stationCont, pltcomps, appendix,
        #           mode='vel', spectrum=True)
        # GMTp.waveform_comparison(outputdir, source,
        #           pyrockoCont, stationCont, pltcomps, appendix,
        #           mode='disp', spectrum=True)

    else:
        print('wrong mode')
        exit()
# exit()
#############################
### Plotting
#############################
print('\nStarting to Plot')
exit()

# plotgmvise = True
plotgmvise = False
# plotindi = True
plotindi = False
showcbar = True
# showcbar = False
outfile = '%s/output/%s' % (datadir, appendix)

if eq == 'Kumamoto':
    outfile += 'yagi_'
    # outfile += 'hayes_'

# if args.gmpes:
#     appendix = 'GMPE'
# if args.s_wv:
#     appendix = 'Pyrocko' + str(appendix).replace('.', '-')

# GMTp.plot_gm_map_normalized(predCont=mergeContMap, # obsCont=stationCont,
#                     predPlotMode='area', # cmapname='gist_stern_r',
#                     # mapextent=mapextend,
#                     # smoothfac=2, 
#                     savename='/home/lehmann/dr/plots/%s_gmpe_pyrocko' %(eq))
# exit()
figtitle = ''

# GMTp.plot_gm_map_indi(predCont=stationCont,  # obsCont=stationCont,
#               predPlotMode='area', mapextent=mapextend, figtitle=figtitle,
#               # savename=[])
#               savename="%snew" % (outfile))

# exit()


GMTp.plot_gm_map(predCont=stationCont,  # obsCont=stationCont,
                predPlotMode='scatter',
                minmax=True, showcbar=showcbar,
                mapextent=mapextend, figtitle=figtitle,
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%struemap" % (outfile))
# GMTp.plot_gm_map(predCont=mergeContMap,
#                 predPlotMode='area',
#                 # predPlotMode='contour',
#                 # predPlotMode='tiles',
#                 # minmax=True,
#                 showcbar=showcbar,
#                 mapextent=mapextend, figtitle=figtitle,
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%spredmap" % (outfile))
# print('predmap_tiles')
GMTp.plot_gm_map(predCont=mergeContMap,
                # predPlotMode='area',
                predPlotMode='tiles',
                minmax=True, showcbar=showcbar,
                mapextent=mapextend, figtitle=figtitle,
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%spredmap_tiles" % (outfile))

# GMTp.plot_gm_map(predCont=mergeContMap, obsCont=stationCont,
#                 # predPlotMode='area',
#                 predPlotMode='tiles',
#                 minmax=True, showcbar=showcbar,
#                 mapextent=mapextend, figtitle=figtitle,
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%smap" % (outfile))

GMTp.plot_gm_map(predCont=mergeContMap, obsCont=stationCont,
                mapextent=mapextend, figtitle=figtitle,
                resCont=mergeContData,
                showcbar=showcbar,
                predPlotMode='resScatter',
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%sresscatter" % (outfile))
# exit()
# GMTp.plot_gm_map(predCont=mergeContMap, obsCont=stationCont,
#                 mapextent=mapextend, figtitle=figtitle,
#                 resCont=mergeContData,
#                 showcbar=showcbar,
#                 predPlotMode='resArea',
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%sresmap" % (outfile))

GMTp.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='distance', distType='hypo',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%sdist_hypo" % (outfile))

GMTp.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='azimuth',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%sazi" % (outfile))

if mt is not True:
    GMTp.plot_1d(obsCont=stationCont, resCont=mergeContData,
                figtitle=figtitle,
                mode='distance', distType='rrup',
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%sdist_rup" % (outfile))
    GMTp.plot_1d(obsCont=stationCont, resCont=mergeContData,
                figtitle=figtitle,
                mode='distance', distType='rjb',
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%sdist_rjb" % (outfile))

    GMTp.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='azimuth', aziType='rup',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%sazi_rup" % (outfile))

    GMTp.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='azimuth', aziType='centre',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%sazi_centre" % (outfile))




# GMTp.plot_gm_map_indi(predCont=stationCont,  # obsCont=stationCont,
#               predPlotMode='scatter', mapextent=mapextend, figtitle=figtitle,
#               # savename=[])
#               savename="%struemap" % (outfile))
# GMTp.plot_gm_map_indi(predCont=mergeContMap, obsCont=stationCont,
#               predPlotMode='area', mapextent=mapextend, figtitle=figtitle,
#               # savename=[])
#               savename="%smap" % (outfile))
# GMTp.plot_gm_map_indi(predCont=mergeContMap, obsCont=stationCont,
#               mapextent=mapextend, figtitle=figtitle,
#               resCont=mergeContData,
#               predPlotMode='resScatter',
#               # savename=[])
#               savename="%sresscatter" % (outfile))
# GMTp.plot_gm_map_indi(predCont=mergeContMap, obsCont=stationCont,
#               mapextent=mapextend, figtitle=figtitle,
#               resCont=mergeContData,
#               predPlotMode='resArea',
#               # savename=[])
#               savename="%sresmap" % (outfile))

# GMTp.plot_1d_indi(obsCont=stationCont, resCont=mergeContData,
#           figtitle=figtitle,
#           mode='distance', distType='rrup', gmpename=ngmpes[0],
#           # savename=[])
#           savename="%sdist_rup" % (outfile))
# GMTp.plot_1d_indi(obsCont=stationCont, resCont=mergeContData,
#           figtitle=figtitle,
#           mode='distance', distType='hypo', gmpename=ngmpes[0],
#           # savename=[])
#           savename="%sdist_hypo" % (outfile))
# GMTp.plot_1d_indi(obsCont=stationCont, resCont=mergeContData,
#           figtitle=figtitle,
#           mode='distance', distType='rjb', gmpename=ngmpes[0],
#           # savename=[])
#           savename="%sdist_rjb" % (outfile))
# GMTp.plot_1d_indi(obsCont=stationCont, resCont=mergeContData,
#           figtitle=figtitle,
#           mode='azimuth', gmpename=ngmpes[0],
#           # savename=[])
#           savename="%sazi" % (outfile))

# exit()

## Putting everything in one PDF
import matplotlib.backends.backend_pdf
outputstr = "%sfull.pdf" % (outfile)
pdf = matplotlib.backends.backend_pdf.PdfPages(outputstr)
for fig in range(1, plt.gcf().number + 1):
    pdf.savefig(fig)
pdf.close()

print('Finished Plotting')
