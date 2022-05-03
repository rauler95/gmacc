import os
import time
import numpy as num

import matplotlib.pyplot as plt

# from pyrocko import io
# from pyrocko import moment_tensor as pmt

from gmacc import config
# import gmacc.gmeval.sources as GMs
import gmacc.gmeval.plot as GMplt
import gmacc.gmeval.observation as GMobs
import gmacc.gmeval.util as GMu
import gmacc.gmeval.inout as GMio
# import gmacc.nnsynth.preprocessing as GMpre

args = config.ObservationalData(config_path='example_event_config.yaml').get_config()
print(args)


#############################
### General Settings
#############################

### if waveform data should be deleted, important if interested in plotting waveform
# deletewf = True
deletewf = False
appendix = ''

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

args.outdir = '%s/%s/' % (args.outdir, args.sourcemode)

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

#############################
### Processing
#############################

# if type(filterfreq) == list:
#     filterfreqstr = '%s_%s' % (filterfreq[0], filterfreq[1])
# else:
#     filterfreqstr = filterfreq

# if mt:
#     appendix = 'mt%s_%s' % (filterfreqstr, appendix) 
# elif pdr:
#     appendix = 'PDR%s_%s' % (filterfreqstr, appendix) 
# else:
#     appendix = 'rect%s_%s' % (filterfreqstr, appendix) 


wvData = GMio.get_waveform_data(args.waveformpath)
source = GMio.get_event_data(args.eventfile)
print(source)
if args.sourcemode in ['RS', 'PDR']:
    source = GMio.get_finte_fault_data(args.faultfile, source)
    print(source)

locationDict = GMio.get_station_data(args.stationpath)


### Station Container
stationCont = GMobs.get_observation_container(source, wvData, locationDict,
                args.mapextent, args.comps, args.filterfrequencies,
                args.imts, args.freqs, deleteWvData=deletewf,
                resample_f=100, rmStas=args.stationstoremove)

#############################
### Generate Map points/coords
#############################
coordinates = []
if args.mapmode == 'random':
    mapCoords = GMu.quasirandom_mapping(source, args.mapextent, args.mappoints,
        rmin=0.)
elif args.mapmode == 'rectangular':
    mapCoords = GMu.rectangular_mapping(source, args.mapextent, args.mappoints,
        rmin=0.)
elif args.mapmode == 'circular':
    mapCoords = GMu.circular_mapping(source, args.mapextent, args.mappoints,
        rmin=0.1)

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

    t1 = time.time()
    pyrockoCont = GMobs.get_pyrocko_container(source, coords,
                                args.comps, args.imts, args.freqs,
                                args.filterfrequencies, delete=False,
                                deleteWvData=deletewf, resample_f=20)

    print(mode, len(coords), time.time() - t1, (time.time() - t1) / len(coords))

    # print(pyrockoCont)
    #############################
    ### Calculation of GMPE with openquake
    #############################
    # openquakeCont = GMgmpe.get_openquake_container(source, gmpes,
    #                             coords, imts, freqs)

    #############################
    ### Merge StationContainers
    #############################
    mergeCont = pyrockoCont
    # mergeCont = GMs.merge_StationContainer(pyrockoCont, openquakeCont)
    # mergeCont = openquakeCont
    # mergeCont.validate()
    # print(mergeCont)
    # exit()

    if mode == 'map':
        mergeContMap = mergeCont

        # GMp.synth_peak_orientation_plot(staCont=pyrockoCont, imts=imts,
        #                               savepath=outputdir + 'synth_')

    elif mode == '1D':
        mergeContData = mergeCont

        if not deletewf:                
            GMplt.waveform_comparison(args.outdir, source,
                        pyrockoCont, stationCont, args.comps, appendix,
                        mode='acc')
            GMplt.waveform_comparison(args.outdir, source,
                        pyrockoCont, stationCont, args.comps, appendix,
                        mode='vel')
            GMplt.waveform_comparison(args.outdir, source,
                        pyrockoCont, stationCont, args.comps, appendix,
                        mode='disp')

            ## Spectra
            # GMplt.waveform_comparison(args.outdir, source,
            #           pyrockoCont, stationCont, args.comps, appendix,
            #           mode='acc', spectrum=True)
            # GMplt.waveform_comparison(args.outdir, source,
            #           pyrockoCont, stationCont, args.comps, appendix,
            #           mode='vel', spectrum=True)
            # GMplt.waveform_comparison(args.outdir, source,
            #           pyrockoCont, stationCont, args.comps, appendix,
            #           mode='disp', spectrum=True)

    else:
        print('wrong mode')
        exit()

#############################
### Plotting
#############################
print('\nStarting to Plot')

# plotgmvise = True
plotgmvise = False
# plotindi = True
plotindi = False
showcbar = True
# showcbar = False

figtitle = ''


GMplt.plot_gm_map(predCont=stationCont,  # obsCont=stationCont,
                predPlotMode='scatter',
                minmax=True, showcbar=showcbar,
                mapextent=args.mapextent, figtitle=figtitle,
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%s/truemap" % (args.outdir))
# GMplt.plot_gm_map(predCont=mergeContMap,
#                 predPlotMode='area',
#                 # predPlotMode='contour',
#                 # predPlotMode='tiles',
#                 # minmax=True,
#                 showcbar=showcbar,
#                 mapextent=args.mapextent, figtitle=figtitle,
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%s/predmap" % (args.outdir))
# print('predmap_tiles')
GMplt.plot_gm_map(predCont=mergeContMap,
                # predPlotMode='area',
                predPlotMode='tiles',
                minmax=True, showcbar=showcbar,
                mapextent=args.mapextent, figtitle=figtitle,
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%s/predmap_tiles" % (args.outdir))

# GMplt.plot_gm_map(predCont=mergeContMap, obsCont=stationCont,
#                 # predPlotMode='area',
#                 predPlotMode='tiles',
#                 minmax=True, showcbar=showcbar,
#                 mapextent=args.mapextent, figtitle=figtitle,
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%s/map" % (args.outdir))

GMplt.plot_gm_map(predCont=mergeContMap, obsCont=stationCont,
                mapextent=args.mapextent, figtitle=figtitle,
                resCont=mergeContData,
                showcbar=showcbar,
                predPlotMode='resScatter',
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%s/resscatter" % (args.outdir))
# exit()
# GMplt.plot_gm_map(predCont=mergeContMap, obsCont=stationCont,
#                 mapextent=args.mapextent, figtitle=figtitle,
#                 resCont=mergeContData,
#                 showcbar=showcbar,
#                 predPlotMode='resArea',
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%s/resmap" % (args.outdir))

GMplt.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='distance', distType='hypo',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%s/dist_hypo" % (args.outdir))

GMplt.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='azimuth',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%s/azi" % (args.outdir))

if args.sourcemode not in ['mt', 'MT']:
    GMplt.plot_1d(obsCont=stationCont, resCont=mergeContData,
                figtitle=figtitle,
                mode='distance', distType='rrup',
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%s/dist_rup" % (args.outdir))
    GMplt.plot_1d(obsCont=stationCont, resCont=mergeContData,
                figtitle=figtitle,
                mode='distance', distType='rjb',
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%s/dist_rjb" % (args.outdir))

    GMplt.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='azimuth', aziType='rup',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%s/azi_rup" % (args.outdir))

    GMplt.plot_1d(obsCont=stationCont, resCont=mergeContData,
            figtitle=figtitle,
            mode='azimuth', aziType='centre',
            plotgmvise=plotgmvise, plotindi=plotindi,
            # savename=[])
            savename="%s/azi_centre" % (args.outdir))

## Putting everything in one PDF
import matplotlib.backends.backend_pdf
outputstr = "%s/full.pdf" % (args.outdir)
pdf = matplotlib.backends.backend_pdf.PdfPages(outputstr)
for fig in range(1, plt.gcf().number + 1):
    pdf.savefig(fig)
pdf.close()

print('Finished Plotting')
