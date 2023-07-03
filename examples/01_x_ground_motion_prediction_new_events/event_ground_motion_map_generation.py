import os
import sys
import time
import shutil
import numpy as num

import matplotlib.pyplot as plt

import gmacc.gmeval.plot as GMplt
import gmacc.gmeval.observation as GMobs
import gmacc.nnsynth.neural_networks as GMnn
import gmacc.gmeval.util as GMu
import gmacc.gmeval.inout as GMio

from gmacc import config
args = config.GroundMotionData(config_path='example_event_config.yaml').get_config()

print(sys.argv)
#############################
### General Settings
#############################
# eventfile = 'panama.xml'
faultfile = ''
eventfile = sys.argv[1]

### if waveform data should be deleted, important if interested in plotting waveform
deletewf = True
# deletewf = False
appendix = ''

args.outdir = os.path.join(args.outdir, eventfile.rsplit('.')[0])
print(args.outdir)
if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

# args.outdir = '%s/%s/' % (args.outdir, args.sourcemode)

if not os.path.exists(args.outdir):
    os.mkdir(args.outdir)

neweventfile = os.path.join(args.outdir, 'event.xml')
shutil.move(eventfile, neweventfile)

#############################
### Processing
#############################


source = GMio.get_event_data(neweventfile)

if source is None:
    print('No source found. Exit.')
    exit()

if args.sourcemode in ['RS', 'PDR']:
    source = GMio.get_finte_fault_data(faultfile, source)
print(source)

#############################
### Generate Map points/coords
#############################
if args.mapmode == 'random':
    coords = GMu.quasirandom_mapping(source, args.mapextent, args.mappoints,
        rmin=0.)
elif args.mapmode == 'rectangular':
    coords = GMu.rectangular_mapping(source, args.mapextent, args.mappoints,
        rmin=0.)
elif args.mapmode == 'circular':
    coords = GMu.circular_mapping(source, args.mapextent, args.mappoints,
        rmin=0.1)

#############################
### Calculation of Synthetic WV with Pyrocko
#############################
# method = 'pyrocko'
method = 'nn'
t1 = time.time()
if method == 'pyrocko':
    evCont = GMobs.get_pyrocko_container(source, coords,
                            args.comps, args.imts, args.freqs,
                            ['None', 0.5], delete=False,
                            gfpath=args.gf,
                            deleteWvData=deletewf, resample_f=20)

elif method == 'nn':
    modelfile = 'NN_MT/model.h5'
    suppfile = 'NN_MT/scalingdict.bin'
    evCont = GMnn.get_NNcontainer(source, modelfile, suppfile, coords, targetsMain=None)
else:
    print('Wrong method: %s' % method)
    exit()

print(len(coords), time.time() - t1, (time.time() - t1) / len(coords))
# exit()


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


# GMplt.plot_gm_map(predCont=evCont,  # obsCont=stationCont,
#                 predPlotMode='scatter',
#                 minmax=True, showcbar=showcbar,
#                 mapextent=args.mapextent, figtitle=figtitle,
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%s/truemap" % (args.outdir))

# GMplt.plot_gm_map(predCont=evCont,
#                 predPlotMode='area',
#                 # predPlotMode='contour',
#                 # predPlotMode='tiles',
#                 # minmax=True,
#                 showcbar=showcbar,
#                 mapextent=args.mapextent, figtitle=figtitle,
#                 plotgmvise=plotgmvise, plotindi=plotindi,
#                 # savename=[])
#                 savename="%s/predmap" % (args.outdir))

print('predmap_tiles')
GMplt.plot_gm_map(predCont=evCont,
                # predPlotMode='area',
                predPlotMode='tiles',
                minmax=True, showcbar=showcbar,
                mapextent=args.mapextent, figtitle=figtitle,
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                savename="%s/predmap_tiles" % (args.outdir))

GMplt.plot_gm_map(predCont=evCont,
                # predPlotMode='area',
                predPlotMode='tiles',
                minmax=True, showcbar=showcbar,
                mapextent=args.mapextent, figtitle=figtitle,
                plotgmvise=plotgmvise, plotindi=plotindi,
                # savename=[])
                valmode='true',
                savename="%s/predmap_tiles_true" % (args.outdir))

## Putting everything in one PDF
import matplotlib.backends.backend_pdf
outputstr = "%s/%s_full.pdf" % (args.outdir, method)
pdf = matplotlib.backends.backend_pdf.PdfPages(outputstr)
for fig in range(1, plt.gcf().number + 1):
    pdf.savefig(fig)
pdf.close()

print('Finished Plotting')
