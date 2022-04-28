import numpy as num

from gmacc import config
from gmacc.nnsynth import neural_networks as GMnn
from gmacc.nnsynth import preprocessing as GMpre

args = config.NeuralNetwork(config_path='neural_network_config.yaml').get_config()
print(args)

xEval, yEval, scalingDict, targets, inputcols = GMpre.read_evaluation_data('%s/%s' % (args.indir, args.filecore))

model = GMnn.load_model(args.modeldir + '/MODEL.h5')

print(model)

print(args.modeldir.rsplit('-')[-1])
if args.modeldir.rsplit('-')[-1] in ['multi', 'multi/']:
    pass
else:
    targets = [args.modeldir.rsplit('-')[-1]]

print(targets)
predDF = GMnn.get_predict_df(model, xEval, targets)

xEval = GMpre.normalize(scalingDict, xEval, mode='inverse')
yEval = GMpre.normalize(scalingDict, yEval, mode='inverse')
predDF = GMpre.normalize(scalingDict, predDF, mode='inverse')

xEval = GMpre.convert_distances(xEval, mode='inverse')

diffEval = yEval - predDF
for target in targets:
    boxdata = []
    labels = []
    x = yEval[target]
    dx = 0.5
    xranges = num.arange(num.floor(min(x)), num.ceil(max(x)) + dx, dx)
    for ii in range(len(xranges) - 1):
        cond = ((x > xranges[ii]) & (x < xranges[ii + 1]))
        y = diffEval[target][cond].values
        y = y[~num.isnan(y)]
        if len(y) == 0:
            continue
        boxdata.append(y)
        labels.append('%s-\n%s\n(%s)' % (xranges[ii], xranges[ii+1], len(y)))

    positions = range(len(boxdata))
    GMnn.violinplot(boxdata, positions, labels, args.modeldir, xlabel='GM - Amplitude',
        fileprefix='%s_' % target, points=200)

columns = ['magnitude', 'rhypo', 'ev_depth']
outdir = args.modeldir
GMnn.evaluate_gm_column(columns, predDF, xEval, yEval, targets, outdir,
    plotmode='violin', violinpoints=200)