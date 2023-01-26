import os
import pandas as pd
import random
from gmacc import config
from gmacc.nnsynth import preprocessing as GMpre


args = config.SyntheticDatabase(config_path='synthetic_database_config.yaml').get_config()
print(args)
outdir = os.path.join(args.outdir, args.sourcemode)

rawdata = pd.read_csv('%s/database.csv' % (outdir))

if args.sourcemode == 'MT':
    selectColumns = [
        'magnitude',
        'ev_depth',
        'strike',
        'dip',
        'rake',
        'src_duration',
        'azimuth',
        'rhypo',
    ]

else:
    selectColumns = [
        'magnitude',
        'ev_depth',
        'strike',
        'dip',
        'rake',
        'width',
        'length',
        'nucleation_x',
        'nucleation_y',
        'src_duration',
        'azimuth',
        'rup_azimuth',
        #'centre_azimuth',
        'rrup',
        'rhypo',
        # 'rjb',
        # 'rx',
        # 'ryo',
    ]
print(selectColumns)

sortcol = 'evID'
#chas = ['Z', 'E', 'N', 'H']
#chas = ['Z' , 'H']
chas = ['Z']
#gms = ['pga', 'pgv', 'pgd']
gms = ['pgd']
#gms = ['pga', 'pgv', 'pgd', 'sigdur', 'ai', 'vi', 'f_0.01', 'f_0.05', 'f_0.1', 'f_0.2', 'f_0.5', 'f_1.0']
targets = []
for cha in chas:
    for gm in gms:
        
        if '%s_%s' % (cha, gm) in rawdata:
            targets.append('%s_%s' % (cha, gm))

allSelectCol = selectColumns + [sortcol] + targets
print(targets)
print(allSelectCol)

data = rawdata[allSelectCol]

evIDs = list(set(data['evID']))
datafac = 1
print(datafac, type(datafac))
print(len(evIDs), type(len(evIDs)))
print(datafac * len(evIDs))
random.seed(0)
evIDs = random.sample(evIDs, k=int(datafac * len(evIDs)))

data = data[data['evID'].isin(evIDs)]
print(data)

data = GMpre.calc_azistrike(data, strikecol='strike', 
    azimuthcol='azimuth', azistrikecol='azistrike', delete=False)
dropcols = ['azimuth', 'strike']

if args.sourcemode == 'RS':
    data = GMpre.calc_azistrike(data, strikecol='strike',
        azimuthcol='rup_azimuth', azistrikecol='rup_azistrike', delete=False)
    dropcols.append('rup_azimuth')
data = data.drop(columns=dropcols)

data = GMpre.convert_distances(data)

data, scalingDict = GMpre.normalize_data(data, sortcol)
print(data)
print(list(data.columns))
xTrain, yTrain, xTest, yTest, xEval, yEval = GMpre.create_subsets(data,
                            targets, sortcol,
                            test_percent=0, eval_percent=0.2)

# outdir += '/%s' % datafac
outdir += '/%s_pgd' % datafac
#outdir += '/%s_ZH_peak' % datafac


if not os.path.exists(outdir):
    os.makedirs(outdir)

del data
del rawdata

filecore = os.path.join(outdir, 'data')
GMpre.write_subsets(filecore, xTrain, yTrain, xTest, yTest, xEval, yEval,
                    scalingDict, targets, filetype='pickle')
