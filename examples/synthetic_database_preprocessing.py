import pandas as pd

from gmacc import config
from gmacc.nnsynth import preprocessing as GMpre


args = config.SyntheticDatabase(config_path='synthetic_database_config.yaml').get_config()

args.outdir = '%s/%s/' % (args.outdir, args.sourcemode)

rawdata = pd.read_csv('%s/database.csv' % (args.outdir))
# print(rawdata.columns)

if args.sourcemode:
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
        'centre_azimuth',
        'rrup',
        'rhypo',
        # 'rjb',
        # 'rx',
        # 'ryo',
    ]

sortcol = 'evID'
# targets = ['Z_pga', 'Z_pgv', 'Z_pgd']
targets = []
for cha in ['Z', 'E', 'N', 'H']:
    for gm in ['pga', 'pgv', 'pgd',
               'sigdur', 'ai', 'vi',
               'f_0.01', 'f_0.05', 'f_0.1', 'f_0.2', 'f_0.5', 'f_1.0']:

        if '%s_%s' % (cha, gm) in rawdata:
            targets.append('%s_%s' % (cha, gm))

allSelectCol = selectColumns + [sortcol] + targets

data = rawdata[allSelectCol]
data = GMpre.calc_azistrike(data)
data = GMpre.convert_distances(data)
# data = GMpre.convert_magnitude_to_moment(data)

data, scalingDict = GMpre.normalize_data(data, sortcol)

xTrain, yTrain, xTest, yTest, xEval, yEval = GMpre.create_subsets(data,
                            rawdata, targets, sortcol,
                            test_percent=0, eval_percent=0.2)

del data
del rawdata

filecore = '%s/data' % (args.outdir)
GMpre.write_subsets(filecore, xTrain, yTrain, xTest, yTest, xEval, yEval,
                    scalingDict, targets)