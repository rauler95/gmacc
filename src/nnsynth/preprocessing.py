import random
import pickle

import numpy as num
import pandas as pd


def get_important_features(data, targets, sortcol, n=10, plotpath=False):
    from sklearn.ensemble import RandomForestRegressor
    import matplotlib.pyplot as plt

    xTrain = data.drop(columns=targets + [sortcol])
    yTrain = data[targets]

    # print(xTrain.columns)
    # print()
    # print(yTrain.columns)
    # exit()

    forest = RandomForestRegressor(n_estimators=10, random_state=0)
    forest.fit(xTrain, yTrain)

    ##### Random Forest
    importances = forest.feature_importances_

    forest_importances = pd.Series(importances, index=xTrain.columns)
    # print(forest_importances.sort_values(ascending=False))

    if plotpath:
        fig, ax = plt.subplots()
        std = num.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")  # mean decrease impurity
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()
        fig.savefig('%s/featureimportance_mdi.png' % plotpath)

    ##### Permutation
    from sklearn.inspection import permutation_importance

    result = permutation_importance(
        forest, xTrain, yTrain, n_repeats=10, n_jobs=5)

    forest_importances = pd.Series(result.importances_mean, index=xTrain.columns)

    if plotpath:
        fig, ax = plt.subplots()
        forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
        ax.set_title("Feature importances using permutation on full model")
        ax.set_ylabel("Mean accuracy decrease")
        fig.tight_layout()
        fig.savefig('%s/featureimportance_perm.png' % plotpath)

    forest_importances = forest_importances.sort_values(ascending=False)
    fi = list(forest_importances.keys()[:n])
    print('The most %s important features:\n' % n, fi)

    return fi


def calc_azistrike(data, strikecol='strike', azimuthcol='azimuth', azistrikecol='azistrike', delete=True):
    data.loc[data[azimuthcol] < 0, azimuthcol] = data[azimuthcol][data[azimuthcol] < 0] + 360
    data[azistrikecol] = data[azimuthcol] - data[strikecol]
    data.loc[data[azistrikecol] < 0, azistrikecol] = data[azistrikecol][data[azistrikecol] < 0] + 360

    if delete:
        data = data.drop(columns=['strike', 'azimuth'])

    return data


def convert_distances(data, mode=False):
    for col in data.columns:

        if mode in ['inverse', 'reverse']:
            if col in ['hypodist', 'rupdist', 'rjb', 'rrup', 'rhypo']:
                data[col] = 10 ** data[col]
        else:
            if col in ['hypodist', 'rupdist', 'rjb', 'rrup', 'rhypo']:
                data[col] = num.where(data[col] == 0.0, 0.01, data[col])
                data[col] = num.log10(data[col])

    return data


def convert_magnitude_to_moment(data):
    # Hanks and Kanamori 1979]
    data['moment'] = 10.0**(1.5 * (data['magnitude'] + 10.7)) * 1.0e-7

    return data


def standardize_frequencies(data, cols, scalingDict={}):
    vals = []
    for col in cols:
        vals += data[col].to_list()

    mean = num.mean(vals)
    std = num.std(vals)

    for col in cols:
        if col not in scalingDict:
            scalingDict[col] = {}
        data[col] = (data[col] - mean) / std
        scalingDict[col] = {'mean': mean, 'std': std}

    return data, scalingDict


def standardize_data(data, sortcol='', exceptions='', scalingDict={}):

    for col in data.columns:
        if col in exceptions:
            continue
        if col in sortcol or col == sortcol:
            continue

        if col not in scalingDict:
            scalingDict[col] = {}
        mean = num.mean(data[col])
        std = num.std(data[col])
        data[col] = (data[col] - mean) / std
        scalingDict[col] = {'mean': mean, 'std': std}

    return data, scalingDict


def standardize(scalingDict, data, mode='forward', verbose=False):
    ndata = {}
    for col in scalingDict.keys():
        if col not in data:
            if verbose is not False:
                print(col, 'not in Data')
            continue

        ndata[col] = standardize_column(scalingDict, data, col, mode, verbose)

    return pd.DataFrame(ndata)


def standardize_column(scalingDict, data, col, mode='forward', verbose=False):

    mean = scalingDict[col]['mean']
    std = scalingDict[col]['std']

    if mode == 'forward':
        ndata = (data[col] - mean) / std

    elif mode == 'inverse':
        ndata = mean + (data[col] * std)

    else:
        print('Wrong scaling mode')
        exit()

    return ndata


def normalize_data(data, sortcol='', exceptions='', scalingDict={}, extra=None):
    skipcol = []
    for col in data.columns:
        if col in exceptions:
            continue
        if col in sortcol or col == sortcol:
            continue

        valmin = num.min(data[col].values)
        valmax = num.max(data[col].values)

        if valmin == -num.inf:
            print(col, valmin, valmax)
            print(data[col])
            print(data[data[col] == -num.inf][col])
            print('FOUND INFINITY in column')
            exit()

        elif valmin == valmax:
            print('FOUND same values in whole column (%s), therefore, it is irrelevant and gets deleted.' % col)
            data.drop(columns=[col], inplace=True)
            continue

        if extra is not None:

            if extra in col:
                continue

            extracol = '%s%s' % (col, extra)
            print(extracol)
            if extracol in data:

                valmin = min(num.min(data[extracol]), valmin)
                valmax = max(num.max(data[extracol]), valmax)

                print(col, valmin, valmax)
                print(extracol, num.min(data[extracol]), num.max(data[extracol]))
                print(col, extracol, valmin, valmax)

                data[extracol] = (data[extracol] - valmin) / (valmax - valmin)
                scalingDict[extracol] = {'min': valmin, 'max': valmax}

                skipcol.append(extracol)

        try:
            data[col] = (data[col] - valmin) / (valmax - valmin)
            scalingDict[col] = {'min': valmin, 'max': valmax}   
        except TypeError as e:
            print('Error:\n', e)

    return data, scalingDict


def normalize(scalingDict, data, mode='forward', verbose=False):
    ndata = {}

    for col in scalingDict.keys():
        if col not in data:
            if verbose is not False:
                print(col, 'not in Data')
            continue

        ndata[col] = normalize_column(scalingDict, data, col, mode, verbose)

    return pd.DataFrame(ndata)


def normalize_column(scalingDict, data, col, mode='forward', verbose=False):
    valmin = scalingDict[col]['min']
    valmax = scalingDict[col]['max']

    if mode == 'forward':
        ndata = (data[col] - valmin) / (valmax - valmin)

    elif mode in ['inverse', 'reverse']:
        ndata = ((valmax - valmin) * data[col]) + valmin
    else:
        print('Wrong scaling mode')
        exit()

    return ndata


def scale(scalingDict, data, mode='forward', verbose=False):
    ndata = {}

    for col in scalingDict.keys():
        if col not in data:
            if verbose is not False:
                print(col, 'not in Data')
            continue

        if 'min' in scalingDict[col] and 'max' in scalingDict[col]:
            ndata[col] = normalize_column(scalingDict, data, col, mode)

        elif 'mean' in scalingDict[col] and 'std' in scalingDict[col]:
            ndata[col] = standardize_column(scalingDict, data, col, mode)
        else:
            print('Wrong scaling mode')
            exit()

    return pd.DataFrame(ndata)


def create_subsets(data, rawdata, targets, sortcol, remove_cols=[], eval_percent=0.2, test_percent=0.0, randomseed=False):

    if randomseed:
        random.seed(randomseed)

    # print(data[sortcol])
    eventNameList = list(set(data[sortcol]))
    random.shuffle(eventNameList)

    length = len(eventNameList)
    testlen = int(test_percent * length)
    evallen = int(eval_percent * length)
    testEvts = eventNameList[:testlen]
    evalEvts = eventNameList[testlen:evallen + testlen]
    trainEvts = eventNameList[evallen + testlen:]

    # print(length, (len(testEvts) + len(evalEvts) + len(trainEvts)))
    assert length == (len(testEvts) + len(evalEvts) + len(trainEvts))
    assert not bool(set(testEvts) & set(evalEvts))
    assert not bool(set(trainEvts) & set(evalEvts))
    assert not bool(set(testEvts) & set(trainEvts))

    trainData = data.loc[data[sortcol].isin(trainEvts)]
    testData = data.loc[data[sortcol].isin(testEvts)]
    evalData = data.loc[data[sortcol].isin(evalEvts)]

    xTrain = trainData.drop(columns=targets + [sortcol] + remove_cols)
    yTrain = trainData[targets]
    xTest = testData.drop(columns=targets + [sortcol] + remove_cols)
    yTest = testData[targets]
    xEval = evalData.drop(columns=targets + [sortcol] + remove_cols)
    yEval = evalData[targets]

    xTrain = xTrain.sort_index().reset_index(drop=True)
    yTrain = yTrain.sort_index().reset_index(drop=True)
    xEval = xEval.sort_index().reset_index(drop=True)
    yEval = yEval.sort_index().reset_index(drop=True)
    xTest = xTest.sort_index().reset_index(drop=True)
    yTest = yTest.sort_index().reset_index(drop=True)

    return xTrain, yTrain, xTest, yTest, xEval, yEval


def write_subsets(filecore, xTrain, yTrain, xTest, yTest, xEval, yEval, scalingDict, targets, filetype='csv'):
    
    if filetype.lower() == 'csv':
        xTrain.to_csv('%s_xtrain.csv' % (filecore), index=False)
        yTrain.to_csv('%s_ytrain.csv' % (filecore), index=False)
        xTest.to_csv('%s_xtest.csv' % (filecore), index=False)
        yTest.to_csv('%s_ytest.csv' % (filecore), index=False)
        xEval.to_csv('%s_xeval.csv' % (filecore), index=False)
        yEval.to_csv('%s_yeval.csv' % (filecore), index=False)

    elif filetype.lower() in ['bin', 'binary', 'pickle', 'pkl']:
        xTrain.to_pickle('%s_xtrain.pkl' % (filecore))
        yTrain.to_pickle('%s_ytrain.pkl' % (filecore))
        xTest.to_pickle('%s_xtest.pkl' % (filecore))
        yTest.to_pickle('%s_ytest.pkl' % (filecore))
        xEval.to_pickle('%s_xeval.pkl' % (filecore))
        yEval.to_pickle('%s_yeval.pkl' % (filecore))

    else:
        print('Wrong filetype: %s' % filetype)
        exit()

    inputcols = list(xTrain.columns)
    # try:
    # inputcols.remove('index')
    print(inputcols)

    saveContent = [scalingDict, targets, inputcols]
    pickle.dump(saveContent, open('%s_scalingdict.bin' % (filecore), 'wb'), protocol=-1)

    # print(xTrain)
    # print(xTest)
    # print(xEval)
    # print(yTrain)
    # print(yTest)
    # print(yEval)

    return


def read_subsets(filecore, filetype='csv'):
    print(filecore)

    if filetype.lower() == 'csv':
        xTrain = pd.read_csv('%s_xtrain.csv' % (filecore))
        yTrain = pd.read_csv('%s_ytrain.csv' % (filecore))
        # xTest = pd.read_csv('%s_xtest.csv' % (filecore))
        # if xTest.values.size == 0:
        xTest = []
        # yTest = pd.read_csv('%s_ytest.csv' % (filecore))
        # if yTest.values.size == 0:
        yTest = []
        xEval = pd.read_csv('%s_xeval.csv' % (filecore))
        yEval = pd.read_csv('%s_yeval.csv' % (filecore))
    
    elif filetype.lower() in ['bin', 'binary', 'pickle', 'pkl']:
        xTrain = pd.read_pickle('%s_xtrain.pkl' % (filecore))
        yTrain = pd.read_pickle('%s_ytrain.pkl' % (filecore))
        # xTest = pd.read_pickle('%s_xtest.pkl' % (filecore))
        # if xTest.values.size == 0:
        xTest = []
        # yTest = pd.read_pickle('%s_ytest.pkl' % (filecore))
        # if yTest.values.size == 0:
        yTest = []
        xEval = pd.read_pickle('%s_xeval.pkl' % (filecore))
        yEval = pd.read_pickle('%s_yeval.pkl' % (filecore))

    else:
        print('Wrong filetype: %s' % filetype)
        exit()

    scalingDict, targets, inputcols = pickle.load(open('%s_scalingdict.bin' % (filecore), 'rb'))

    # print(xTrain)
    # print(xTest)
    # print(xEval)

    # print(scalingDict)
    # print(targets)

    return xTrain, yTrain, xTest, yTest, xEval, yEval, scalingDict, targets, inputcols


def read_evaluation_data(filecore):
    print(filecore)

    xEval = pd.read_csv('%s_xeval.csv' % (filecore))
    yEval = pd.read_csv('%s_yeval.csv' % (filecore))
    scalingDict, targets, inputcols = pickle.load(open('%s_scalingdict.bin' % (filecore), 'rb'))

    print(xEval)

    print(scalingDict)
    print(targets)

    return xEval, yEval, scalingDict, targets, inputcols


def convert_categorial_to_numeric(data, cols):
    ncols = []
    for col in cols:
        options = data[col].unique()
        print(options)
        for opt in options:
            data['%s_%s' % (col, opt)] = 0
            data['%s_%s' % (col, opt)][data[col] == opt] = 1
            ncols.append('%s_%s' % (col, opt))

    data = data.drop(columns=cols)
    print(cols)
    print(data[ncols])
    # exit()
    return data


def normalize_elevation(data, cols, colstr='elevation', scalingDict={}, maxelev=9000, minelev=-11000):

    if len(cols) == 0:
        return data, scalingDict

    # ndata = data[cols]
    maxs = data[cols].max(axis=1)
    mins = data[cols].min(axis=1)

    # print(maxs)
    # print(mins)

    for c in cols:
        if ('_max' in c) or ('_min' in c):
            continue

        data[c] = (data[c] - mins) / (maxs - mins)

        data.loc[data[c].isnull(), c] = 0

    data['%s_max' % colstr] = (maxs - minelev) / (maxelev - minelev)
    data['%s_min' % colstr] = (mins - minelev) / (maxelev - minelev)

    if '%s_min' % colstr not in scalingDict:
        scalingDict['%s_min' % colstr] = {}

    if '%s_max' % colstr not in scalingDict:
        scalingDict['%s_max' % colstr] = {}

    scalingDict['%s_min' % colstr]['min'] = minelev
    scalingDict['%s_min' % colstr]['max'] = maxelev
    scalingDict['%s_max' % colstr]['min'] = minelev
    scalingDict['%s_max' % colstr]['max'] = maxelev

    return data, scalingDict


def normalize_frequencies(data, cols, scalingDict={}):

    maxs = max(data[cols].max(axis=1))
    mins = min(data[cols].min(axis=1))

    for c in cols:
        data[c] = (data[c] - mins) / (maxs - mins)
        if c not in scalingDict:
            scalingDict[c] = {}
        scalingDict[c]['min'] = mins
        scalingDict[c]['max'] = maxs

    return data, scalingDict
