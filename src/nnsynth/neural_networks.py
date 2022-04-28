import random
import os
import time
import itertools
import pickle

import numpy as num
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

from gmacc.nnsynth import preprocessing as GMpre

import ewrica.gm.sources as GMs

#####################
### Model generation
#####################


def nn_computation(args, xTrain, yTrain, xTest, yTest, xEval, yEval,
        scalingDict, targets, inputcols):

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    print(scalingDict)
    print(targets)
    print(inputcols)
    print(xTrain.columns)
    print(yTrain.columns)
    print(yEval.columns)

    xLen = len(xTrain)
    batchsizes = []
    for bs in args.batchsizes:
        batchsizes.append(int(xLen * bs))
    
    sortcol = 'evID'

    fac = len(xTrain.columns)
    print('Number of inputs:', fac)

    iters = itertools.product(batchsizes, args.hiddenlayers, args.learningtypes,
            args.activations, args.optimizers)
    print(iters)

    parameters = {
        'device': args.device,
        'dropout': args.dropout,
        'validepochnum': args.validepochnum,
        'maxepochnum': args.maxepochnum,
        'minepochnum': args.minepochnum,
        'outdir': args.outdir,


    }
    for batchsize, hiddenlayer, learning_type, activation, optimizer in iters:

        parameters['activation'] = activation
        parameters['learningtype'] = learning_type
        parameters['optimizer'] = optimizer
        parameters['batchsize'] = int(batchsize)
        # print(options)
        print(batchsize, hiddenlayer, learning_type, activation, optimizer)

        appendix = ''

        if args.targetmode == 'single':
            for ycol in targets:
                if ycol != 'Z_pgd':
                    continue
                print(ycol)

                noutputdir = '%s/%s_%s_%s_%s_%s_%s_%s-%s' % (args.outdir,
                                        parameters['device'], parameters['activation'],
                                        parameters['optimizer'], parameters['batchsize'],
                                        parameters['learningtype'], parameters['dropout'],
                                        hiddenlayer, ycol)
                if not os.path.exists(noutputdir):
                    os.makedirs(noutputdir)
                else:
                    if len(os.listdir(noutputdir)) != 0:
                        continue

                parameters['outdir'] = noutputdir

                model, history = tensorflow_fit(hiddenlayer, xTrain, yTrain[ycol].to_frame(),
                                            parameters, xTest, yTest)

                nn_evaluation(model, history,
                    xTrain, yTrain[ycol].to_frame(), xTest, yTest, xEval, yEval,
                    [ycol], scalingDict, hiddenlayer, parameters, targetwise=True)

                reset_session(model, history)

        elif args.targetmode == 'multi':

            noutputdir = '%s/%s_%s_%s_%s_%s_%s_%s-%s' % (args.outdir,
                                        parameters['device'], parameters['activation'],
                                        parameters['optimizer'], parameters['batchsize'],
                                        parameters['learningtype'], parameters['dropout'],
                                        hiddenlayer, 'multi')
            if not os.path.exists(noutputdir):
                os.makedirs(noutputdir)
            else:
                if len(os.listdir(noutputdir)) != 0:
                    continue

            parameters['outdir'] = noutputdir

            model, history = tensorflow_fit(hiddenlayer, xTrain, yTrain,
                                            parameters, xTest, yTest)

            nn_evaluation(model, history,
                xTrain, yTrain, xTest, yTest, xEval, yEval,
                targets, scalingDict, hiddenlayer, parameters, targetwise=True)

            reset_session(model, history)

        else:
            print('Wrong target mode.')
            exit()


def get_compiled_tensorflow_model(layers, activation='relu', solver='adam',
                                dropout=None,
                                inputsize=False, outputsize=1):

    import tensorflow as tf
    from tensorflow.keras.constraints import max_norm

    tf.keras.backend.clear_session()

    if tf.test.gpu_device_name() == '/device:GPU:0':
        print("\nUsing a GPU\n")
        device = 'gpu'
    else:
        print("\nUsing a CPU\n")
        device = 'cpu'

    modellayers = []

    # from tensorflow.keras.layers.experimental import preprocessing
    # norm = preprocessing.Normalization(input_shape=(inputsize,))
    # modellayers.append(norm)

    for ii, lay in enumerate(layers):
        print(lay)
        if ii == 0 and inputsize:
            layer = tf.keras.layers.Dense(lay, activation=activation,
                    input_shape=(inputsize,),
                    name='hidden_%s' % (ii))

        else:
            layer = tf.keras.layers.Dense(lay, activation=activation,
                    name='hidden_%s' % (ii), kernel_constraint=max_norm(5.0))
        modellayers.append(layer)

        if dropout is not None and dropout != 0.:
            modellayers.append(tf.keras.layers.Dropout(dropout))

    modellayers.append(tf.keras.layers.Dense(outputsize, activation='linear',
                        name='prediction'))

    model = tf.keras.Sequential(modellayers)

    # lr = num.log10(num.log10(model.count_params()))
    # lr = num.log10(num.log10(model.count_params())) / 100.
    # lr = num.log10(model.count_params()) / 500.
    # lr = 0.01
    # lr = max(lr, 0.01)
    # lr = min(lr, 0.5)

    if solver == 'RMS':
        optimizer = tf.keras.optimizers.RMSprop()
    elif solver == 'adam':
        optimizer = tf.keras.optimizers.Adam()
    elif solver == 'sgd':
        optimizer = tf.keras.optimizers.SGD(momentum=0.9)
    else:
        print('Wrong solver/optimizer chosen')
        exit()

    model.compile(loss='mse',
                optimizer=optimizer,
                metrics=['mae', 'mse'])  # 'msle' # 'accuracy'

    print(model.summary())

    return model, device


def tensorflow_fit(layers, xTrain, yTrain, options, xTest=[], yTest=[]):

    model, device = get_compiled_tensorflow_model(layers,
                                options['activation'], options['optimizer'],
                                dropout=options['dropout'],
                                inputsize=len(xTrain.columns),
                                outputsize=len(yTrain.columns))

    if device != options['device'].lower():
        print('\nSet %s, but only %s available.\nChange device mode.' % (options['device'], device))
        exit()

    callbacks = []

    if options['learningtype'] == 'own':
        callbacks.append(combined_callback(cancel_patience=int(options['validepochnum']),
                            decrease_fac=1 / num.sqrt(1.33), decrease_patience=16,
                            increase_fac=num.sqrt(1.33), increase_patience=8,
                            max_lr=0.05, min_lr=0.00001, start_lr=0.0005,
                            min_epochs=options['minepochnum'],
                            outputdir=options['outdir']))

    elif options['learningtype'] == 'default':
        callbacks.append(combined_callback(cancel_patience=int(options['validepochnum']),
                            decrease_fac=1, decrease_patience=num.Inf,
                            increase_fac=1, increase_patience=num.Inf,
                            min_epochs=options['minepochnum'],
                            outputdir=options['outdir']))

    elif options['learningtype'] == 'triangle':
        callbacks.append(triangle_callback(cancel_patience=int(options['validepochnum']),
                            max_lr=0.01, min_lr=0.00001,
                            increase_fac=15, 
                            min_epochs=options['minepochnum'],
                            outputdir=options['outdir']))

    callbacks.append(TimeKeeping(outputdir=options['outdir']))

    verbosity = 4

    if len(xTest) == 0 and len(yTest) == 0:

        history = model.fit(x=xTrain.values, y=yTrain.values,
                shuffle=True,
                verbose=verbosity,
                validation_split=0.2,
                batch_size=options['batchsize'],
                # validation_batch_size=1,
                validation_batch_size=int(xTrain.shape[0]),
                epochs=options['maxepochnum'],
                callbacks=callbacks)

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist)
        print(model.summary())

    else:

        train_dataset = tf.data.Dataset.from_tensor_slices((xTrain.values,
                                                        yTrain.values))
        train_dataset = train_dataset.shuffle(len(xTrain)).batch(options['batchsize'])

        test_dataset = tf.data.Dataset.from_tensor_slices((xTest.values,
                        yTest.values))
        test_dataset = test_dataset.batch(options['batchsize'])

        history = model.fit(train_dataset,
                    shuffle=True,
                    verbose=verbosity,
                    epochs=options['epochs'],
                    callbacks=callbacks,
                    validation_data=test_dataset)

        hist = pd.DataFrame(history.history)
        hist['epoch'] = history.epoch
        print(hist)
        print(model.summary())

        del test_dataset

    model.save('%s/MODEL.h5' % (options['outdir']), include_optimizer=True)
    print('Model saved')
    histdf = pd.DataFrame(history.history)
    histdf.to_csv('%s/history.csv' % (options['outdir']), index=False)
    print('History saved')

    return model, history


def reset_session(model, history):
    ### delete
    import tensorflow as tf
    tf.keras.backend.clear_session()
    del model
    del history

    return


def save_tensorflow(model, outputdir, appendix=''):
    model.save('%s/MODEL_%s.h5' % (outputdir, appendix),
        include_optimizer=True)

    print('Model saved')


#####################
### Model usage
#####################

def check_multi_or_single_nn(path, targets):

    if path.replace('//', '/').rsplit('/')[-2].rsplit('-')[-1] in ['multi', 'multi/']:
        pass
    else:
        targets = [path.replace('//', '/').rsplit('/')[-2].rsplit('-')[-1]]

    return targets


def get_NNcontainer(source, modelfile, suppfile, coords, targetsMain=None):
    print(modelfile)
    model = load_model(modelfile)
    try:
        scalingDict, targets, inputcols = load_scalingdict(suppfile)
    except ValueError:
        scalingDict, targets = load_scalingdict(suppfile)
        inputcols = None

    targets = check_multi_or_single_nn(modelfile, targets)

    lons, lats = coords.T

    data = {}
    for params in scalingDict.keys():
        if params == 'ev_depth':
            data[params] = [source['depth']] * len(coords)
        elif params == 'azistrike':
            data['azimuth'] = source.calc_azimuth(lons=lons, lats=lats)
            data['strike'] = [source.strike] * len(coords)
        elif params == 'rup_azimuth':
            rupazi, centazi = source.calc_rupture_azimuth(lons=lons, lats=lats)
            data['rup_azimuth'] = rupazi
            data['centre_azimuth'] = centazi

        elif params in ['rhypo', 'rrup', 'ry0', 'rx', 'rjb']:
            data[params] = source.calc_distance(lons=lons, lats=lats, distType=params)

        elif params in ['src_duration']:
            data[params] = [source.duration] * len(coords)
        else:
            if params == 'moment':
                continue
            try:
                data[params] = [source[params]] * len(coords)
            except KeyError as e:
                print('Error', e)

    data = pd.DataFrame(data)
    data = GMpre.calc_azistrike(data)
    data = GMpre.convert_distances(data)

    # if 'moment' in scalingDict:
    #     data = GMpre.convert_magnitude_to_moment(data)

    data = GMpre.normalize(scalingDict, data, mode='forward')

    # rearranging the columns, otherwise the NN does not work properly
    # there should be a fix for that to be sure when this is really correct !!!
    # cols = [col for col in scalingDict.keys() if col in data.columns]
    if inputcols is not None:
        cols = inputcols
    else:
        cols = [col for col in scalingDict.keys() if col in data.columns]
    data = data[cols]
    print(data.columns)

    ## Predicting
    pred = model_predict(model, data, int(data.shape[0] / 100))

    preddict = {}
    for nn in range(len(targets)):
        preddict[targets[nn]] = pred[nn]

    preddict = GMpre.normalize(scalingDict, preddict, mode='inverse')
    print(preddict)

    if targetsMain:
        targets = targetsMain
        preddict = preddict[targetsMain]

    preddict['lon'] = lons
    preddict['lat'] = lats

    staDict = {}
    for idx, row in preddict.iterrows():
        ns = 'PR.S%s' % (idx)

        for chagm in targets:
            t = chagm.rsplit('_')

            t = chagm.rsplit('_')
            comp = t[0]

            if t[1] == 'f':
                gm = '%s_%s' % (t[1], t[2])
            else:
                gm = t[1]

            lat = row.lat
            lon = row.lon
            if ns not in staDict:
                STA = GMs.StationGMClass(
                    network=ns.rsplit('.')[0],
                    station=ns.rsplit('.')[1],
                    lat=float(lat),
                    lon=float(lon),
                    components={})
            else:
                STA = staDict['%s.%s' % (ns.rsplit('.')[0], ns.rsplit('.')[1])]

            if comp not in STA.components:
                COMP = GMs.ComponentGMClass(
                        component=comp,
                        gms={})

            GM = GMs.GMClass(name=gm,
                            value=float(row[chagm]))

            COMP.gms[GM.name] = GM

            STA.components[COMP.component] = COMP
            staDict[ns] = STA

    NNCont = GMs.StationContainer(refSource=source, stations=staDict)
    NNCont.validate()

    return NNCont


#####################
### Misc
#####################

def load_model(file):
    model = tf.keras.models.load_model(file)
    return model


def load_scalingdict(file):
    print(file)
    try:
        scalingDict, targets, inputcols = pickle.load(open(file, 'rb'))
        return scalingDict, targets, inputcols

    except ValueError:
        scalingDict, targets = pickle.load(open(file, 'rb'))
        return scalingDict, targets


def model_predict(model, data, batchsize=100):
    output = model.predict(data, batch_size=batchsize).T
    return output


def get_predict_df(model, data, targets, batchsize=100):

    pred = model_predict(model, data, batchsize=batchsize)

    predDict = {}
    for ii in range(len(pred)):
        target = targets[ii]
        predDict[target] = pred[ii]
    predDF = pd.DataFrame(predDict)

    return predDF

#####################
### Evaluation
#####################
def evaluation_synthetic_database(model, xEval, yEval, scalingDict, targets, outdir):

    if outdir.rsplit('-')[-1] in ['multi', 'multi/']:
        pass
    else:
        targets = [outdir.rsplit('-')[-1]]

    predDF = get_predict_df(model, xEval, targets, batchsize=100)

    xEval_norm = GMpre.normalize(scalingDict, xEval, mode='inverse')
    yEval_norm = GMpre.normalize(scalingDict, yEval, mode='inverse')
    predDF_norm = GMpre.normalize(scalingDict, predDF, mode='inverse')

    evaluate_gm_general(predDF_norm, yEval_norm, targets, outdir)
    evaluate_gm_column(['magnitude', 'rhypo'], predDF_norm, xEval_norm, yEval_norm, targets,
            outdir)


def boxplot(diffs, positions, labels, outdir, xlabel='', fileprefix='', predirectory=False, widths=None):

    try:
        nlabels = []
        for ii, key in enumerate(diffs):
            mean = num.mean(diffs[key])
            std = num.std(diffs[key])
            nlab = '%s\n(%0.2f; %0.2f)' % (labels[ii], mean, std)
            nlabels.append(nlab)
        labels = nlabels
    except TypeError as e:
        print(e)
        pass

    fig = plt.figure(figsize=(16, 8))
    plt.boxplot(diffs,
                notch=True,
                # whis=[5, 95], showfliers=False,
                widths=widths,
                positions=positions, labels=labels)
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(-1, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle=':')
    plt.axhline(-0.5, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle='-', alpha=0.25, zorder=-2)
    plt.ylabel('Difference')
    plt.xlabel(xlabel)
    plt.xticks(rotation='60')
    plt.tight_layout()
    plt.savefig('%s/%sboxplot.png' % (outdir, fileprefix))

    if predirectory:
        plt.savefig('%s_%sboxplot.png' % (outdir, fileprefix))
    ax = plt.gca()

    return fig, ax


def violinplot(diffs, positions, labels, outdir, xlabel='', fileprefix='', predirectory=False,
        points=20):

    fig = plt.figure(figsize=(16, 8))
    widths = (num.nanmax(positions) - num.nanmin(positions)) / (len(positions))
    ## widths alternative with num.diff
    parts = plt.violinplot(diffs, positions=positions, points=points, 
        showmedians=False, showextrema=False,
        widths=widths)
    for pc in parts['bodies']:
        pc.set_facecolor('grey')
        pc.set_edgecolor(None)
        pc.set_alpha(1)

    doublestd = num.array([num.percentile(list(d), [2.5, 97.5]) for d in diffs]).T
    std = num.array([num.percentile(list(d), [16, 84]) for d in diffs]).T
    plt.scatter(positions, [num.median(d) for d in diffs], marker='o', color='darkgrey', s=30, zorder=3, label='median')
    plt.vlines(positions, *doublestd, color='darkgrey', linestyle='-', lw=5, label='2*Std')
    plt.vlines(positions, *std, color='black', linestyle='-', lw=5, label='Std')

    plt.grid(True, 'both')
    plt.xticks(positions, labels=labels)
    plt.ylabel('Difference')
    plt.xlabel(xlabel)
    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(-1, color='black', linestyle='--')
    plt.axhline(0.5, color='black', linestyle=':')
    plt.axhline(-0.5, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle='-', alpha=0.25, zorder=-2)
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s/%sviolinplot.png' % (outdir, fileprefix))

    if predirectory:
        plt.savefig('%s_%sboxplot.png' % (outdir, fileprefix))
    ax = plt.gca()

    return fig, ax


def evaluate_gm_general(predDF, yEval, targets, outdir, plotmode='box', violinpoints=20):

    positions = num.arange(0, len(targets))

    diffs = []
    for target in targets:
        obsdat = yEval[target]
        preddat = predDF[target]

        diffs.append((obsdat - preddat).to_numpy())

    if plotmode == 'box':
        boxplot(diffs, positions, targets, outdir=outdir,
                predirectory=True)
    elif plotmode == 'violin':
        violinplot(diffs, positions, targets, outdir=outdir,
                predirectory=True, points=violinpoints)
    else:
        print('Wrong plotmode: %s' % plotmode)

    plt.close('all')


def evaluate_gm_column(columns, predDF, xEval, yEval, targets, outdir, plotmode='box', violinpoints=20):

    for col in columns:
        if col not in xEval:
            print('Column %s not in data.' % col)
            continue

        maxcol = max(xEval[col])
        mincol = min(xEval[col])

        colranges = num.linspace(mincol * 0.99, maxcol * 1.01, 10)
        pltcols = num.round(colranges[:-1] + (colranges[1] - colranges[0]) / 2, 2)
        positions = num.arange(0, len(pltcols))
        
        for target in targets:
            diffs = []
            pltlabels = []
            obsdat = yEval[target]
            preddat = predDF[target]

            for imag in range(len(colranges) - 1):
                cond = ((xEval[col] > colranges[imag]) & (xEval[col] <= colranges[imag + 1]))

                diff = obsdat[cond] - preddat[cond]
                diffs.append(diff)
                pltlabels.append('%0.2f\n(%s)' % ((colranges[imag+1] + colranges[imag]) / 2, len(diff)))

            if plotmode == 'box':
                boxplot(diffs, positions, pltcols, xlabel=col,
                    outdir=outdir, fileprefix='%s_%s_' % (target, col))
            elif plotmode == 'violin':
                violinplot(diffs, positions, pltcols, xlabel=col,
                    outdir=outdir, fileprefix='%s_%s_' % (target, col), 
                    points=violinpoints)
            else:
                print('Wrong plotmode: %s' % plotmode)

            plt.close('all')

    return


def nn_evaluation(model, history,
                xTrain, yTrain, xTest, yTest, xEval, yEval,
                targets, scalingDict,
                hiddenlayer, options, targetwise=True):

    #### Mit scaling dict noch verrechnene, damit man die werte besser interpretieren kann?
    outputdir = options['outdir'].replace('//', '/')
    tmp = outputdir.rsplit('/')[:-1]
    resultdir = os.path.join(*tmp)
    resultfile = '%s/results_NN.csv' % (resultdir)

    resultDict = {
        'layers': str(hiddenlayer).replace(',', ' '),
        'device': options['device'],
        'activation': options['activation'],
        'optimizer': options['optimizer'],
        'batchsize': options['batchsize'],
        'max_epochs': options['maxepochnum'],
        'valid_epochs': options['validepochnum'],
        'dropout': options['dropout'],
        'learning_type': options['learningtype'],
        'exit_epoch': len(history.history['loss']),
        'num_params': model.count_params(),
        'train_time': model.training_time,
        'targets': str(targets).replace(',', ';'),
    }

    evl_batchsize = int(xTrain.shape[0] / 1000.)
    for xdat, ydat, name in zip([xTrain, xTest, xEval],
                                [yTrain, yTest, yEval],
                                ['Train', 'Test', 'Eval']):
        if type(xdat) == list:
            if xdat == []:
                continue
        else:
            if xdat.empty:
                continue

        evaluation = model.evaluate(xdat, ydat, batch_size=evl_batchsize)
        prediction = model_predict(model, xdat, batchsize=evl_batchsize)
        trueval = ydat.values.T
        evalcriteria = list(history.history.keys())[:len(evaluation)]
        print(name, evaluation)

        for ii in range(len(evaluation)):
            resultDict['%s_%s' % (name, evalcriteria[ii])] = '%0.7f' % evaluation[ii]

        if targetwise:
            for ii in range(len(targets)):
                target = targets[ii]

                diff = prediction[ii] - trueval[ii]

                resultDict['%s-%s_rms' % (target, name)] = '%0.7f' % num.sqrt(num.mean(diff**2))
                # resultDict['%s-%s_std' % (target, name)] = num.round(num.std(diff), 5)
                del diff

    print(resultDict)

    if not os.path.exists(resultfile):
        with open(resultfile, "w") as file:
            header = ''
            for ii, key in enumerate(resultDict.keys()):
                if ii >= len(resultDict.keys()) - 1:
                    header += '%s\n' % key
                else:
                    header += '%s,' % key

            file.write(header)

    with open(resultfile, "a") as file:
        line = ''
        for ii, val in enumerate(resultDict.values()):
            if ii >= len(resultDict.values()) - 1:
                line += '%s\n' % val
            else:
                line += '%s,' % val

        file.write(line)

    return


def plot_low2high_ampspectra(model, targets, scalingDict, outdir, xTrain, yTrain, xEval, yEval):
    pred = model_predict(model, xEval)

    # pred_train = model_predict(model, xTrain)

    # try:
    #     lfstr = 'lf_'
    #     lfcols = [c for c in xTrain.columns if lfstr in c and '_m' in c]
    #     # selectColumns += lfcols

    #     ulti_landform = xEval_true[lfcols].idxmax(axis=1)
    #     xEval_true['landform'] = ulti_landform
    #     print(xEval_true['landform'])
    #     # exit()
    # except:
    #     pass

    # print(xEval.columns)
    # print()
    # print(xEval_true.columns)
    # print()
    # print(scalingDict.keys())
    # exit()

    predDict = {}
    # predDict_train = {}
    for ii in range(len(pred)):
        target = targets[ii]
        predDict[target] = pred[ii]
        # predDict_train[target] = pred_train[ii]
    predDF = pd.DataFrame(predDict)
    # predDF_train = pd.DataFrame(predDict_train)

    predDF = GMpre.normalize(scalingDict, predDF, mode='inverse')
    # predDF_train = GMpre.normalize(scalingDict, predDF_train, mode='inverse')

    xEval_true = GMpre.normalize(scalingDict, xEval, mode='inverse')
    yEval_true = GMpre.normalize(scalingDict, yEval, mode='inverse')

    for target in targets:
        if '_res' in target:
            predDF[target] = predDF[target] + xEval_true['%s' % target.replace('_res', '_lowfr')]
            yEval_true[target] = yEval_true[target] + xEval_true['%s' % target.replace('_res', '_lowfr')]

    for cha in ['Z', 'E', 'N']:
        
        # for target in targets:
        #     if cha != 'Z':
        #         continue

        #     obsdat = yEval_true[target]
        #     preddat = predDF[target]
        #     for col in ['magnitude', 'rhypo']:
        #         maxcol = max(xEval[col])
        #         mincol = min(xEval[col])

        #         colranges = num.linspace(mincol, maxcol, 10)
        #         pltcols = num.round(colranges[:-1] + (colranges[1] - colranges[0])/2, 2)
        #         coldiffs = []
        #         pltlabels = []
        #         for imag in range(len(colranges) - 1):
        #             cond = ((xEval[col] > colranges[imag]) & (xEval[col] < colranges[imag + 1]))
        #             coldiff = obsdat[cond] - preddat[cond]
        #             coldiffs.append(coldiff)
        #             pltlabels.append('%0.2f\n(%s)' % ((colranges[imag+1] + colranges[imag]) / 2, len(coldiff)))

        #         boxplot(coldiffs, pltcols, pltlabels, outdir, widths=pltcols[1] - pltcols[0],
        #             xlabel=col, fileprefix='%s_%s_%s' % (col, target, cha), predirectory=False)

        chatargets = [t for t in targets if '%s_' % cha in t]

        diffDF = yEval_true[chatargets] - predDF[chatargets]
        plt.figure(figsize=(12, 6))
        positions = num.arange(0, len(chatargets), 1)
        if cha == 'Z':
            predirectory = True
        else:
            predirectory = False
        boxplot(diffDF, positions, chatargets, outdir, xlabel='',
                fileprefix='%s_' % cha, predirectory=predirectory)

    ## Spectra plots
    inputtargets = [c for c in xEval.columns if '_lowfr' in c]

    random.seed(1)
    idxs = random.choices(range(len(yEval_true[targets[0]])), k=10)
    for nn in idxs:
        plt.figure(nn, figsize=(12, 12))
        for cha in ['Z', 'E', 'N']:
            truexs = []
            trueys = []
            predys = []
            lowys = []
            lowxs = []
            for inptarget in inputtargets:
                if inptarget.rsplit('_')[0] == cha:
                    f = inptarget.rsplit('_')[-2]
                    val = xEval_true[inptarget][nn]
                    lowxs.append(f)
                    lowys.append(val)

            lowxs = num.array(lowxs)
            lowys = num.array(lowys)

            # sortidx = lowfreqs.argsort()

            # lowfreqs = lowfreqs[sortidx]
            # lowvals = lowvals[sortidx]

            for target in targets:
                if target.rsplit('_')[0] == cha:
                    if '_res' in target:
                        t = target.rsplit('_')[-2]
                    else:
                        t = target.rsplit('_')[-1]
                    true = yEval_true[target][nn]
                    valpred = predDF[target][nn]

                    truexs.append(t)
                    trueys.append(true)
                    predys.append(valpred)

            if cha == 'Z':
                ax = plt.subplot(3, 1, 1)
                titlestr = ''
                try:
                    titlestr += 'Mag: %0.2f, Dip: %0.1f, Rake: %0.1f, Depth: %0.1f, \nDist %0.1f' % (
                            xEval_true['magnitude'][nn], xEval_true['dip'][nn],
                            xEval_true['rake'][nn], xEval_true['ev_depth'][nn],
                            10**xEval_true['rhypo'][nn])
                except:
                    pass
                try:
                    titlestr += '\nJapan: %s, Landform: %s' % (xEval_true['japan'][nn], xEval_true['landform'][nn])
                except:
                    pass
                ax.set_title(titlestr)
                ax.xaxis.set_ticklabels([])
            elif cha == 'E':
                ax = plt.subplot(3, 1, 2)
                ax.xaxis.set_ticklabels([])
            elif cha == 'N':
                ax = plt.subplot(3, 1, 3)
                ax.set_xlabel('Frequency [Hz]')
            color = 'red'
            tcolor = 'green'
            ax.scatter(truexs, trueys, color=tcolor, linestyle=':', marker='o', label='True values') # if nn == 0 else None)
            ax.scatter(truexs, predys, color=color, linestyle=':', marker='+', label='Predicted values') # if nn == 0 else None)
            ax.scatter(lowxs, lowys, color='black', linestyle=':', marker='v', label='Lowf values') # if nn == 0 else None)
            ax.set_ylabel('%s Amplitude Difference [log 10]' % (cha))
        ax.legend()
        plt.xticks(rotation='60')
        plt.tight_layout()
        plt.savefig('%s/spectra_%s.png' % (outdir, nn))
        plt.close('all')
    plt.close('all')

    return


#####################
### Callbacks
#####################
class combined_callback(tf.keras.callbacks.Callback):
    """Combined callback for learning rate and loss early stopping
    """

    def __init__(self,
                cancel_patience=0, min_epochs=1,
                decrease_fac=5, decrease_patience=None, max_lr=0.1,
                increase_fac=5, increase_patience=None, min_lr=0.0001,
                start_lr=None, outputdir=''):
        super(combined_callback, self).__init__()

        #  Cancel values
        self.cancel_patience = cancel_patience

        # Values if loss is decreasing
        if decrease_patience is None:
            self.decrease_patience = num.Inf
        else:
            self.decrease_patience = decrease_patience
        self.decrease_patience = decrease_patience
        self.decrease_fac = decrease_fac

        # Values if loss is increasing
        if increase_patience is None:
            self.increase_patience = num.Inf
        else:
            self.increase_patience = increase_patience
        self.increase_fac = increase_fac

        # Min/Max learning rate
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.start_lr = start_lr

        # General values
        self.outputdir = outputdir
        self.min_epochs = min_epochs

    def on_train_begin(self, logs=None):
        self.best_weights = None
        self.best_epoch = 0

        self.ref_cancel_patience = self.cancel_patience

        if self.start_lr is not None:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

        # Epoch where the training is stopped
        self.stopped_epoch = 0

        # The number of epochs it has waited
        self.cancel_wait = 0
        self.decrease_wait = 0
        self.increase_wait = 0

        # Initialize the best as infinity.
        self.best = num.Inf

        # Collect learning rates over training
        self.lrs = []

        # Predefine Loss dict
        self.lossdict = {}

    def on_epoch_end(self, epoch, logs=None):

        # Get current loss
        current = logs.get("val_loss")

        for losskey in logs.keys():
            if losskey not in self.lossdict:
                self.lossdict[losskey] = []
            self.lossdict[losskey].append(logs[losskey])

        # Get the current learning rate from model's optimizer.
        self.lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(self.lr)

        if epoch == 1:
            self.best_weights = self.model.get_weights()

        if num.less(current, self.best):
            self.cancel_wait = 0
            self.decrease_wait = 0
            self.increase_wait += 1

            # Record the best values if current results is better (less).
            self.best = current
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

            if self.increase_wait >= self.increase_patience:
                self.increase_wait = 0

                if not hasattr(self.model.optimizer, "lr"):
                    raise ValueError('Optimizer must have a "lr" attribute.')

                new_lr = self.lr * self.increase_fac
                new_lr = min(new_lr, self.max_lr)
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)
                print('\n### Increasing learning rate from %0.6f to %0.6f' % (self.lr, new_lr))

        else:
            self.cancel_wait += 1
            self.decrease_wait += 1
            self.increase_wait = 0

            if num.isinf(self.decrease_patience) or self.decrease_fac == 1:
                lr_criteria = True
            else:
                if self.lr <= self.min_lr:
                    lr_criteria = True
                else:
                    lr_criteria = False

            if (self.cancel_wait >= self.cancel_patience) and (epoch > self.min_epochs) and lr_criteria:
                # Epoch when training is stopped
                self.stopped_epoch = epoch
                self.model.stop_training = True

                print("\n### Restoring model weights from the end of the best epoch %s." % (self.best_epoch))
                self.model.set_weights(self.best_weights)

            if self.decrease_wait >= self.decrease_patience:
                self.decrease_wait = 0

                ## Resetting model weights to best
                ## Useful or not? Because taking the best can lead to get stuck in the same minimum
                ## But it enhances the search space
                # self.model.set_weights(self.best_weights)
                # print('Updating the weights to former best')

                if not hasattr(self.model.optimizer, "lr"):
                    raise ValueError('Optimizer must have a "lr" attribute.')

                new_lr = self.lr * self.decrease_fac
                new_lr = max(new_lr, self.min_lr)
                if new_lr <= self.min_lr:
                    self.cancel_patience = int(self.ref_cancel_patience / 2.)
                    print('\n### Minimum learning rate of %0.6f reached' % (new_lr))
                    print('\n### Saving current Model')
                    self.model.save('%s/pre_model.h5' % (self.outputdir), include_optimizer=True)
                else:
                    print('\n### Decreasing learning rate from %0.6f to %0.6f' % (self.lr, new_lr))
                tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

        # Plot Learning rate and losses in one figure 
        fig, ax = plt.subplots(figsize=(12, 12))
        for key in self.lossdict.keys():
            if 'loss' in key:
                continue
            ax.semilogy(self.lossdict[key], label=' %s (min=%0.7f)' % (key, min(self.lossdict[key])))

        ax.axvline(x=self.best_epoch, color='red', linestyle='--',
                    label='Best Epoch: %s' % self.best_epoch)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

        ax2 = ax.twinx()

        ax2.plot(self.lrs, 'k:', label='Learning rate')
        ax2.set_ylabel('Learning Rate')
        # ax2.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.savefig('%s/loss-learning_rate-epoch.png' % (self.outputdir))
        plt.close('all')


class triangle_callback(tf.keras.callbacks.Callback):
    """Triangle callback with in- and decreasing learning rate and loss-based early stopping
    """

    def __init__(self,
                cancel_patience=0, min_epochs=1,
                max_lr=0.1, min_lr=0.0001,
                increase_fac=5, 
                start_lr=None, outputdir=''):
        super(triangle_callback, self).__init__()

        #  Cancel values
        self.cancel_patience = cancel_patience

        # Factor for learning rate modification
        self.increase_fac = increase_fac

        # Min/Max learning rate
        self.max_lr = max_lr
        self.min_lr = min_lr
        if start_lr is None:
            self.start_lr = max_lr
        else:
            self.start_lr = start_lr

        # General values
        self.outputdir = outputdir
        self.min_epochs = min_epochs

    def on_train_begin(self, logs=None):
        self.best_weights = None
        self.best_epoch = 0

        self.ref_cancel_patience = self.cancel_patience

        if self.start_lr is not None:
            tf.keras.backend.set_value(self.model.optimizer.lr, self.start_lr)

        # Epoch where the training is stopped
        self.stopped_epoch = 0

        # The number of epochs it has waited
        self.cancel_wait = 0

        # Initialize the best as infinity.
        self.best = num.Inf

        # Collect learning rates over training
        self.lrs = []

        # Predefine Loss dict
        self.lossdict = {}

        # Flag for direction of learning rate evolution
        self.lr_flag = 'decrease'

    def on_epoch_end(self, epoch, logs=None):

        # Get current loss
        current = logs.get("val_loss")

        for losskey in logs.keys():
            if losskey not in self.lossdict:
                self.lossdict[losskey] = []
            self.lossdict[losskey].append(logs[losskey])

        # Get the current learning rate from model's optimizer.
        self.lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        self.lrs.append(self.lr)

        if self.lr_flag == 'increase':
            # new_lr = self.lr * self.increase_fac
            new_lr = self.lr + self.min_lr * self.increase_fac

        elif self.lr_flag == 'decrease':
            # new_lr = self.lr / self.increase_fac
            new_lr = self.lr - self.min_lr * self.increase_fac

        if new_lr >= self.max_lr:
            new_lr = self.max_lr
            self.lr_flag = 'decrease'

        elif new_lr <= self.min_lr:
            new_lr = self.min_lr
            self.lr_flag = 'increase'

        print('\n### %s learning rate from %0.6f to %0.6f' % (self.lr_flag, self.lr, new_lr))
        tf.keras.backend.set_value(self.model.optimizer.lr, new_lr)

        if num.less(current, self.best):
            self.cancel_wait = 0

            # Record the best values if current results is better (less).
            self.best = current
            self.best_weights = self.model.get_weights()
            self.best_epoch = epoch

        else:
            self.cancel_wait += 1

            if (self.cancel_wait >= self.cancel_patience) and (epoch > self.min_epochs):
                # Epoch when training is stopped
                self.stopped_epoch = epoch
                self.model.stop_training = True

                print("\n### Restoring model weights from the end of the best epoch %s." % (self.best_epoch))
                self.model.set_weights(self.best_weights)

    def on_train_end(self, logs=None):
        if self.stopped_epoch > 0:
            print("Epoch %05d: early stopping" % (self.stopped_epoch + 1))

        # Plot Learning rate and losses in one figure 
        fig, ax = plt.subplots(figsize=(12, 12))
        for key in self.lossdict.keys():
            if 'loss' in key:
                continue
            ax.semilogy(self.lossdict[key], label=' %s (min=%0.7f)' % (key, min(self.lossdict[key])))

        ax.axvline(x=self.best_epoch, color='red', linestyle='--',
                    label='Best Epoch: %s' % self.best_epoch)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.legend()

        ax2 = ax.twinx()

        ax2.plot(self.lrs, 'k:', label='Learning rate')
        ax2.set_ylabel('Learning Rate')
        # ax2.legend()

        plt.grid(True)
        plt.tight_layout()
        plt.savefig('%s/loss-learning_rate-epoch.png' % (outputdir))
        plt.close('all')


class TimeKeeping(tf.keras.callbacks.Callback):
    def __init__(self, outputdir):
        # super(TimeKeeping, self).__init__()
        self.times = []
        self.valid_times = []
        # use this value as reference to calculate cummulative time taken
        # self.timebegin = time.process_time()
        self.outputdir = outputdir

    def on_test_begin(self, logs):
        self.valid_timebegin = time.time()

    def on_test_end(self, logs):
        self.valid_times.append(time.time() - self.valid_timebegin)

    def on_epoch_begin(self, epoch, logs=None):
        self.timebegin = time.time()

    def on_epoch_end(self, epoch, logs=None):
        self.times.append(time.time() - self.timebegin)

    def on_train_end(self, logs=None):
        plt.figure(figsize=(12, 12))
        plt.xlabel('Epoch')
        plt.ylabel('time per epoch [s]')
        plt.plot(self.times, label='Epoch-train')
        plt.plot(self.valid_times, label='Epoch-validation')
        print(num.sum(self.times))
        plt.title('Cumulative time: %0.2fs for %s epochs'
                    % (num.sum(self.times) + num.sum(self.valid_times), len(self.times)))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('%s/time_progress.png' % (self.outputdir))
        plt.close('all')

        self.model.training_time = num.sum(self.times) + num.sum(self.valid_times)
