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

import gmacc.gmeval.sources as GMs
import gmacc.gmeval.util as GMu
#####################
### Model generation
#####################


def nn_computation(args, xTrain, yTrain, xTest, yTest, xEval, yEval,
        scalingDict, targets, inputcols, prefix='', gpu_num="1"):

    if args.device == 'cpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num

    # print(scalingDict)
    # print(targets)
    # print(inputcols)
    # print(xTrain.columns)
    # print(yTrain.columns)
    # print(yEval.columns)

    xLen = len(xTrain)
    batchsizes = []
    for bs in args.batchsizes:
        batchsizes.append(int(xLen * bs))
    
    # sortcol = 'evID'

    fac = len(xTrain.columns)
    print('Number of inputs:', fac)

    iters = itertools.product(batchsizes, args.hiddenlayers, args.learningtypes,
            args.activations, args.optimizers, args.losses, args.dropouts)

    parameters = {
        'device': args.device,
        'validepochnum': args.validepochnum,
        'maxepochnum': args.maxepochnum,
        'minepochnum': args.minepochnum,
        'outdir': args.outdir,
    }

    for batchsize, hiddenlayer, learning_type, activation, optimizer, loss, dropout in iters:

        parameters['activation'] = activation
        parameters['learningtype'] = learning_type
        parameters['optimizer'] = optimizer
        parameters['batchsize'] = int(batchsize)
        parameters['loss'] = loss
        parameters['dropout'] = dropout
        # print(options)
        print(batchsize, hiddenlayer, learning_type, activation, optimizer)

        # appendix = ''

        if args.targetmode == 'single':
            for ycol in targets:
                # if ycol != 'Z_pgd':
                #     continue
                print(ycol)

                noutputdir = '%s/%s%s_%s_%s_%s_%s_%s_%s_%s-%s' % (args.outdir, prefix,
                                        parameters['device'], parameters['activation'],
                                        parameters['optimizer'], parameters['batchsize'],
                                        parameters['learningtype'], parameters['loss'],
                                        parameters['dropout'],
                                        hiddenlayer, ycol)

                noutputdir = noutputdir.replace(' ', '')
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
                    [ycol], scalingDict, hiddenlayer, parameters, 
                    targetwise=True, prefix=prefix)

                reset_session(model, history)

        elif args.targetmode == 'multi':

            noutputdir = '%s/%s%s_%s_%s_%s_%s_%s_%s_%s-%s' % (args.outdir, prefix,
                                        parameters['device'], parameters['activation'],
                                        parameters['optimizer'], parameters['batchsize'],
                                        parameters['learningtype'], parameters['loss'],
                                        parameters['dropout'],
                                        hiddenlayer, 'multi')

            noutputdir = noutputdir.replace(' ', '')
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
                targets, scalingDict, hiddenlayer, parameters, 
                targetwise=True, prefix=prefix)

            reset_session(model, history)

        else:
            print('Wrong target mode.')
            exit()


# def CorrelationCoefficient(num_nonwaveform):
#     def loss(y_true, y_pred):
#         '''
#         At the moment the first three values are assumed to be non-waveform
#         '''
#         # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
#         import tensorflow.keras.backend as K
#         import tensorflow.keras.losses as L

#         xs = y_true[:num_nonwaveform]
#         ys = y_pred[:num_nonwaveform]

#         mse = L.MeanSquaredError()
#         rms = mse(xs, ys)

#         x = y_true[num_nonwaveform:]
#         y = y_pred[num_nonwaveform:]
#         mx = K.mean(x)
#         my = K.mean(y)
#         xm, ym = x - mx, y - my
#         r_num = K.sum(tf.multiply(xm, ym))
#         r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
#         r = r_num / r_den

#         r = K.maximum(K.minimum(r, 1.0), -1.0)

#         cc = 1 - K.square(r)

#         # return cc + K.sqrt(rms)
#         return cc + rms

def CorrelationCoefficient1(y_true, y_pred):
    '''
    At the moment the first three values are assumed to be non-waveform
    '''
    # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.losses as L

    num_nonwaveform = 1
    xs = y_true[:num_nonwaveform]
    ys = y_pred[:num_nonwaveform]

    mse = L.MeanSquaredError()
    rms = mse(xs, ys)

    x = y_true[num_nonwaveform:]
    y = y_pred[num_nonwaveform:]
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)

    cc = 1 - K.square(r)

    # return cc + K.sqrt(rms)
    return cc + rms


def CorrelationCoefficient2(y_true, y_pred):
    '''
    At the moment the first three values are assumed to be non-waveform
    '''
    # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.losses as L

    num_nonwaveform = 2
    xs = y_true[:num_nonwaveform]
    ys = y_pred[:num_nonwaveform]

    mse = L.MeanSquaredError()
    rms = mse(xs, ys)

    x = y_true[num_nonwaveform:]
    y = y_pred[num_nonwaveform:]
    mx = K.mean(x)
    my = K.mean(y)
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)

    cc = 1 - K.square(r)

    # return cc + K.sqrt(rms)
    return cc + rms


def CorrelationCoefficient3(y_true, y_pred):
    '''
    At the moment the first three values are assumed to be non-waveform
    '''
    # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.losses as L

    num_nonwaveform = 3
    xs = y_true[:num_nonwaveform]
    ys = y_pred[:num_nonwaveform]

    mse = L.MeanSquaredError()
    rms = mse(xs, ys)

    x = y_true[num_nonwaveform:]
    y = y_pred[num_nonwaveform:]
    mx = K.mean(x)
    my = K.mean(y)
    # mx = x.mean()
    # my = y.mean()
    xm, ym = x - mx, y - my
    r_num = K.sum(tf.multiply(xm, ym))
    r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    r = r_num / r_den

    r = K.maximum(K.minimum(r, 1.0), -1.0)

    cc = 1 - K.square(r)

    # return cc + K.sqrt(rms)
    return cc + rms


def ownCorrelationCoefficient3(y_true, y_pred):
    '''
    An idea would be to use a normalized cross-correlation and
    its absolute maximum value as well as the lag time to those maximum 
    '''
    # https://stackoverflow.com/questions/46619869/how-to-specify-the-correlation-coefficient-as-the-loss-function-in-keras
    import tensorflow.keras.backend as K
    import tensorflow.keras.losses as L
    # from tensorflow_probability.python.internal import dtype_util


    num_nonwaveform = 3
    xs = y_true[:num_nonwaveform]
    ys = y_pred[:num_nonwaveform]

    mse = L.MeanSquaredError()
    rms = mse(xs, ys)

    x = y_true[num_nonwaveform:]
    y = y_pred[num_nonwaveform:]

    # x /= tf.norm(x) 
    # y /= tf.norm(y) 
    # mx = K.mean(x)
    # my = K.mean(y)
    # x /= float(tf.norm(x)) 
    # y /= float(tf.norm(y)) 

    # mx = x.mean()
    # my = y.mean()
    # xm, ym = x - mx, y - my
    # r_num = K.sum(tf.multiply(xm, ym))
    # r_den = K.sqrt(tf.multiply(K.sum(K.square(xm)), K.sum(K.square(ym))))
    # r = r_num / r_den

    # r = K.maximum(K.minimum(r, 1.0), -1.0)

    # cc = 1 - K.square(r)
    # cc2 = (1 - (num.correlate(x, y, mode='full').max())**2)
    # cc3 = 1 - num.corrcoef(x, y)[0, 1]**2
    # cc = r
    # cc2 = num.correlate(x, y, mode='full').max()
    # cc3 = num.corrcoef(x, y)[0, 1]
    # print(cc)
    # print(cc2)
    # print(cc3)
    # print()
    dtype = x.dtype
    x = tf.complex(x, 0.)
    y = tf.complex(y, 0.)
    print(x)
    fft_data = tf.multiply(tf.signal.fft(x), tf.signal.fft(y))
    print(fft_data)
    shifted_product = tf.signal.ifft(fft_data)
    print(shifted_product)

    # Cast back to real-valued if x was real to begin with.
    shifted_product2 = tf.cast(shifted_product, dtype)
    print(shifted_product2)
    row_max = K.max(shifted_product2, axis=1)
    print(row_max)
    cc = 1 - tf.math.reduce_mean(row_max, axis=0)
    # print(cc.numpy)
    # exit()
    # return cc + K.sqrt(rms)
    return cc #+ rms


def get_compiled_tensorflow_model(layers, activation='relu', solver='adam',
                                loss='mse',
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

    nlayers = []
    for lay in layers:
        if type(lay) is str and 'n' in lay:
            nlayers.append(int(lay.replace('n', '')) * inputsize)
        else:
            nlayers.append(lay)

    modellayers = []

    # from tensorflow.keras.layers.experimental import preprocessing
    # norm = preprocessing.Normalization(input_shape=(inputsize,))
    # modellayers.append(norm)

    for ii, lay in enumerate(nlayers):
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

    if loss == 'CorrelationCoefficient3':
        loss = CorrelationCoefficient3
    elif loss == 'CorrelationCoefficient2':
        loss = CorrelationCoefficient2
    elif loss == 'CorrelationCoefficient1':
        loss = CorrelationCoefficient1

    model.compile(loss=loss,
                optimizer=optimizer)  # 'msle' # 'accuracy'

    print(model.summary())

    return model, device


def tensorflow_fit(layers, xTrain, yTrain, options, xTest=[], yTest=[]):

    model, device = get_compiled_tensorflow_model(layers,
                                options['activation'], options['optimizer'],
                                options['loss'],
                                dropout=options['dropout'],
                                inputsize=len(xTrain.columns),
                                outputsize=len(yTrain.columns))

    if device != options['device'].lower():
        print('\nSet %s, but only %s available.\nChange device mode.' % (options['device'], device))
        exit()

    callbacks = []

    if options['learningtype'] == 'own':
        callbacks.append(CombinedCallback(cancel_patience=int(options['validepochnum']),
                            decrease_fac=1 / num.sqrt(1.33), decrease_patience=16,
                            increase_fac=num.sqrt(1.33), increase_patience=8,
                            max_lr=0.05, min_lr=0.00001, start_lr=0.0005,
                            min_epochs=options['minepochnum'],
                            outputdir=options['outdir']))

    elif options['learningtype'] == 'default':
        callbacks.append(CombinedCallback(cancel_patience=int(options['validepochnum']),
                            decrease_fac=1, decrease_patience=num.Inf,
                            increase_fac=1, increase_patience=num.Inf,
                            min_epochs=options['minepochnum'],
                            outputdir=options['outdir']))

    elif options['learningtype'] == 'triangle':
        callbacks.append(TriagnleCallback(cancel_patience=int(options['validepochnum']),
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


def prepare_NN_prediction_data(source, scalingDict, targets, inputcols, coords):
    # print(modelfile)
    # model = load_model(modelfile)

    lons, lats = num.array(coords).T

    r_hypos = source.calc_distance(lons=lons, lats=lats, distType='rhypo')
    azis = source.calc_azimuth(lons=lons, lats=lats)

    if source.form == 'point':
        rrups = None
        rupazis = None
    else:
        rrups = source.calc_distance(lons=lons, lats=lats, distType='rrup')
        rupazis, _ = source.calc_rupture_azimuth(lons=lons, lats=lats)

    data = setup_dataframe(source, scalingDict, inputcols,
                    azis, r_hypos, rrups, rupazis, lenfac=len(coords))

    return data


def get_NNcontainer(source, modelfile, suppfile, coords, targetsMain=None):
    model = load_model(modelfile)
    scalingDict, targets, inputcols = load_supportinfo(suppfile)
    lons, lats = num.array(coords).T

    # dabs_ref_time = time.time()
    data = prepare_NN_prediction_data(source, scalingDict, targets, inputcols, coords)
    # print('Finished data pre preparation: in %s s' % (time.time() - dabs_ref_time))

    ## Predicting
    # dabs_ref_time = time.time()
    preddict = get_predict_df(model, data, targets)
    preddict = GMpre.scale(scalingDict, preddict, mode='inverse')

    preddict['lon'] = lons
    preddict['lat'] = lats
    staDict = {}
    for idx, row in preddict.iterrows():
        net = 'PR'
        sta = 'S%s' % (idx)
        ns = '%s_%s' % (net, sta)

        STA = GMs.StationGMClass(
            network=net,
            station=sta,
            lat=float(row.lat),
            lon=float(row.lon),
            components={})

        for chagm in targets:
            t = chagm.rsplit('_')
            comp = t[0]
            gm = chagm[2:]

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
    # NNCont.validate()
    # print('Finished data post preparation: in %s s' % (time.time() - dabs_ref_time))

    return NNCont


def get_NNdict(source, modelfile, suppfile, coords, targetsMain=None):
    model = load_model(modelfile)
    scalingDict, targets, inputcols = load_supportinfo(suppfile)

    lons, lats = num.array(coords).T

    data = prepare_NN_prediction_data(source, scalingDict, targets, inputcols, coords)

    ## Predicting
    preddf = get_predict_df(model, data, targets)
    preddf = GMpre.scale(scalingDict, preddf, mode='inverse')

    preddict = {}
    for chagm in preddf.columns:
        t = chagm.rsplit('_')
        comp = t[0]
        gm = chagm[2:]

        if gm not in preddict:
            preddict[gm] = {}

        if comp not in preddict[gm]:
            preddict[gm][comp] = {'vals': [], 'lons': [], 'lats': []}

        for rr, row in preddf.iterrows():
            val = row[chagm]
            preddict[gm][comp]['vals'].append(val)
            preddict[gm][comp]['lons'].append(lons[rr])
            preddict[gm][comp]['lats'].append(lats[rr])

    return preddict


###
# Predicting based on source
###

def setup_dataframe(src, scaling_dict, inputcols,
                    azis, r_hypos, rrups, rupazis, lenfac):
    data = {}
    for params in scaling_dict.keys():
        if params == 'azistrike':
            data['azimuth'] = azis
            data['strike'] = [float(src.strike)] * lenfac

        if params == 'rhypo':
            data[params] = r_hypos

        if params == 'src_duration':
            data[params] = [float(src.duration)] * lenfac
            # except AttributeError:
            #     print(
            #         'Source STF has no duration (Needs to be added).')
            #     # dur = GMu.calc_rupture_duration(source=src, mode='uncertain')
            #     dur = GMu.calc_rise_time(source=src)
            #     print('Calculated STF duration of {} s'.format(dur))
            #     data[params] = [float(dur)] * lenfac

        if params == 'ev_depth':
            data[params] = [float(src.depth)] * lenfac

        if params == 'dip':
            data[params] = [float(src.dip)] * lenfac

        if params == 'rake':
            data[params] = [float(src.rake)] * lenfac

        if params == 'magnitude':
            try:
                data[params] = [float(src.magnitude)] * lenfac
            except AttributeError:
                data[params] = [float(src.moment_magnitude)] * lenfac

        if params == 'length':
            try:
                data['length'] = [float(src.length)] * lenfac
            except AttributeError:
                raise AttributeError('Source has no length information')

        if params == 'width':
            try:
                data['width'] = [float(src.width)] * lenfac
            except AttributeError:
                raise AttributeError('Source has no width information')

        if params == 'nucleation_x':
            try:
                data['nucleation_x'] = [float(src.nucleation_x)] * lenfac
                data['nucleation_y'] = [float(src.nucleation_y)] * lenfac
            except AttributeError:
                raise AttributeError('Source has no nucleation information')

        if params == 'rrup':
            data['rrup'] = rrups

        if params == 'rup_azimuth' or params == 'rup_azistrike':
            data['rup_azimuth'] = rupazis

    data = pd.DataFrame(data)
    # data = GMpre.calc_azistrike(data)
    data = GMpre.calc_azistrike(data, strikecol='strike',
        azimuthcol='azimuth', azistrikecol='azistrike', delete=False)
    dropcols = ['azimuth', 'strike']

    if 'rup_azimuth' in data:
        data = GMpre.calc_azistrike(data, strikecol='strike',
            azimuthcol='rup_azimuth', azistrikecol='rup_azistrike', delete=False)
        dropcols.append('rup_azimuth')
    data = data.drop(columns=dropcols)
    data = GMpre.convert_distances(data)

    data = GMpre.normalize(scaling_dict, data, mode='forward')

    if inputcols is not None:
        cols = inputcols
    else:
        cols = [col for col in scaling_dict.keys() if col in data.columns]
    data = data[cols]

    return data


def dataframe_to_stadict(preddf, multicoords):

    numstas = len(multicoords[0])
    numevs = int(len(preddf) / numstas)
    
    targets = preddf.columns
    chas = [t.rsplit('_')[0] for t in targets]
    chas = list(set(chas))

    gms = [t.rsplit('_')[1] for t in targets]
    gms = list(set(gms))

    staDicts = []
    for ee in range(numevs):
        coords = multicoords[ee]
        lons, lats = coords.T
        staDict = {}
        for ss in range(numstas):
            net = 'NN'
            sta = ss

            ns = '%s.%s' % (net, sta)

            for cha in chas:
                # cha = 'Z'  # needs to be adapted

                COMP = GMs.ComponentGMClass(
                    component=cha, gms={})

                for gm in gms:
                    GM = GMs.GMClass(
                        name=gm,
                        value=float(preddf.iloc[int(ee * numstas + ss)]),
                        unit='UKN')
                COMP.gms[GM.name] = GM

                if ns not in staDict:
                    STA = GMs.StationGMClass(
                        network=net,
                        station=sta,
                        lat=float(lats[ss]),
                        lon=float(lons[ss]),
                        components={})

                    STA.components[COMP.component] = COMP
                    staDict['%s.%s' % (STA.network, STA.station)] = STA

                else:
                    staDict[ns].components[COMP.component] = COMP
        staDicts.append(staDict)

    return staDicts


def get_NNcont_prediction_together(srcs, modelfile, suppfile, multicoords):
    model = load_model(modelfile)
    
    scalingdict, targets, inputcols = load_supportinfo(suppfile)
    datas = []

    # dabs_ref_time = time.time()
    for coords, src in zip(multicoords, srcs):
        data = prepare_NN_prediction_data(src, scalingdict, targets, inputcols, coords)
        datas.append(data)

    alldata = pd.concat(datas, ignore_index=True)
    # print('Finished data pre preparation: in %s s' % (time.time() - dabs_ref_time))

    # dabs_ref_time = time.time()
    preddf = get_predict_df(model, alldata, targets, batchsize=10000)
    preddf = GMpre.scale(scalingdict, preddf, mode='inverse')

    stadicts = dataframe_to_stadict(preddf, multicoords)

    conts = []
    for ss in range(len(srcs)):
        conts.append(GMs.StationContainer(refSource=srcs[ss], stations=stadicts[ss]))
    # print('Finished data post preparation: in %s s' % (time.time() - dabs_ref_time))
    return conts


def get_NN_prediction_prob_map(srcs, modelfile, suppfile, multicoords):

    print(modelfile)
    model = load_model(modelfile)
    
    scaling_dict, targets, inputcols = load_supportinfo(suppfile)

    datas = []
    # alldata = pd.DataFrame()

    dabs_ref_time = time.time()
    for coords, src in zip(multicoords, srcs):
        # src.depth = src.depth * 1000.
        lons, lats = num.array(coords).T
        r_hypos = GMs.get_distances(lons, lats, src, distType='hypo')
        azis = GMs.get_azimuths(lons, lats, src, aziType='hypo')

        # print(src)

        if src.form == 'point':
            rrups = None
            rupazis = None

        elif src.form == 'rectangular':

            rrups = GMs.get_distances(lons, lats, src, distType='rrup')
            rupazis = GMs.get_azimuths(lons, lats, src, aziType='rup')
        else:
            print('Wrong form: %s, not implemented yet.' % src.form)
            exit()

        lenfac = len(lons)
        # src.depth = src.depth / 1000.
        data = setup_dataframe(src, scaling_dict, inputcols,
            azis, r_hypos, rrups, rupazis, lenfac)

        # alldata = pd.concat([alldata, data], ignore_index=True)
        datas.append(data)
        # alldata.append(data)
    alldata = pd.concat(datas, ignore_index=True)
    print(alldata)
    print('Finished data pre preparation: in %s s' % (time.time() - dabs_ref_time))

    preddf = get_predict_df(model, alldata, targets, batchsize=10000)
    preddf = GMpre.scale(scaling_dict, preddf, mode='inverse')

    return preddf


#####################
### Misc
#####################

def load_model(file):
    return tf.keras.models.load_model(file,
        custom_objects={'CorrelationCoefficient1': CorrelationCoefficient1,
                        'CorrelationCoefficient2': CorrelationCoefficient2,
                        'CorrelationCoefficient3': CorrelationCoefficient3})


# def load_scalingdict(file):
#     print(file)
#     try:
#         scalingDict, targets, inputcols = pickle.load(open(file, 'rb'))
#         return scalingDict, targets, inputcols

#     except ValueError:
#         scalingDict, targets = pickle.load(open(file, 'rb'))
#         return scalingDict, targets

def load_supportinfo(file):
    ## produces output: scalingDict, targets, inputcols
    return pickle.load(open(file, 'rb'))


def model_predict(model, data, batchsize=10000):
    return model.predict(data, batch_size=batchsize).T


def get_predict_df(model, data, targets, batchsize=10000):

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

    xEval_norm = GMpre.scale(scalingDict, xEval, mode='inverse')
    yEval_norm = GMpre.scale(scalingDict, yEval, mode='inverse')
    predDF_norm = GMpre.scale(scalingDict, predDF, mode='inverse')

    evaluate_gm_general(predDF_norm, yEval_norm, targets, outdir)
    evaluate_gm_column(['magnitude', 'rhypo'], predDF_norm, xEval_norm, yEval_norm, targets,
            outdir)


def boxplot(diffs, positions, labels, outdir, xlabel='', ylabel='Difference', fileprefix='', predirectory=False, widths=None):

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
                # whis=[5, 95],
                showfliers=False,
                widths=widths,
                positions=positions, labels=labels)

    plt.axhline(1, color='black', linestyle='--')
    plt.axhline(-1, color='black', linestyle='--')
    plt.axhline(0.3, color='black', linestyle='-.')
    plt.axhline(-0.3, color='black', linestyle='-.')
    plt.axhline(0.5, color='black', linestyle=':')
    plt.axhline(-0.5, color='black', linestyle=':')
    plt.axhline(0, color='black', linestyle='-', alpha=0.25, zorder=-2)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.xticks(rotation='60')
    plt.tight_layout()
    plt.savefig('%s/%sboxplot.png' % (outdir, fileprefix))

    if predirectory:
        plt.savefig('%s_%sboxplot.png' % (outdir, fileprefix))
    ax = plt.gca()

    return fig, ax


def violinplot(diffs, positions, labels, outdir, xlabel='', ylabel='Difference', fileprefix='', predirectory=False,
        points=20, ymin=-2, ymax=2, axhline=1, figsize=(8, 6), grid=False):

    fig = plt.figure(figsize=figsize)
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
    plt.vlines(positions, *std, color='black', linestyle='-', lw=5, label='Std', zorder=2)
    plt.vlines(positions, *doublestd, color='darkgrey', linestyle='-', lw=5, label='2*Std', zorder=1)

    if grid is True:
        plt.grid(True, 'both')
    else:
        plt.xticks(positions, labels=labels)
    plt.ylabel(ylabel, fontsize=17)
    plt.xlabel(xlabel, fontsize=17)
    if axhline:
        plt.axhline(1 * axhline, color='black', linestyle='--')
        plt.axhline(-1 * axhline, color='black', linestyle='--')
        plt.axhline(0.3 * axhline, color='black', linestyle='-.')
        plt.axhline(-0.3 * axhline, color='black', linestyle='-.')
        plt.axhline(0.5 * axhline, color='black', linestyle=':')
        plt.axhline(-0.5 * axhline, color='black', linestyle=':')
        plt.axhline(0 * axhline, color='black', linestyle='-', alpha=0.25, zorder=-2)
    else:
        plt.grid('both')
    plt.ylim((ymin, ymax))
    plt.xticks(rotation='60')
    plt.legend()
    plt.tight_layout()
    plt.savefig('%s/%sviolinplot.png' % (outdir, fileprefix))

    if predirectory:
        plt.savefig('%s_%sviolinplot.png' % (outdir, fileprefix))
    ax = plt.gca()

    return fig, ax


def evaluate_gm_general(predDF, yEval, targets, outdir, predirectory=True, fileprefix='', plotmode='box', violinpoints=20):

    positions = num.arange(0, len(targets))

    diffs = []
    for target in targets:
        obsdat = yEval[target]
        preddat = predDF[target]

        diffs.append((obsdat - preddat).to_numpy())

    if plotmode == 'box':
        boxplot(diffs, positions, targets, outdir=outdir, fileprefix=fileprefix,
                predirectory=predirectory)
    elif plotmode == 'violin':
        violinplot(diffs, positions, targets, outdir=outdir, fileprefix=fileprefix,
                predirectory=predirectory, points=violinpoints)
    else:
        print('Wrong plotmode: %s' % plotmode)

    plt.close('all')


def evaluate_gm_column(columns, predDF, xEval, yEval, targets, outdir, plotmode='box', violinpoints=20, figsize=(8, 6)):

    for col in columns:
        if col not in xEval:
            print('Column %s not in data.' % col)
            continue

        maxcol = max(xEval[col])
        mincol = min(xEval[col])

        colranges = num.linspace(mincol * 0.99, maxcol * 1.01, 11)
        pltcols = num.round(colranges[:-1] + (colranges[1] - colranges[0]) / 2, 2)
        # positions = num.arange(0, len(pltcols))
        positions = pltcols
        
        for target in targets:
            diffs = []
            pltlabels = []
            obsdat = yEval[target]
            preddat = predDF[target]

            for imag in range(len(colranges) - 1):
                cond = ((xEval[col] > colranges[imag]) & (xEval[col] <= colranges[imag + 1]))

                diff = obsdat[cond] - preddat[cond]
                if len(diff) == 0:
                    diff = [0.]
                diffs.append(diff)
                pltlabels.append('%0.2f\n(%s)' % ((colranges[imag + 1] + colranges[imag]) / 2, len(diff)))

            if plotmode == 'box':
                boxplot(diffs, positions, pltcols, xlabel=col.capitalize(),
                    outdir=outdir, fileprefix='%s_%s_' % (target, col))
            elif plotmode == 'violin':
                violinplot(diffs, positions, pltcols, xlabel=col.capitalize(),
                    outdir=outdir, fileprefix='%s_%s_' % (target, col), 
                    ymin=-0.2, ymax=0.2, axhline=0, grid=True,
                    points=violinpoints, figsize=figsize, ylabel='Difference [log10]')
            else:
                print('Wrong plotmode: %s' % plotmode)

            plt.close('all')

    return


def nn_evaluation(model, history,
                xTrain, yTrain, xTest, yTest, xEval, yEval,
                targets, scalingDict,
                hiddenlayer, options, targetwise=True, prefix=''):

    #### Mit scaling dict noch verrechnene, damit man die werte besser interpretieren kann?
    outputdir = options['outdir'].replace('//', '/')
    tmp = outputdir.rsplit('/')[-1]
    resultdir = outputdir.replace(tmp, '')
    resultfile = os.path.join(resultdir, 'results_NN.csv')

    resultDict = {
        'prefix': prefix,
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

    # evl_batchsize = int(xTrain.shape[0] / 100.)
    evl_batchsize = 10
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
        print(evaluation)
        prediction = model_predict(model, xdat, batchsize=evl_batchsize)
        trueval = ydat.values.T
        if type(evaluation) is float:
            evaluation = [evaluation]
    
        evalcriteria = list(history.history.keys())[:len(evaluation)]
        # print(name, evaluation)
        print(evalcriteria)

        for ii in range(len(evaluation)):
            resultDict['%s_%s' % (name, evalcriteria[ii])] = '%0.7f' % evaluation[ii]

        if targetwise:
            for ii in range(len(targets)):
                target = targets[ii]

                diff = prediction[ii] - trueval[ii]

                resultDict['%s-%s_rms' % (target, name)] = '%0.7f' % num.sqrt(num.mean(diff**2))
                # resultDict['%s-%s_std' % (target, name)] = num.round(num.std(diff), 5)
                del diff

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

    predDF = GMpre.scale(scalingDict, predDF, mode='inverse')
    # predDF_train = GMpre.scale(scalingDict, predDF_train, mode='inverse')

    xEval_true = GMpre.scale(scalingDict, xEval, mode='inverse')
    yEval_true = GMpre.scale(scalingDict, yEval, mode='inverse')

    for target in targets:
        if '_res' in target:
            try:
                predDF[target] = predDF[target] + xEval_true['%s' % target.replace('_res', '_lowfr')]
                yEval_true[target] = yEval_true[target] + xEval_true['%s' % target.replace('_res', '_lowfr')]
            except KeyError as e:
                # print('Error:', e)
                cha = target.rsplit('_')[0]
                sfreq = max([float(t.rsplit('_')[-2]) for t in xEval.columns if ('_lowfr' in t) and (float(t.rsplit('_')[-2]) < 0.5)])
                scol = '%s_f_%s_lowfr' % (cha, sfreq)
                # try:
                predDF[target] = predDF[target] + xEval_true[scol]
                yEval_true[target] = yEval_true[target] + xEval_true[scol]
                # except:
                #     predDF[target] = num.zeros(len(predDF[target]))
                #     yEval_true[target] = num.zeros(len(yEval_true[target]))

    evaluate_gm_general(predDF, yEval_true, targets, outdir, plotmode='violin', violinpoints=200, predirectory=False)
    evaluate_gm_general(predDF, yEval_true, targets, outdir, plotmode='box', predirectory=False)

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
        evaluate_gm_general(predDF[chatargets], yEval_true[chatargets], chatargets, outdir, fileprefix='%s_' % cha, plotmode='violin', violinpoints=200, predirectory=False)
        evaluate_gm_general(predDF[chatargets], yEval_true[chatargets], chatargets, outdir, fileprefix='%s_' % cha, plotmode='box', predirectory=False)

    ## Spectra plots
    inputtargets = [c for c in xEval.columns if '_lowfr' in c]

    random.seed(1)
    idxs = random.choices(range(len(yEval_true[targets[0]])), k=10)
    for nn in idxs:
        plt.figure(nn, figsize=(8, 8))
        for cha in ['Z', 'E', 'N']:
            truexs = []
            trueys = []
            predys = []
            # lowys = []
            # lowxs = []
            # for inptarget in inputtargets:
            #     if inptarget.rsplit('_')[0] == cha:
            #         f = inptarget.rsplit('_')[-2]
            #         val = xEval_true[inptarget][nn]
            #         lowxs.append(f)
            #         lowys.append(val)

            # lowxs = num.array(lowxs)
            # lowys = num.array(lowys)

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
            # ax.scatter(lowxs, lowys, color='black', linestyle=':', marker='v', label='Lowf values') # if nn == 0 else None)
            ax.set_ylabel('%s Amplitude\nDifference [log 10]' % (cha))
        ax.legend()
        plt.xticks(rotation='60')
        plt.tight_layout()
        plt.savefig('%s/spectra_%s.png' % (outdir, nn))
        plt.close('all')
    plt.close('all')

    return


def modify_col_ablation(mode, data, col):
    if mode == 'random':
        data[col] = num.random.uniform(0, 1, len(data[col]))
    elif mode == 'shuffle':
        data[col] = data[col].sample(frac=1).values
    else:
        print('Wrong mode')
        exit()
    return data


def feature_importance_one_rng(model, xEval, yEval, twoparams=False, 
        losses=['loss', 'mae', 'mse'],
        mode='random'):
    '''
    modes:
    - random: uniform random between  0 and 1
    - shuffle: shuffles the values of one column
    '''
    import pandas as pd

    df = pd.DataFrame(columns=losses)
    df.loc['None_replaced'] = model.evaluate(xEval, yEval)

    nxEval = xEval.copy(deep=True)
    for col in xEval.columns:
        nxEval = modify_col_ablation(mode, nxEval, col)

    df.loc['All_replaced'] = model.evaluate(nxEval, yEval)

    for col in xEval.columns:
        nxEval = xEval.copy(deep=True)
        nxEval = modify_col_ablation(mode, nxEval, col)

        df.loc['     %s' % col] = model.evaluate(nxEval, yEval)

    if twoparams:
        blacklist = []
        for col1 in xEval.columns:
            nxEval = xEval.copy(deep=True)
            nxEval = modify_col_ablation(mode, nxEval, col1)

            for col2 in xEval.columns:

                if col1 == col2:
                    continue

                if col2 in blacklist:
                    continue

                nxEval = modify_col_ablation(mode, nxEval, col2)

                df.loc['%s & %s' % (col1, col2)] = model.evaluate(nxEval, yEval)

            blacklist.append(col1)

    print('Named features were replaced by %s:' % mode)
    print(df.sort_values(by='loss'))


def feature_importance_one_true(model, xEval, yEval, twoparams=False, 
        losses=['loss', 'mae', 'mse'],
        mode='random'):
    '''
    modes:
    - random: uniform random between  0 and 1
    - shuffle: shuffles the values of one column
    '''
    import pandas as pd

    df = pd.DataFrame(columns=losses)
    df.loc['None_replaced'] = model.evaluate(xEval, yEval)

    nxEval = xEval.copy(deep=True)
    for col in xEval.columns:
        nxEval = modify_col_ablation(mode, nxEval, col)

    df.loc['All_replaced'] = model.evaluate(nxEval, yEval)

    for truecol in xEval.columns:
        nxEval = xEval.copy(deep=True)
        for col in xEval.columns:
            if col == truecol:
                continue
            nxEval = modify_col_ablation(mode, nxEval, col)

        df.loc['     %s' % truecol] = model.evaluate(nxEval, yEval)

    if twoparams:
        blacklist = []
        for truecol1 in xEval.columns:
            nxEval = xEval.copy(deep=True)

            for truecol2 in xEval.columns:

                if truecol1 == truecol2:
                    continue

                if truecol2 in blacklist:
                    continue

                for col in xEval.columns:
                    if col == truecol1 or col == truecol2:
                        continue

                    nxEval = modify_col_ablation(mode, nxEval, col)

                df.loc['%s & %s' % (truecol1, truecol2)] = model.evaluate(nxEval, yEval)

            blacklist.append(truecol1)

    print('All features except the named ones were replaced by %s:' % mode)
    print(df.sort_values(by='loss'))


#####################
### Callbacks
#####################
class CombinedCallback(tf.keras.callbacks.Callback):
    """Combined callback for learning rate and loss early stopping
    """

    def __init__(self,
                cancel_patience=0, min_epochs=1,
                decrease_fac=5, decrease_patience=None, max_lr=0.1,
                increase_fac=5, increase_patience=None, min_lr=0.0001,
                start_lr=None, outputdir=''):
        super(CombinedCallback, self).__init__()

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
            # if 'loss' in key:
            #     continue
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


class TriagnleCallback(tf.keras.callbacks.Callback):
    """Triangle callback with in- and decreasing learning rate and loss-based early stopping
    """

    def __init__(self,
                cancel_patience=0, min_epochs=1,
                max_lr=0.1, min_lr=0.0001,
                increase_fac=5, 
                start_lr=None, outputdir=''):
        super(TriagnleCallback, self).__init__()

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
        plt.savefig('%s/loss-learning_rate-epoch.png' % (self.outputdir))
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
