import os

import numpy as num
import matplotlib.pyplot as plt

from gmacc.gmeval import util as GMu


def nn_waveform_plots_none(predDF, yEval, outputdir):

    state = num.random.RandomState(seed=1)
    idx = num.arange(predDF.shape[0])
    state.shuffle(idx)

    for mm in idx[:10]:
        plt.figure(figsize=(16, 9))
        xvals = range(len(predDF.iloc[mm]))

        predval = predDF.iloc[mm] 
        trueval = yEval.T[mm]

        predval -= num.mean(predval)
        predval /= num.max(predval)

        trueval -= num.mean(trueval)
        trueval /= num.max(trueval)

        plt.plot(xvals, predval, label='NN Predicted')
        plt.plot(xvals, trueval, label='GF Ouput')
        # plt.ylim(-1, 1)
        plt.legend(loc='best')

        titlestr = str(mm)
        plt.title(titlestr)
        plt.tight_layout()
        plt.savefig(outputdir + '/wv_%s.png' % mm)

    plt.close('all')


def nn_waveform_plots_time(predDF, yEval, outputdir):

    cntstart = 0
    if 'maxval' in yEval:
        cntstart += 1

    if 'maxval-sign' in yEval:
        cntstart += 1

    elif 'maxval-sign-pos' in yEval:
        cntstart += 2

    for mm, row in predDF.iterrows():

        pred = predDF.iloc[mm]
        # pred /= max(pred)
        true = yEval.iloc[mm]

        # plt.figure(figsize=(8, 5))
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12,10))
        xvals = range(len(pred) - cntstart)
        # ax1.axhline(1, color='black', alpha=0.3)
        # ax1.axhline(-1, color='black', alpha=0.3)
        ax1.plot(xvals, pred[cntstart:], label='NN Predicted')
        ax1.plot(xvals, true[cntstart:], label='GF Ouput')
        # ax1.set_ylim(-1, 1)
        ax1.legend(loc='best')

        titlestr = str(mm)
        if 'maxval' in yEval:
            titlestr += '\nPGD (pred, true):\nAmp: %0.3f %0.3f' % (pred[0], true[0])
        if 'maxval-sign' in yEval:
            titlestr += '\nSign: %0.3f %0.3f' % (pred[1], true[1])
        elif 'maxval-sign-pos' in yEval:
            titlestr += '\nSign pos: %0.3f %0.3f' % (pred[1], true[1])
            titlestr += '\nSign neg: %0.3f %0.3f' % (pred[2], true[2])

        x = pred[cntstart:]
        y = true[cntstart:]

        titlestr += '\nRMS: %.3f' % GMu.get_rms(x, y)
        titlestr += '\nCC: %.3f' % GMu.get_cc(x, y)
        titlestr += '\nWS: %.3f' % GMu.get_wasserstein_dist(x, y)
        ax1.set_title(titlestr)

        ### Results plot
        trMaxpred = (10**pred[0])
        trMaxtrue = (10**true[0])
        if 'maxval-sign-pos' in yEval:
            if pred[1] > pred[2]:
                trMaxpred *= 1
            else:
                trMaxpred *= -1

            if true[1] > true[2]:
                trMaxtrue *= 1
            else:
                trMaxtrue *= -1

        ax2.plot(xvals, trMaxpred * pred[cntstart:], label='NN Predicted')
        ax2.plot(xvals, trMaxtrue * true[cntstart:], label='GF Ouput')
        ax2.set_title('Final result')
        plt.tight_layout()
        plt.savefig(outputdir + '/wv_%s.png' % mm)

    plt.close('all')


def nn_waveform_plots_freq(predDF, yEval, outputdir):

    cntstart = 0
    if 'maxval' in yEval:
        cntstart += 1

    if 'maxval-sign' in yEval:
        cntstart += 1

    elif 'maxval-sign-pos' in yEval:
        cntstart += 2

    realscolumns = [x for x in predDF.columns if 'reals' in x]
    imagscolumns = [x for x in predDF.columns if 'imags' in x]
    print(realscolumns)
    print(imagscolumns)

    for mm, row in predDF.iterrows():

        pred = predDF.iloc[mm]
        true = yEval.iloc[mm]

        trMaxpred = (10**pred[0])
        trMaxtrue = (10**true[0])

        if pred[1] > pred[2]:
            trMaxpred *= 1
        else:
            trMaxpred *= -1

        if true[1] > true[2]:
            trMaxtrue *= 1
        else:
            trMaxtrue *= -1

        reals = num.array(pred[realscolumns])
        imags = num.array(pred[imagscolumns])
        reals *= trMaxpred
        imags *= trMaxpred
        predft = reals + 1j * imags
        predvals = num.fft.irfft(predft)

        reals = num.array(true[realscolumns])
        imags = num.array(true[imagscolumns])
        reals *= trMaxtrue
        imags *= trMaxtrue
        evalft = reals + 1j * imags
        evalvals = num.fft.irfft(evalft)

        plt.figure(figsize=(8, 6))
        ax = plt.subplot(2, 1, 1)
        ax.plot(pred[cntstart:], label='NN Predicted')
        ax.plot(true[cntstart:], label='GF Ouput')
        ax.vlines(len(realscolumns),
            ymin=min(min(pred[cntstart:]), min(true)),
            ymax=max(max(pred[cntstart:]), max(true)),
            zorder=-1, linestyle=':', color='gray')
        ax.set_ylim((-1.25, 1.25))
        ax.legend(loc='best')

        titlestr = str(mm)
        if 'maxval' in yEval:
            titlestr += '\nPGD (pred, true):\nAmp: %0.3f %0.3f' % (trMaxpred, trMaxtrue)
        if 'maxval-sign' in yEval:
            titlestr += '\nSign: %0.3f %0.3f' % (pred[1], true[1])
        elif 'maxval-sign-pos' in yEval:
            titlestr += '\nSign pos: %0.3f %0.3f' % (pred[1], true[1])
            titlestr += '\nSign neg: %0.3f %0.3f' % (pred[2], true[2])

        titlestr += '\nRMS: %.3f' % (num.sqrt(num.mean((pred[cntstart:] - true[cntstart:])**2)))
        titlestr += '\nCC: %.3f' % (num.corrcoef(pred[cntstart:], true[cntstart:])[0, 1])

        ax.set_title(titlestr)

        ## timeplot
        ax = plt.subplot(2, 1, 2)
        titlestr = '\nRMS: %.3f' % (num.sqrt(num.mean((predvals - evalvals)**2)))
        titlestr += '\nCC: %.3f' % (num.corrcoef(predvals, evalvals)[0, 1])
        ax.set_title(titlestr)

        ax.plot(predvals, label='NN Predicted')
        ax.plot(evalvals, label='GF Ouput')
        # plt.ylim(-1, 1)
        ax.legend(loc='best')

        plt.tight_layout()
        plt.savefig(outputdir + '/wv_%s.png' % mm)

    plt.close('all')


def evaluate_waveform_general(predDF, yEval, outputdir, predirectory=True, plot=True):

    cntstart = 0
    if 'maxval' in yEval:
        cntstart += 1

    if 'maxval-sign' in yEval:
        cntstart += 1

    elif 'maxval-sign-pos' in yEval:
        cntstart += 2

    maxrmss = num.abs(predDF.iloc[:, 0].values - yEval.iloc[:, 0].values)
    signrms1s = num.abs(predDF.iloc[:, 1].values - yEval.iloc[:, 1].values)
    signrms2s = num.abs(predDF.iloc[:, 2].values - yEval.iloc[:, 2].values)
    
    x = predDF.iloc[:, cntstart:]
    y = yEval.iloc[:, cntstart:]

    rmss = GMu.get_rms_df(x, y)
    ccs = 1 - GMu.get_cc_df(x, y)
    # wss = GMu.get_wasserstein_dist_df(x, y)

    if plot:
        plt.figure()
        plt.boxplot([maxrmss, signrms1s, signrms2s, rmss, ccs],
            labels=['maxval RMS', 'Sign1 RMS', 'Sign2 RMS', 'RMS', '1-CC'])
        plt.ylim((-0.01, 1.01))
        plt.savefig(os.path.join(outputdir, 'general.png'))

        if predirectory:
            plt.savefig('%s_%s' % (outputdir, 'general.png'))

    maxdiff = predDF.iloc[:, 0].values - yEval.iloc[:, 0].values
    sign1diff = predDF.iloc[:, 1].values - yEval.iloc[:, 1].values
    sign2diff = predDF.iloc[:, 2].values - yEval.iloc[:, 2].values

    meandiff = (x - y).mean(axis=1)
    stddiff = (x - y).std(axis=1)

    if plot:
        plt.figure()
        vals = [maxdiff, sign1diff, sign2diff, meandiff, stddiff]
        labels = ['Max', 'Sign1', 'Sign2', 'Mean-WV', 'Std-WV']
        positions = num.linspace(0, len(vals), len(vals))
        print(len(vals))
        print(positions)
        parts = plt.violinplot(vals,
            points=30,
            positions=positions,
            # points=points, 
            showmedians=False, showextrema=False,
            # widths=widths
            )
        
        plt.axhline(0, color='black', linestyle='-', alpha=0.25, zorder=-2)
        plt.xticks(positions, labels=labels)
        for pc in parts['bodies']:
            pc.set_facecolor('grey')
            pc.set_edgecolor(None)
            pc.set_alpha(1)

        plt.savefig(os.path.join(outputdir, 'violin.png'))

    return maxdiff, sign1diff, sign2diff, meandiff, stddiff


def line_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.05, azimuth=0, log=True):
    coords = []
    dcor = min(mapextent[1], mapextent[0])

    if log:
        r = num.logspace(num.log10(rmin), num.log10(dcor), ncoords)
    else:
        r = num.linspace(rmin, dcor, ncoords)

    theta = (azimuth * (num.pi / 180))

    R, Theta = num.meshgrid(r, theta)
    lons = R * num.cos(Theta) + source.lon
    lats = R * num.sin(Theta) + source.lat

    for lon, lat in zip(lons.flatten(), lats.flatten()):
        coords.append([lon, lat])

    coords = num.array(coords)
    # print(coords)

    # plt.figure()
    # plt.plot(coords.T[0], coords.T[1], '*b')
    # plt.axis('equal')
    # plt.savefig('/home/lehmann/dr/plots/test_line_mapping.png')
    # exit()
    return coords


def rebuild_time_nn(data):

    cntstart = 0
    addcol = []
    if 'maxval' in data:
        cntstart += 1
        addcol.append('maxval')

    if 'maxval-sign' in data:
        cntstart += 1
        addcol.append('maxval-sign')

    elif 'maxval-sign-pos' in data:
        cntstart += 2
        addcol.append('maxval-sign-pos')
        addcol.append('maxval-sign-neg')

    print(data)

    trMax = (10**data['maxval'])

    trSign = num.zeros(len(trMax))
    if 'maxval-sign-pos' in data:
        trSign[data['maxval-sign-pos'] > data['maxval-sign-neg']] = 1
        trSign[data['maxval-sign-pos'] < data['maxval-sign-neg']] = -1

    selcols = [col for col in data.columns if col not in addcol]

    trueTr = data[selcols]
    trueTr = trueTr.multiply(trMax, axis=0)
    if 'maxval-sign-pos' in data:
        trueTr = trueTr.multiply(trSign, axis=0)

    return trueTr
