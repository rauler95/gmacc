import os

import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

from openquake.hazardlib.geo import geodetic
from pyrocko import orthodrome

from gmacc.gmeval import util as GMu


def read_addtional_information(filepath):
    adddict = {}
    with open(filepath) as f:
        for line in f:
            if ':' in line:
                line = line.rsplit()
                adddict[line[0].replace(':', '')] = float(line[1])
                print(line)

    return adddict


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
        # pred -= pred[cntstart]
        # pred /= max(abs(pred))
        true = yEval.iloc[mm]

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        xvals = range(len(pred) - cntstart)
        ax1.plot(xvals, pred[cntstart:], label='NN Predicted')
        ax1.plot(xvals, true[cntstart:], label='GF Ouput')
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

        elif 'maxval-sign' in yEval:
            trMaxpred *= num.sign(pred[1])
            trMaxtrue *= num.sign(true[1])

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

    valslist = []
    labelslist = []
    cntstart = 0
    if 'maxval' in yEval:
        cntstart += 1
        maxrmss = predDF.iloc[:, 0].values - yEval.iloc[:, 0].values
        valslist.append(maxrmss)
        labelslist.append('maxval RMS')

    if 'maxval-sign-pos' in yEval:
        cntstart += 2

        signrms1s = predDF.iloc[:, 1].values - yEval.iloc[:, 1].values
        signrms2s = predDF.iloc[:, 2].values - yEval.iloc[:, 2].values
        valslist.append(signrms1s)
        labelslist.append('sign1 RMS')
        valslist.append(signrms2s)
        labelslist.append('sign2 RMS')

    elif 'maxval-sign' in yEval:
        cntstart += 1
        signrmss = predDF.iloc[:, 1].values - yEval.iloc[:, 1].values
        valslist.append(signrmss)
        labelslist.append('sign RMS')
    
    x = predDF.iloc[:, cntstart:]
    y = yEval.iloc[:, cntstart:]

    rmss = GMu.get_rms_df(x, y)
    valslist.append(rmss)
    labelslist.append('RMS')

    ccs = 1 - GMu.get_cc_df(x, y)
    valslist.append(ccs)
    labelslist.append('1-CC')

    if plot:
        plt.figure()
        plt.boxplot(list(num.abs(valslist)),
            labels=labelslist)
        plt.ylim((-0.01, 1.01))
        plt.savefig(os.path.join(outputdir, 'general.png'))

        if predirectory:
            plt.savefig('%s_%s' % (outputdir, 'general.png'))

    # maxdiff = predDF.iloc[:, 0].values - yEval.iloc[:, 0].values
    # sign1diff = predDF.iloc[:, 1].values - yEval.iloc[:, 1].values
    # sign2diff = predDF.iloc[:, 2].values - yEval.iloc[:, 2].values

    meandiff = (x - y).mean(axis=1)
    stddiff = (x - y).std(axis=1)
    valslist.append(meandiff)
    labelslist.append('mean-WV')
    valslist.append(stddiff)
    labelslist.append('Std-WV')

    if plot:
        plt.figure()
        vals = valslist
        labels = labelslist
        positions = num.linspace(0, len(vals), len(vals))

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

    return valslist, labelslist


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

        trMax = (10**data['maxval'])

    if 'maxval-sign' in data:
        cntstart += 1
        addcol.append('maxval-sign')

    elif 'maxval-sign-pos' in data:
        cntstart += 2
        addcol.append('maxval-sign-pos')
        addcol.append('maxval-sign-neg')

        trSign = num.zeros(len(trMax))
        trSign[data['maxval-sign-pos'] > data['maxval-sign-neg']] = 1
        trSign[data['maxval-sign-pos'] < data['maxval-sign-neg']] = -1

    selcols = [col for col in data.columns if col not in addcol]

    trueTr = data[selcols]
    if 'maxval' in data:
        trueTr = trueTr.multiply(trMax, axis=0)

    if 'maxval-sign-pos' in data:
        trueTr = trueTr.multiply(trSign, axis=0)

    return trueTr


def prepare_NN_predf(coords, mag, strike, dip, rake, depth, duration,
        hypolon, hypolat):

    data = {}
    lenfac = len(coords)
    data['magnitude'] = [mag] * lenfac
    data['strike'] = [strike] * lenfac
    data['dip'] = [dip] * lenfac
    data['rake'] = [rake] * lenfac
    data['ev_depth'] = [depth] * lenfac
    data['src_duration'] = [duration] * lenfac 

    lons, lats = num.array(coords).T
    r_hypos = geodetic.distance(hypolon, hypolat, depth, lons, lats, 0.)
    azimuths = orthodrome.azimuth_numpy(num.array(hypolat), num.array(hypolon), num.array(lats), num.array(lons))
    azimuths[azimuths > 180] = azimuths[azimuths > 180] - 360.

    data['azimuth'] = azimuths
    data['rhypo'] = r_hypos

    data = pd.DataFrame(data)

    return data


from gmacc.nnsynth import neural_networks as GMnn
from gmacc.nnsynth import preprocessing as GMpre
def get_NN_predwv(alldata, model, scalingDict, targets):

    alldata = GMpre.calc_azistrike(alldata, strikecol='strike',
        azimuthcol='azimuth', azistrikecol='azistrike', delete=False)
    dropcols = ['azimuth', 'strike']
    alldata = alldata.drop(columns=dropcols)
    alldata = GMpre.convert_distances(alldata)
    alldata = GMpre.normalize(scalingDict, alldata, mode='forward')
    allpredDF = GMnn.get_predict_df(model, alldata, targets, batchsize=10000)
    allpredDF = rebuild_time_nn(allpredDF)

    return allpredDF


def inital_random_params(num_srcs, coords, hypolon, hypolat):
    from pyrocko import moment_tensor as pmt

    evaldict = {}
    datas = []
    for ii in range(num_srcs):

        testdict = {}

        mag = num.random.uniform(5.0, 7.5)
        (strike, dip, rake) = pmt.random_strike_dip_rake()
        depth = num.random.uniform(0.1, 10.)
        duration = GMu.calc_rupture_duration(mag=mag, rake=rake)

        data = prepare_NN_predf(coords, mag, strike, dip, rake, depth, duration,
            hypolon, hypolat)
        datas.append(data)

        testdict = {
            'mag': mag,
            'strike': strike,
            'dip': dip,
            'rake': rake,
            'depth': depth,
            'src_duration': duration,
        }
        evaldict[ii] = testdict

    return datas, evaldict


def nn_evaluation_score(x, y):

    rmss = num.mean(GMu.get_rms_df_asym(x, y), axis=1)
    ccs = num.mean(GMu.get_cc_df_asym(x, y), axis=1)
    return rmss, ccs


def iterative_params(bestDF, coords, hypolon, hypolat, fac=10):
    evaldict = {}
    datas = []

    cnt = 0
    for it, row in bestDF.iterrows():
        # print(row)
        for ff in range(fac):
            cnt += 1
            testdict = {}

            if ff == 0:
                mag = row['mag']
                strike = row['strike']
                dip = row['dip']
                rake = row['rake']
                depth = row['depth']
                duration = row['src_duration']
            else:
                mag = num.random.normal(row['mag'], 0.1)
                strike = num.random.normal(row['strike'], 5)
                dip = num.random.normal(row['dip'], 5)
                rake = num.random.normal(row['rake'], 5)
                depth = num.random.normal(row['depth'], 2)
                duration = num.random.normal(row['src_duration'], 5)

            data = prepare_NN_predf(coords, mag, strike, dip, rake, depth, duration,
                hypolon, hypolat)
            datas.append(data)

            testdict = {
                'mag': mag,
                'strike': strike,
                'dip': dip,
                'rake': rake,
                'depth': depth,
                'src_duration': duration,
            }
            evaldict[cnt] = testdict
        # print(datas)
        # exit()

    return datas, evaldict


def own_inversion(refDF, model, scalingDict, targets, numiter, num_srcs, coords, plotdir):

    numselecttop = int(0.01 * num_srcs)
    searchfac = int(num_srcs / numselecttop)
    print(numselecttop, searchfac)

    hypolon = 0.
    hypolat = 0.

    bestDF = False
    for ii in range(numiter):
        print('\nNew iter %s' % ii)
        if ii == 0:
            datas, evaldict = inital_random_params(num_srcs, coords, hypolon, hypolat)
        else:
            datas, evaldict = iterative_params(bestDF, coords, hypolon, hypolat, searchfac)

        alldata = pd.concat(datas, ignore_index=True)
        allpredDF = get_NN_predwv(alldata, model, scalingDict, targets)

        rmss, ccs = nn_evaluation_score(refDF, allpredDF)

        evalDF = pd.DataFrame(evaldict).T
        # print(evalDF)
        evalDF['rms'] = rmss
        evalDF['cc'] = 1 - ccs
        evalDF['rmscc'] = evalDF['cc'] * evalDF['rms']

        score = 'rmscc'
        # score = 'cc'
        # score = 'rms'
        bestDF = evalDF.nsmallest(numselecttop, score, keep='all')
        # print(bestDF)
        bestsrcparams = bestDF.nsmallest(1, score, keep='all')
        print(bestsrcparams)

        for col in evalDF.columns:
            if col in ['rms', 'cc', 'rmscc']:
                continue

            # plt.figure()
            fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

            ax1c = 'red'
            ax1.semilogy(bestDF[col], bestDF['rms'], 'o', color='black', alpha=1)
            ax1.semilogy(evalDF[col], evalDF['rms'], '.', label='RMS', color=ax1c)
            ax1.set_ylabel('RMS', color=ax1c)
            ax1.set_xlabel(col)

            # ax2 = ax1.twinx()
            ax2c = 'green'
            ax2.plot(bestDF[col], bestDF['cc'], 'o', color='black', alpha=1)
            ax2.plot(evalDF[col], evalDF['cc'], '.', label='cc', color=ax2c)
            ax2.set_ylabel('1 - CC', color=ax2c)
            ax2.set_xlabel(col)

            ax3c = 'blue'
            
            ax3.semilogy(bestDF[col], bestDF['rmscc'], 'o', color='black', alpha=1)
            ax3.semilogy(evalDF[col], evalDF['rmscc'], '.', label='rmscc', color=ax3c)
            ax3.set_ylabel('RMSCC', color=ax3c)
            ax3.set_xlabel(col)

            plt.tight_layout()
            fig.savefig(os.path.join(plotdir, '%s_%s.png' % (col, ii)))
            plt.close()

    return bestsrcparams, bestDF