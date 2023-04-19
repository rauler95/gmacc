import os
import time

import numpy as num
import matplotlib.pyplot as plt
import pandas as pd

from openquake.hazardlib.geo import geodetic
from pyrocko import orthodrome

from gmacc.gmeval import util as GMu
from gmacc.nnsynth import neural_networks as GMnn
from gmacc.nnsynth import preprocessing as GMpre

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

    lenfac = len(coords)
    # data = {}
    # data['magnitude'] = [mag] * lenfac
    # data['strike'] = [strike] * lenfac
    # data['dip'] = [dip] * lenfac
    # data['rake'] = [rake] * lenfac
    # data['ev_depth'] = [depth] * lenfac
    # data['src_duration'] = [duration] * lenfac 

    lons, lats = num.array(coords).T
    r_hypos = geodetic.distance(hypolon, hypolat, depth, lons, lats, 0.)
    azimuths = geodetic.azimuth(hypolon, hypolat, lons, lats)
    azimuths[azimuths > 180] = azimuths[azimuths > 180] - 360.

    # data['azimuth'] = azimuths
    # data['rhypo'] = r_hypos

    data = {
        'magnitude': [mag] * lenfac,
        'strike': [strike] * lenfac,
        'dip': [dip] * lenfac,
        'rake': [rake] * lenfac,
        'ev_depth': [depth] * lenfac,
        'src_duration': [duration] * lenfac,
        'azimuth': azimuths,
        'rhypo': r_hypos
    }
    data = pd.DataFrame(data)

    return data


def get_NN_predwv(alldata, model, scalingDict, targets):

    alldata = GMpre.calc_azistrike(alldata, strikecol='strike',
        azimuthcol='azimuth', azistrikecol='azistrike', delete=False)
    dropcols = ['azimuth', 'strike']
    alldata = alldata.drop(columns=dropcols)
    alldata = GMpre.convert_distances(alldata)
    alldata = GMpre.normalize(scalingDict, alldata, mode='forward')
    allpredDF = GMnn.get_predict_df(model, alldata, targets, batchsize=10000)
    allpredDF = rebuild_time_nn(allpredDF)

    # print(allpredDF)
    # allpredDF = allpredDF.diff(axis=1)
    # allpredDF = allpredDF.fillna(0)
    # allpredDF = allpredDF.drop(columns='0')
    # print(allpredDF)
    # exit()

    return allpredDF


def inital_random_params(num_srcs, coords, hypolonguess, hypolatguess):
    from pyrocko import moment_tensor as pmt

    # t1 = time.time()
    # evaldict = {}
    # datas = []
    # for ii in range(num_srcs):

    #     mag = num.random.uniform(5.0, 7.5)
    #     (strike, dip, rake) = pmt.random_strike_dip_rake()
    #     depth = num.random.uniform(0.1, 10.)
    #     duration = GMu.calc_rupture_duration(mag=mag, rake=rake)
    #     hypolon = num.random.normal(hypolonguess, 0.25)
    #     hypolat = num.random.normal(hypolatguess, 0.25)

    #     data = prepare_NN_predf(coords, mag, strike, dip, rake, depth, duration,
    #         hypolon, hypolat)
    #     datas.append(data)

    #     evaldict[ii] = {
    #         'mag': mag,
    #         'strike': strike,
    #         'dip': dip,
    #         'rake': rake,
    #         'depth': depth,
    #         'src_duration': duration,
    #         'lon': hypolon,
    #         'lat': hypolat,
    #     }
    # # print(datas)
    # alldata = pd.concat(datas, ignore_index=True)
    # print(alldata.shape)
    # print('t1', time.time() - t1)
    # evalDF = pd.DataFrame(evaldict).T
    # print(evalDF)

    # t2 = time.time()
    evaldict = {}
    alldatas = {}

    lenfac = len(coords)

    mag = num.random.uniform(5.0, 7.5, num_srcs)
    depth = num.random.uniform(0.1, 10., num_srcs)
    hypolon = num.random.normal(hypolonguess, 0.25, num_srcs)
    hypolat = num.random.normal(hypolatguess, 0.25, num_srcs)

    strike = num.random.uniform(0, 360, num_srcs)
    dip = num.random.uniform(0, 90, num_srcs)
    rake = num.random.uniform(-180, 180, num_srcs)

    # t3 = time.time()
    duration = []
    for ll in range(len(mag)):
        dur = GMu.calc_rupture_duration(mag=mag[ll], rake=rake[ll])
        duration.append(dur)
    # print('duration calculation', time.time() - t3)

    evaldict = {
        'magnitude': mag,
        'strike': strike,
        'dip': dip,
        'rake': rake,
        'depth': depth,
        'duration': duration,
        'lon': hypolon,
        'lat': hypolat,
    }

    alldatas['magnitude'] = num.repeat(mag, lenfac)
    alldatas['strike'] = num.repeat(strike, lenfac)
    alldatas['dip'] = num.repeat(dip, lenfac)
    alldatas['rake'] = num.repeat(rake, lenfac)
    alldatas['src_duration'] = num.repeat(duration, lenfac)
    alldatas['ev_depth'] = num.repeat(depth, lenfac)
    hypolats = num.repeat(hypolat, lenfac)
    hypolons = num.repeat(hypolon, lenfac)

    lons, lats = num.array(coords).T
    lons = list(lons) * len(mag)
    lats = list(lats) * len(mag)
    r_hypos = geodetic.distance(hypolons, hypolats, alldatas['ev_depth'], lons, lats, 0.)
    azimuths = geodetic.azimuth(hypolons, hypolats, lons, lats)

    alldatas['rhypo'] = r_hypos
    alldatas['azimuth'] = azimuths

    alldatas = pd.DataFrame(alldatas)
    evalDF = pd.DataFrame(evaldict)
    # print('t2', time.time() - t2)

    return alldatas, evalDF


def iterative_params(bestDF, coords, fac=10, coolingfac=1):
    cnt = 0

    # t2 = time.time()
    evaldict = {}
    alldatas = {}

    lenfac = len(coords)
    mag = []
    depth = []
    hypolon = []
    hypolat = []
    strike = []
    rake = []
    dip = []
    duration = []
    for it, row in bestDF.iterrows():
        # print(row)
        for ff in range(fac):
            cnt += 1

            if ff == 0:
                mags = row['magnitude']
                strikes = row['strike']
                dips = row['dip']
                rakes = row['rake']
                depths = row['depth']
                durations = row['duration']
                hlon = row['lon']
                hlat = row['lat']
                # hlon = 0.
                # hlat = 0.
            else:
                magf = False
                while not magf:
                    mags = num.random.normal(row['magnitude'], 0.2 * coolingfac)
                    if mags >= 5.0 and mags <= 7.5:
                        magf = True

                depthf = False
                while not depthf:
                    depths = num.random.normal(row['depth'], 2 * coolingfac)
                    if depths > 0.0 and depths <= 10.0:
                        depthf = True
                
                durationf = False
                while not durationf:
                    durations = num.random.normal(row['duration'], 5 * coolingfac)
                    if durations > 0.0:
                        durationf = True

                # rakes = num.random.normal(row['rake'], 5 * coolingfac)
                rakef = False
                while not rakef:
                    rakes = num.random.normal(row['rake'], 5 * coolingfac)
                    if rakes >= -180.0 and rakes <= 180.:
                        rakef = True

                # dips = num.random.normal(row['dip'], 5 * coolingfac)
                dipf = False
                while not dipf:
                    dips = num.random.normal(row['dip'], 5 * coolingfac)
                    if dips >= -180.0 and dips <= 180.:
                        dipf = True

                strikes = num.random.normal(row['strike'], 5 * coolingfac)
                # if strikes > 360:
                #     strikes -= 360
                # elif strikes < 0.0:
                #     strikes += 360

                hlon = num.random.normal(row['lon'], 0.25 * coolingfac)
                hlat = num.random.normal(row['lat'], 0.25 * coolingfac)
                # hlon = 0.
                # hlat = 0.

            mag.append(mags)
            depth.append(depths)
            hypolon.append(hlon)
            hypolat.append(hlat)
            strike.append(strikes)
            dip.append(dips)
            rake.append(rakes)
            duration.append(durations)

    evaldict = {
        'magnitude': mag,
        'strike': strike,
        'dip': dip,
        'rake': rake,
        'depth': depth,
        'duration': duration,
        'lon': hypolon,
        'lat': hypolat,
    }

    alldatas['magnitude'] = num.repeat(mag, lenfac)
    alldatas['strike'] = num.repeat(strike, lenfac)
    alldatas['dip'] = num.repeat(dip, lenfac)
    alldatas['rake'] = num.repeat(rake, lenfac)
    alldatas['src_duration'] = num.repeat(duration, lenfac)
    alldatas['ev_depth'] = num.repeat(depth, lenfac)
    hypolats = num.repeat(hypolat, lenfac)
    hypolons = num.repeat(hypolon, lenfac)

    lons, lats = num.array(coords).T
    lons = list(lons) * len(mag)
    lats = list(lats) * len(mag)
    r_hypos = geodetic.distance(hypolons, hypolats, alldatas['ev_depth'], lons, lats, 0.)
    azimuths = geodetic.azimuth(hypolons, hypolats, lons, lats)

    alldatas['rhypo'] = r_hypos
    alldatas['azimuth'] = azimuths

    alldatas = pd.DataFrame(alldatas)
    evalDF = pd.DataFrame(evaldict)
    # print('t2', time.time() - t2)
    # print(alldatas.shape)
    # print(evalDF)

    return alldatas, evalDF


def nn_evaluation_score(x, y):
    rmss = num.mean(GMu.get_rms_df_asym_once(x, y), axis=1)
    ccs = num.mean(GMu.get_cc_df_asym_once(x, y), axis=1)

    return rmss, ccs


def own_inversion(refDF, model, scalingDict, targets, numiter, num_srcs, bestpercentage,
        hypolonguess, hypolatguess, coords, plotdir, plot=False):

    # numselecttop = int(0.01 * num_srcs)
    numselecttop = int(bestpercentage * num_srcs)
    # numselecttop = int(0.2 * num_srcs)
    searchfac = int(num_srcs / numselecttop)
    print(numselecttop, searchfac)

    bestDFall = pd.DataFrame()

    bestDF = False
    for ii in range(numiter):
        print('\nNew iter %s' % ii)

        t1 = time.time()
        if ii == 0:
            alldata, evalDF = inital_random_params(num_srcs, coords, 
                hypolonguess, hypolatguess)
        else:
            # coolingfac = num.sqrt(num.sqrt(1 / ii))
            # coolingfac = num.sqrt(1 / ii)
            coolingfac = 1 / ii
            # coolingfac = 2 / ii
            print(coolingfac)
            alldata, evalDF = iterative_params(bestDF, coords,
               searchfac, coolingfac=coolingfac)
            # alldata = pd.concat(datas, ignore_index=True)
        print('Time-sources', time.time() - t1)

        t2 = time.time()
        allpredDF = get_NN_predwv(alldata, model, scalingDict, targets)
        print('Time-prediction', time.time() - t2)

        t3 = time.time()
        rmss, ccs = nn_evaluation_score(refDF, allpredDF)
        print('Time-scores', time.time() - t3)
        
        # print(evalDF)
        evalDF['rms'] = rmss
        evalDF['cc'] = 1 - ccs
        evalDF['rmscc'] = evalDF['cc'] * evalDF['rms']
        # evalDF['rmscc'] = evalDF['cc'] + (evalDF['rms'] * 10)
        # evalDF['rmscc'] = (10 * evalDF['cc']**2) + evalDF['rms']
        # evalDF['rmscc'] = (10 * evalDF['cc']**2) * evalDF['rms']
        # evalDF['rmscc'] = evalDF['cc']**2 + evalDF['rms']
        # evalDF['rmscc'] = evalDF['cc']**2 * evalDF['rms']

        score = 'rmscc'
        # score = 'cc'
        # score = 'rms'
        bestDF = evalDF.nsmallest(numselecttop, score, keep='all')
        # print(bestDF)
        bestsrcparams = bestDF.nsmallest(1, score, keep='all')
        print(bestsrcparams)

        # bestDFall = pd.concat([bestDFall, bestDF], ignore_index=True)
        bestDFall = pd.concat([bestDFall, bestDF], ignore_index=True)

        if plot:
            bidx = bestsrcparams.index[0]
            lenfac = len(coords)
            bestDFwv = allpredDF.iloc[bidx * lenfac: (bidx + 1) * lenfac]
            fig, axs = plt.subplots(len(bestDFwv), 1, figsize=(16, 16), sharex=True)#, hspace=0.)
            for ww in range(len(bestDFwv)):
                axs[ww].plot(bestDFwv.iloc[ww], label='pred')
                axs[ww].plot(refDF.iloc[ww], label='ref')

            plt.legend()
            plt.tight_layout()
            # plt.show()
            fig.savefig(os.path.join(plotdir, 'waveform_%s.png' % (ii)))
            # exit()

        #     for col in evalDF.columns:
        #         if col in ['rms', 'cc', 'rmscc']:
        #             continue

        #         # plt.figure()
        #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))

        #         ax1c = 'red'
        #         ax1.semilogy(bestDF[col], bestDF['rms'], 'o', color='black', alpha=1)
        #         ax1.semilogy(evalDF[col], evalDF['rms'], '.', label='RMS', color=ax1c)
        #         ax1.set_ylabel('RMS', color=ax1c)
        #         ax1.set_xlabel(col)

        #         # ax2 = ax1.twinx()
        #         ax2c = 'green'
        #         ax2.plot(bestDF[col], bestDF['cc'], 'o', color='black', alpha=1)
        #         ax2.plot(evalDF[col], evalDF['cc'], '.', label='cc', color=ax2c)
        #         ax2.set_ylabel('1 - CC', color=ax2c)
        #         ax2.set_xlabel(col)

        #         ax3c = 'blue'
                
        #         ax3.semilogy(bestDF[col], bestDF['rmscc'], 'o', color='black', alpha=1)
        #         ax3.semilogy(evalDF[col], evalDF['rmscc'], '.', label='rmscc', color=ax3c)
        #         ax3.set_ylabel('RMSCC', color=ax3c)
        #         ax3.set_xlabel(col)

        #         plt.tight_layout()
        #         fig.savefig(os.path.join(plotdir, '%s_%s.png' % (col, ii)))
        #         plt.close()

    # if plot:
    #     print('\nPlotting')
    #     print(bestDFall)
    #     import matplotlib as mpl
    #     from matplotlib import cm

    #     norm = mpl.colors.Normalize(vmin=0, vmax=numiter)
    #     cmap = cm.get_cmap('hsv')
        
    #     for col in bestDFall.columns:
    #         fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 8))
    #         for ni in range(numiter):
    #             # if ni == numiter:
    #             #     continue
    #             idx1 = numselecttop * ni
    #             idx2 = numselecttop * (ni + 1)
    #             color = cmap(norm(ni))
    #             # print(ni, numselecttop)
    #             # print(color)
    #             # print(bestDFall[col])
    #             # print(idx1)
    #             # print(idx2)
    #             # print(bestDFall[col].iloc[idx1:idx2])
    #             # exit()

    #             # ax1c = 'red'
    #             # ax1.semilogy(bestDFall[col], bestDFall['rms'], 'o', color='black', alpha=1)
    #             ax1.semilogy(bestDFall[col].iloc[idx1:idx2], bestDFall['rms'].iloc[idx1:idx2], '.', label='RMS', color=color)
    #             ax1.set_ylabel('RMS')
    #             ax1.set_xlabel(col)

    #             # ax2 = ax1.twinx()
    #             # ax2c = 'green'
    #             # ax2.plot(bestDFall[col], bestDFall['cc'], 'o', color='black', alpha=1)
    #             ax2.plot(bestDFall[col].iloc[idx1:idx2], bestDFall['cc'].iloc[idx1:idx2], '.', label='cc', color=color)
    #             ax2.set_ylabel('1 - CC')
    #             ax2.set_xlabel(col)

    #             # ax3c = 'blue'
    #             # ax3.semilogy(bestDFall[col], bestDFall['rmscc'], 'o', color='black', alpha=1)
    #             ax3.semilogy(bestDFall[col].iloc[idx1:idx2], bestDFall['rmscc'].iloc[idx1:idx2], '.', label='rmscc', color=color)
    #             ax3.set_ylabel('RMSCC')
    #             ax3.set_xlabel(col)

    #         # cb1 = mpl.colorbar.ColorbarBase(ax2, cmap=cmap,
    #         #                     norm=norm,
    #         #                     orientation='horizontal')
    #         # cb1.set_label('Some Units')
    #         fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    #         plt.tight_layout()
    #         fig.savefig(os.path.join(plotdir, '%s.png' % (col)))

    return bestsrcparams, bestDFall