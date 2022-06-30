import os
import time
import copy

import matplotlib.pyplot as plt
import numpy as num


from pyrocko import orthodrome, trace, gf, io
from pyrocko import moment_tensor as pmt

from openquake.hazardlib.geo import geodetic, Point, Mesh, PlanarSurface


from gmacc.gmeval.sources import StationGMClass, ComponentGMClass, GMClass,\
            StationContainer, own_differentation, create_stationdict_synthetic


###
# Classes
###
class StationContainerObservation(StationContainer):

    def calc_rupture_azimuths(self):

        lons = []
        lats = []
        for ii in self.stations:
            lons.append(self.stations[ii].lon)
            lats.append(self.stations[ii].lat)

        mesh = Mesh(num.array(lons), num.array(lats))

        rupAzis = self.refSource.surface.get_azimuth_of_closest_point(mesh)
        centroidAzis = self.refSource.surface.get_azimuth(mesh)

        for nn, ii in enumerate(self.stations):
            self.stations[ii].rup_azimuth = float(rupAzis[nn])
            self.stations[ii].centre_azimuth = float(centroidAzis[nn])

        return

    def calc_h2_from_gm(self, delete=True, imts=['pgv', 'pga'], SDOFs=[0.3, 1, 3]):
        '''
        To do:
        - Vectorsum of two max values?
        '''

        for sta in self.stations:
            for comp in self.stations[sta].components:
                delGM = []
                for gm in self.stations[sta].components[comp].gms:

                    SAs = []
                    for freq in SDOFs:
                        SAs.append('sa(%s)' % (freq))

                    if gm in SAs or gm in imts:
                        pass
                    else:
                        delGM.append(gm)

                for gm in delGM:
                    del self.stations[sta].components[comp].gms[gm]

            if 'E' in self.stations[sta].components.keys() and \
                'N' in self.stations[sta].components.keys():

                COMP = ComponentGMClass(component='H', gms={})

                for gm in self.stations[sta].components['E'].gms:
                    val1 = self.stations[sta].components['E'].gms[gm].value
                    val2 = self.stations[sta].components['N'].gms[gm].value

                    ## max
                    val = max(val1, val2)
                    ## vectorsum, resonable for only max values??
                    # val = num.sqrt(val1**2 + val2**2)

                    GM = GMClass(name=gm, value=val,
                        unit=self.stations[sta].components['E'].gms[gm].unit)

                    COMP.gms[gm] = GM
                self.stations[sta].components['H'] = COMP

                if delete:
                    del self.stations[sta].components['N']
                    del self.stations[sta].components['E']

            if '1' in self.stations[sta].components.keys() and \
                '2' in self.stations[sta].components.keys():

                COMP = ComponentGMClass(component='H', gms={})

                for gm in self.stations[sta].components['1'].gms:
                    val1 = self.stations[sta].components['1'].gms[gm].value
                    val2 = self.stations[sta].components['2'].gms[gm].value

                    ## max
                    val = max(val1, val2)
                    ## vectorsum, resonable for only max values??
                    # val = num.sqrt(val1**2 + val2**2)

                    GM = GMClass(name=gm, value=val,
                        unit=self.stations[sta].components['1'].gms[gm].unit)

                    COMP.gms[gm] = GM
                self.stations[sta].components['H'] = COMP

                if delete:
                    del self.stations[sta].components['1']
                    del self.stations[sta].components['2']

        return

    def remove_distant_stations(self, lonmax, latmax):
        rmSta = []
        source = self.refSource
        for sta in self.stations:
            lon = self.stations[sta].lon
            lat = self.stations[sta].lat

            lonCond = (source.lon + lonmax) >= lon >= (source.lon - lonmax)
            latCond = (source.lat + latmax) >= lat >= (source.lat - latmax)

            if lonCond is False or latCond is False:
                rmSta.append(sta)

        for sta in rmSta:
            del self.stations[sta]

    def remove_distant_stations_radius(self, refdist, mindist=0.):
        '''
        Dist in km
        '''
        rmSta = []
        source = self.refSource
        for sta in self.stations:
            lon = self.stations[sta].lon
            lat = self.stations[sta].lat
            dist = orthodrome.distance_accurate50m_numpy(source.lat, source.lon,
                                                        lat, lon)[0] / 1000.

            if dist > refdist:
                rmSta.append(sta)

            if dist < mindist:
                rmSta.append(sta)

        rmSta = list(set(rmSta))
        for sta in rmSta:
            del self.stations[sta]

    def select_stations(self, stationList):
        rmSta = []
        for sta in self.stations:
            if sta not in stationList:
                rmSta.append(sta)

        for sta in rmSta:
            del self.stations[sta]

    def remove_stations(self, stationList):
        rmSta = []
        for sta in self.stations:
            if sta in stationList:
                rmSta.append(sta)

        for sta in rmSta:
            del self.stations[sta]

    def remove_stations_without_components(self, componentList):
        rmSta = []
        rawlen = len(componentList)
        if 'E' in componentList and 'N' in componentList:
            cmpList = componentList + ['0', '1', '2']
        else:
            cmpList = copy.deepcopy(componentList)

        for sta in self.stations:
            failcnt = []
            for cha in cmpList:
                if cha in self.stations[sta].components.keys():
                    failcnt.append(True)
                else:
                    failcnt.append(False)

            if failcnt.count(True) < rawlen:
                rmSta.append(sta)

        for sta in rmSta:
            del self.stations[sta]

    def remove_notneeded_components(self, componentList):
        if 'E' in componentList and 'N' in componentList:
            cmpList = componentList + ['0', '1', '2']
        else:
            cmpList = copy.deepcopy(componentList)

        for sta in self.stations:
            comps = list(self.stations[sta].components.keys())
            for comp in comps:
                if comp not in cmpList:
                    del self.stations[sta].components[comp]

    def remove_short_traces(self, mint):
        # src = self.refSource
        for sta in self.stations:
            comps = list(self.stations[sta].components.keys())
            for comp in comps:
                tr = self.stations[sta].components[comp].trace
                diffT = tr.tmax - tr.tmin
                if diffT < mint:
                    del self.stations[sta].components[comp]

    def compare_to_flatfile(self, file):
        import pandas as pd
        flat = pd.read_csv(file, sep=';')
        print(flat)
        print(self.refSource.name)
        eqname = self.refSource.name
        if eqname == 'us1000731j':
            evid = 'EMSC-20161030_0000029'
        elif eqname == 'us10006g7d':
            evid = 'EMSC-20160824_0000006'
        elif eqname == 'ci38457511':
            evid = 'EMSC-20190706_0000043'
        else:
            evid = self.refSource.name

        data = flat[flat['event_id'] == evid]
        data['ns'] = data[['network_code', 'station_code']].agg('.'.join, axis=1)
        print(data)

        self.calc_distances()

        for sta in self.stations:
            if sta in data['ns'].to_list():
                print()
                print(sta)
                # print(sta, num.round(self.stations[sta].rhypo, 2), num.round(self.stations[sta].rrup, 2))
                # for gm in ['pga', 'pgv', 'pgd']:
                for gm in ['pgd']:
                    for comp in self.stations[sta].components.keys():
                        val = self.stations[sta].components[comp].gms[gm].value
                        # val = 10**(val)
                        if gm == 'pga':
                            val += 1
                        if comp == 'H':
                            comp = 'rotD100'
                        elif comp in ['E', 'N']:
                            continue
                        elif comp == 'Z':
                            comp = 'W'

                        ims = '%s_%s' % (comp, gm)
                        refval = float(data[data['ns'] == sta][ims])
                        refval = num.log10(num.abs(refval))
                        # print(gm, comp, val, refval)
                        print(num.round(val - refval, 3), gm, comp, )

                        # print(num.round(10**val, 3), num.round(10**refval, 3), num.round(val - refval, 3), gm, comp, )

    def add_gps_data(self, gpsfile):
        print(gpsfile)
        with open(gpsfile, 'r') as f:
            for line in f:
                line = line.rsplit()
                for sta in self.stations:
                    if sta.rsplit('.')[1] == line[0]:
                        print(sta, float(line[4]), float(line[6]), float(line[8]))
                        exit()
                        self.stations[sta].components['E'].gms['GPS'] = GMClass(name='GPS',
                                value=float(line[4]) / 100., unit='m')
                        self.stations[sta].components['N'].gms['GPS'] = GMClass(name='GPS',
                                value=float(line[6]) / 100., unit='m')
                        self.stations[sta].components['Z'].gms['GPS'] = GMClass(name='GPS',
                                value=float(line[8]) / 100., unit='m')

    def modify_disp_trace(self, vP=5.5, plot=True, mode='disp'):
        # from obspy.signal.trigger import pk_baer

        if mode not in ['ebasco', 'disp']:
            print('Wrong mode of tilt_correction: %s' % mode)
            exit()

        self.calc_distances()
        eq = self.refSource.name
        ot = self.refSource.time
        # vS = vP / num.sqrt(3.)

        if plot is True:
            plotdir = os.path.join(os.getcwd(), 'baseline_check_%s/processing/' % (eq))
            if not os.path.exists(plotdir):
                print(plotdir)
                os.makedirs(plotdir)

        delstas = []

        for sta in self.stations:
            # print(sta)
            staDict = self.stations[sta]

            rhypo = staDict.rhypo
            rrup = staDict.rrup
            if rrup is None:
                dist = rhypo
            else:
                dist = rrup

            pth = (rhypo / vP) + ot

            accZ = staDict.components['Z'].traces['acc'].copy()
            accE = staDict.components['E'].traces['acc'].copy()
            accN = staDict.components['N'].traces['acc'].copy()

            if not accZ.deltat == accE.deltat == accN.deltat:
                delstas.append(sta)
                continue    

            tmin = max(accZ.tmin, accE.tmin, accN.tmin)
            tmax = min(accZ.tmax, accE.tmax, accN.tmax)
            accZ.chop(tmin, tmax, include_last=True)
            accE.chop(tmin, tmax, include_last=True)
            accN.chop(tmin, tmax, include_last=True)

            ### idx2
            tilttime = get_tilt_point(accZ, accE, accN, sta=sta, eq=eq,
                            dist=dist, plot=False, mode='all')

            ### idx1 and idx3
            alldata = accZ.ydata**2 + accE.ydata**2 + accN.ydata**2

            cumsum = num.cumsum(alldata)
            idxdmax = num.where(cumsum > 0.99 * cumsum.max())[0][0]
            idxdmin = num.where(cumsum < 0.0001 * cumsum.max())[0]

            idxdminflag = True
            cnt = 0
            while idxdminflag:
                if len(idxdmin) > 10:
                    idxdmin = idxdmin[-1]
                    idxdminflag = False
                else:
                    idxdmin = num.where(cumsum < cnt * 0.001 * cumsum.max())[0]
                    cnt += 2

            dmax = accZ.deltat * idxdmax + accZ.tmin
            dmin = accZ.deltat * idxdmin + accZ.tmin

            if plot:
                print('Plot baseline correction')
                if mode == 'comp':
                    f, axes = plt.subplots(6, 1, sharex=True, figsize=(16, 9))
                elif mode == 'ebasco':
                    f, axes = plt.subplots(9, 1, sharex=True, figsize=(16, 9))
                else:
                    f, axes = plt.subplots(9, 1, sharex=True, figsize=(16, 9))
            else:
                axes = None
            
            tilt = None
            for comp in staDict.components:
                atr = staDict.components[comp].traces['acc'].copy()
                dt = atr.deltat
                amp = num.max(num.abs(atr.ydata))
                if amp > 0.75:
                    tilt = 'tilt'

            # if tilt == 'tilt':
            #     continue
            for comp in staDict.components:
                # if tilt != 'tilt':
                #     continue
                atr = staDict.components[comp].traces['acc'].copy()
                vtr = staDict.components[comp].traces['vel'].copy()
                dt = atr.deltat

                ### Acceleration correction
                idxdmax = int(max(((dmax) - vtr.tmin), 0) / dt)
                idxdmin = int(max(((dmin - 1) - vtr.tmin), 0) / dt)
                if idxdmin < 10:
                    idxdmin = int(max(((dmin) - vtr.tmin), 0) / dt)
                idxtilt = int(max(((tilttime) - vtr.tmin), 0) / dt)

                idx1 = idxdmin
                idx2 = idxtilt
                idx3 = idxdmax

                # try:
                vtr.ydata -= vtr.ydata[0]
                facs = num.polyfit(vtr.get_xdata()[:idx1] - vtr.tmin, vtr.ydata[:idx1], 1)
                m1 = facs[-2]
                atr.ydata[:idx2] -= m1
                # except TypeError as e:
                #     print('Error:', e)
                #     delstas.append(sta)
                #     break

                # if tilt != 'tilt':
                if True:
                    facs = num.polyfit(vtr.get_xdata()[idx3:] - vtr.tmin, vtr.ydata[idx3:], 1)
                    m2 = facs[-2]
                    atr.ydata[idx2:] -= m2

                atr.ydata -= atr.ydata[0]

                staDict.components[comp].traces['acc'] = atr
                staDict.components[comp].traces['vel'] = own_integrate(atr, 1)
                staDict.components[comp].traces['disp'] = own_integrate(atr, 2)

            for comp in staDict.components:
                if comp == 'H':
                    continue
                elif comp == 'N':
                    color = 'green'
                    # continue
                elif comp == 'E':
                    color = 'orange'
                    # continue
                elif comp == 'Z':
                    color = 'blue'

                rawatr = staDict.components[comp].traces['acc']
                atr = rawatr.copy()

                rawvtr = staDict.components[comp].traces['vel']
                vtr = rawvtr.copy()

                rawtr = staDict.components[comp].traces['disp']
                tr = rawtr.copy()

                dt = tr.deltat

                idxp = int(max((pth - tr.tmin), 0) / dt)

                if idxdmin > 250:
                    idxp = idxdmin
                elif idxp > 500:
                    pass
                else:
                    idxp = max(idxp, idxdmin)

                idxdmax = int(max(((dmax) - tr.tmin), 0) / dt)
                idxdmin = int(max(((dmin - 1) - tr.tmin), 0) / dt)
                if idxdmin < 10:
                    idxdmin = int(max(((dmin) - vtr.tmin), 0) / dt)
                idxtilt = int(max(((tilttime) - tr.tmin), 0) / dt)

                # idx1 = idxp
                idx1 = idxdmin
                idx2 = idxtilt
                idx3 = idxdmax

                # try:
                if mode == 'disp':
                    # tr = trace_baseline_correction(tr, vtr, comp, idx1, idx2, idx3, axes, tilt, plot=plot)
                    # plt.suptitle('Correction on Disp')
                    tr = trace_baseline_correction2(tr, vtr, comp, idx1, idx2, idx3, axes, tilt, plot=plot)
                    plt.suptitle('Correction on Disp2 !!!!!!!!!!!!!!')
                if mode == 'ebasco':
                    tr = trace_baseline_correction_ebasco(atr, comp, idx1, idx2, idx3, axes, tilt, plot=plot)
                    # tr = trace_baseline_correction_ebasco_own(atr, comp, idx1, idx2, idx3, axes, tilt, plot=plot)
                    plt.suptitle('Correction: eBASCO')

                if plot:
                    for axidx in [0, 1, 2, -3, -2, -1]:
                        ax = axes[axidx]

                        ax.axvline(dt * idx1 + tr.tmin, color='black', linestyle='-', zorder=-10)
                        ax.axvline(dt * idx2 + tr.tmin, color='red', linestyle='-', zorder=-10)
                        ax.axvline(dt * idx3 + tr.tmin, color='purple', linestyle='-', zorder=-10)

                        if axidx == 2:
                            ax.plot(rawtr.get_xdata(), rawtr.ydata, color=color, label=comp)
                            ax.set_ylabel('Disp\nRaw')
                            ax.axhline(y=0, color='gray', zorder=-10)

                        elif axidx == 1:
                            ax.plot(rawvtr.get_xdata(), rawvtr.ydata, color=color, label=comp)
                            ax.set_ylabel('Vel\nRaw')
                            ax.axhline(y=0, color='gray', zorder=-10)

                        elif axidx == 0:
                            ax.plot(staDict.components[comp].traces['acc'].get_xdata(), staDict.components[comp].traces['acc'].ydata, color=color, label=comp)
                            ax.set_ylabel('Acc\nRaw')
                            ax.axhline(y=0, color='gray', zorder=-10)

                        elif axidx == -3:
                            ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)
                            ax.axhline(y=0, color='gray', zorder=-10)
                            ax.set_ylabel('Mod\nDisp')

                        elif axidx == -2:
                            nvtr = own_differentation(tr, 1)
                            ax.set_ylabel('Mod\nVel')
                            ax.plot(nvtr.get_xdata(), nvtr.ydata, color=color, label=comp)

                        elif axidx == -1:
                            ax.axhline(y=0, color='gray', zorder=-10)
                            nvtr = own_differentation(tr, 2)
                            ax.set_ylabel('Mod\nAcc')
                            ax.plot(nvtr.get_xdata(), nvtr.ydata, color=color, label=comp)

                staDict.components[comp].traces['disp'] = tr
                staDict.components[comp].traces['vel'] = own_differentation(tr, 1)
                # except TypeError as e:
                #     print(e)
                #     print('Deleting station')
                #     delstas.append(sta)
                #     break

            if plot:
                axes[1].set_title('%s\ndist=%.1f' % (sta, dist))
                plt.tight_layout()
                plt.subplots_adjust(hspace=0)
                axes[0].legend(fontsize=10)
                print('%s%s_%s.png' % (plotdir, sta, mode))
                f.savefig('%s%s_%s.png' % (plotdir, sta, mode))

                plt.close('all')

        delstas = list(set(delstas))
        for sta in delstas:
            del self.stations[sta]

    def convert_to_ZNE(self, E=None, N=None, Z=None):
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                if comp == E and E and 'E' not in self.stations[sta].components:
                    self.stations[sta].components['E'] = self.stations[sta].components[E]
                    del self.stations[sta].components[E]

                if comp == N and N and 'N' not in self.stations[sta].components:
                    self.stations[sta].components['N'] = self.stations[sta].components[N]
                    del self.stations[sta].components[N]

                if comp == Z and Z and 'Z' not in self.stations[sta].components:
                    self.stations[sta].components['Z'] = self.stations[sta].components[Z]
                    del self.stations[sta].components[Z]

            for comp in ['0', '1', '2']:
                if comp in self.stations[sta].components:
                    del self.stations[sta].components[comp]

    def create_all_waveforms(self, disp=True, vel=True, acc=True):

        ot = self.refSource.time

        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                trA = self.stations[sta].components[comp].trace.copy()

                idxot = int(max((ot - trA.tmin), 0) / trA.deltat)
                if idxot > 0:
                    trA.ydata -= num.mean(trA.ydata[:idxot])
                    # print(idxot, num.mean(trA.ydata[:idxot]))

                del self.stations[sta].components[comp].trace

                if disp:
                    trD = own_integrate(trA, detrend=True, intval=2)
                    self.stations[sta].components[comp].traces['disp'] = trD

                if vel:
                    trV = own_integrate(trA, detrend=True, intval=1)
                    self.stations[sta].components[comp].traces['vel'] = trV

                if acc:
                    self.stations[sta].components[comp].traces['acc'] = trA

    def save_waveform(self, savepath, mode='disp'):
        trsave = []
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                if self.stations[sta].components[comp].traces:
                    for key, reftr in self.stations[sta].components[comp].traces.items():
                        if key == mode:
                            tr = reftr.copy()
                            trsave.append(tr)
                else:
                    tr = self.stations[sta].components[comp].trace.copy()
                    trsave.append(tr)

        io.save(trsave, '%s.mseed' % (savepath))

    def cut_waveforms(self):
        vP = 6.
        vS = vP / num.sqrt(3.)
        ot = self.refSource.time
        
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                for ttype in ['acc', 'vel', 'disp']:
                    tr = self.stations[sta].components[comp].traces[ttype]

                    ### rough p-pick
                    # tpick = tr.get_xdata()[num.where(abs(tr.ydata) > 0. * abs(max(tr.ydata)))][0]
                    # tb, tf = eqsig.im.calc_sig_dur_vals(tr.ydata, tr.deltat,
                    #                   start=0.1, end=0.9, se=True)
                    # tf = tf + tr.tmin
                    # tb = tb + tr.tmin
                    # print(tb, tf)
                    # te = tb + 4 * (tf - tb)
                    # tr.chop(tr.tmin, te)

                    rhypo = geodetic.distance(self.refSource.lon, self.refSource.lat,
                                        self.refSource.depth,
                                        self.stations[sta].lon, self.stations[sta].lat,
                                        0.)

                    parr = (rhypo / vP) + ot

                    addtime = 2.5 * (rhypo / (0.8 * vS))
                    if addtime < 150.:
                        addtime = 150.
                    tend = addtime + ot
                    tr.chop(tr.tmin, tend)

                    self.stations[sta].components[comp].traces[ttype] = tr

    def resample_waveform_disp(self, resample_f=200, resample_fac=1.):
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                tr = self.stations[sta].components[comp].trace.copy()

                # tr.ydata -= num.mean(tr.ydata)
                if resample_f:
                    if (1 / resample_f) != tr.deltat:
                        tr.resample(1 / resample_f)
                elif resample_fac != 1:
                    tr.resample(tr.deltat * resample_fac)

                ## cut 1.5s before and after, due to artifacts
                tr.chop(tr.tmin + 1.5, tr.tmax - 1.5, include_last=True)  # cleaner way
                self.stations[sta].components[comp].trace = tr

    def filter_waveform(self, freqs):
        if freqs is not None:
            for sta in self.stations:
                for comp in self.stations[sta].components.keys():
                    for key, reftr in self.stations[sta].components[comp].traces.items():

                        tr = reftr.copy()
                        # print('Applied Filter:', lpcnrfreq)
                        if type(freqs) == list:
                            if freqs[0] not in [None, 'None']:
                                tr.highpass(4, float(freqs[0]), demean=False)
                                # print('highpass filtered %s' % freqs[0])
                            if freqs[1] not in [None, 'None']:
                                tr.lowpass(4, float(freqs[1]), demean=False)
                                # print('lowpass filtered %s' % freqs[1])
                        elif type(freqs) in [float, int]:
                            tr.lowpass(4, float(freqs), demean=False)
                        tr.chop(tr.tmin + 3., tr.tmax, include_last=True)
                        # tr.ydata -= tr.ydata[0]

                        self.stations[sta].components[comp].traces[key] = tr

    def modify_waveform(self, quality_check=False, baseline_correction=False):
        vP = 6.
        vS = vP / num.sqrt(3.)
        ot = self.refSource.time
        rmSta = []
        for sta in self.stations:
            if quality_check:
                rhypo = geodetic.distance(self.refSource.lon, self.refSource.lat,
                                        self.refSource.depth,
                                        self.stations[sta].lon, self.stations[sta].lat,
                                        0.)

                parr = (rhypo / vP) + ot
            
            for comp in self.stations[sta].components.keys():

                tr = self.stations[sta].components[comp].trace.copy()

                if baseline_correction:
                    tr.highpass(4, 0.01)  # easiest way to do it
                    # tr.lowpass(4, 20)

                if quality_check:
                    '''
                    More to add?
                    '''
                    ### rough p-pick
                    # tpick = tr.get_xdata()[num.where(abs(tr.ydata) > 0. * abs(max(tr.ydata)))][0]
                    # tb, tf = eqsig.im.calc_sig_dur_vals(tr.ydata, tr.deltat,
                    #                   start=0.1, end=0.9, se=True)
                    # tf = tf + tr.tmin
                    # tb = tb + tr.tmin
                    # print(tb, tf)
                    # te = tb + 4 * (tf - tb)
                    # tr.chop(tr.tmin, te)

                    # print(sta, tr.tmin-parr, parr-parr, tr.tmax-parr)
                    if (tr.tmin - parr) > -3 or (tr.tmax - parr) < 3:
                        # print('Removing %s, due to missing P' % (sta))
                        rmSta.append(sta)
                        break

                    # addtime = 5 * (rHypo / (0.8 * vS))
                    # if addtime < 50.:
                    #     addtime = 50.
                    # tend = addtime + ot
                    # tend = tr.tmax
                    # tr.chop(tr.tmin, tend)
                    # print(rHypo, tend - ot)

                # remove global mean
                # tr.ydata -= num.mean(tr.ydata)
                # tr.ydata = signal.detrend(tr.ydata, type='linear')
                tr.ydata -= tr.ydata[0]

                self.stations[sta].components[comp].trace = tr

        rmSta = list(set(rmSta))
        if len(rmSta) > 0:
            print('Due to missing P, removed: %s' % (rmSta))
        for sta in rmSta:
            del self.stations[sta]

    def remove_tilted_waveforms(self):
        # from scipy.stats import linregress

        rmSta = []
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():

                tr = self.stations[sta].components[comp].traces['disp']

                ### remove tilted traces, mainly near-field

                # xs = [tr.get_xdata()[0], tr.get_xdata()[-1]]
                # ys = [tr.ydata[0], tr.ydata[-1]]
                # res = linregress(xs, ys)
                # tr.ydata -= res.intercept + (res.slope * tr.get_xdata())

                maxdata = max(abs(tr.ydata)) 

                if maxdata == max(abs(tr.ydata[-100:])) or maxdata == max(abs(tr.ydata[:100])):

                # if num.sqrt(num.mean(num.square(tr.ydata / maxdata))) > 0.5:
                    rmSta.append(sta)
                    print('Plot removed Stations')
                    plt.figure(figsize=(16, 8))
                    plt.plot(tr.get_xdata(), tr.ydata)
                    plt.savefig('/home/lehmann/dr/plots/%s_wvdata.png' %(sta))
                    plt.close('all')

                self.stations[sta].components[comp].trace = tr

        rmSta = list(set(rmSta))
        print('Stations are tilted', rmSta)
        for sta in rmSta:
            del self.stations[sta]

    def calc_peak_ratios(self, delete=False):
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                # self.stations[sta].components[comp].traces = {}
                gms = self.stations[sta].components[comp].gms
                imts = []
                for im in ['pga', 'pgv', 'pgd']:
                    if im in gms.keys():
                        imts.append(im)

                for im1 in imts:
                    for im2 in imts:
                        if im1 == im2:
                            continue

                        inv_name = '%s:%s' % (im2, im1)
                        if inv_name in self.stations[sta].components[comp].gms:
                            continue

                        name = '%s:%s' % (im1, im2)
                        val = gms[im1].value - gms[im2].value

                        GM = GMClass(name=name, value=val, unit='Log(Ratio)')
                        self.stations[sta].components[comp].gms[name] = GM

                if delete:
                    for im in imts:
                        del self.stations[sta].components[comp].gms[im]

    def calc_freq_ratios(self, delete=False):
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                # self.stations[sta].components[comp].traces = {}
                gms = self.stations[sta].components[comp].gms
                imts = []
                for im in gms:
                    if 'f_' in im:
                        imts.append(im)

                for im1 in imts:
                    for im2 in imts:
                        if im1 == im2:
                            continue

                        inv_name = '%s:%s' % (im2, im1)
                        if inv_name in self.stations[sta].components[comp].gms:
                            continue

                        name = '%s:%s' % (im1, im2)
                        val = gms[im1].value - gms[im2].value

                        GM = GMClass(name=name, value=val, unit='Log(Ratio)')
                        self.stations[sta].components[comp].gms[name] = GM

                if delete:
                    for im in imts:
                        del self.stations[sta].components[comp].gms[im]

    pass


###
# ...
###
def create_synthetic_waveform(gfPath, source, coords, stf=None, timecut=False,
                            chaCodes=['Z', 'N', 'E']):
    '''
    To do:
    - displacement to acceleration?
        Default is not so accurate as pyrock... (see below)
    '''

    engine = gf.LocalEngine(store_superdirs=[gfPath], store_dirs=[gfPath])

    ### Rectangular Fault description
    if source.form == 'rectangular':

        anchor = 'top_left'
        anchorP = source.surface.top_left

        if source.rupture_velocity is not None and source.rupture_velocity not in [-99.0, 0.0, 999.0]:
            vr = source.rupture_velocity * 1000.
            print('Rupture velocity:', vr)
        else:
            vr = 0.8 * 3460

        pyrockoSource = gf.RectangularSource(
                    time=source.time,
                    lat=anchorP.latitude, lon=anchorP.longitude,
                    depth=anchorP.depth * 1000.,
                    anchor=anchor,
                    dip=source.dip, strike=source.strike, rake=source.rake,
                    width=source.width * 1000., length=source.length * 1000.,
                    nucleation_x=source.nucleation_x,
                    nucleation_y=source.nucleation_y,
                    velocity=vr,  # vr = vs * 0.8
                    # aggressive_oversampling=True,
                    # decimation_factor=20,  # auf 1 oder max 2 lassen
                    # slip=10., # abfrage, am besten glatten/constanten slip
                    magnitude=source.magnitude)
        print(pyrockoSource)

    ### PseudoDynamic Fault description
    elif source.form == 'pdr':

        anchor = 'top_left'
        anchorP = source.surface.top_left
        patchsize = 2.  # in km
        pyrockoSource = gf.PseudoDynamicRupture(
                        time=source.time,
                        lat=anchorP.latitude, lon=anchorP.longitude,
                        depth=anchorP.depth * 1000.,
                        anchor=anchor,
                        dip=source.dip, strike=source.strike, rake=source.rake,
                        width=source.width * 1000., length=source.length * 1000.,
                        nucleation_x=source.nucleation_x,
                        nucleation_y=source.nucleation_y,
                        magnitude=source.magnitude,

                        # Number of discrete patches
                        nx=int(source.length / patchsize),
                        ny=int(source.width / patchsize),
                        # Relation between subsurface model s-wave velocity vs
                        # and rupture velocity vr
                        gamma=0.8,
                        nthreads=4)

    ### Point source description
    elif source.form == 'point':

        if source.tensor:
            print('Has full MT available')
            tn = source['tensor']
            pyrockoSource = gf.MTSource(lat=source.lat, lon=source.lon,
                                depth=source.depth * 1000.,
                                time=source.time,
                                mnn=tn['mnn'], mee=tn['mee'], mdd=tn['mdd'],
                                mne=tn['mne'], mnd=tn['mnd'], med=tn['med'])

        elif source.strike is not None and source.dip is not None \
                and source.rake is not None:
            print('DC available', source.strike, source.dip, source.rake)

            pyrockoSource = gf.DCSource(lat=source.lat, lon=source.lon,
                                depth=source.depth * 1000.,
                                magnitude=source.magnitude,
                                time=source.time,
                                dip=source.dip, strike=source.strike,
                                rake=source.rake)

        else:
            pyrockoSource = gf.ExplosionSource(lat=source.lat, lon=source.lon,
                                        depth=source.depth * 1000.,
                                        time=source.time,
                                        magnitude=source.magnitude)

    if stf is None:
        pass
    else:
        if source.form != 'pdr':
            pyrockoSource.update(stf=stf)

    if timecut:
        tmin = source.time - 40
        tmax = source.time + 190
        print('TimeCut:', tmax - tmin, tmin, tmax)

        waveform_targets = [gf.Target(quantity='displacement',
                                lat=coords[ii][1], lon=coords[ii][0],
                                store_id=str(gfPath.rsplit('/')[-2]),
                                tmin=tmin,
                                tmax=tmax,
                                codes=('PR', 'S' + str(ii),
                                    '00', channel_code))
                        for channel_code in chaCodes
                        for ii in range(len(coords))]
    else:
        waveform_targets = [gf.Target(quantity='displacement',
                                lat=coords[ii][1], lon=coords[ii][0],
                                store_id=str(gfPath.rsplit('/')[-2]),
                                codes=('PR', 'S' + str(ii),
                                    '00', channel_code))
                        for channel_code in chaCodes
                        for ii in range(len(coords))]

    t1 = time.time()
    response = engine.process(pyrockoSource, waveform_targets)
    synthetic_traces = response.pyrocko_traces()
    print('Target process time', time.time() - t1)

    if timecut:
        for tr in synthetic_traces:
            tr.chop(tmin=tmin + 10, tmax=tmax - 10)

    # print('Finished Pyrocko-WV')

    # newSynthetic_traces = []
    # for tr in synthetic_traces:
    #   newTr = tr.transfer(3,
    #               transfer_function=trace.DifferentiationResponse(2))
    #   newSynthetic_traces.append(newTr)

    # return newSynthetic_traces, waveform_targets

    return synthetic_traces, waveform_targets

###
# ---
###
def create_stationdict_with_traces(traces, locDict):
    staDict = {}
    # print(locDict)
    # print(list(locDict.keys())[0].rsplit('.'))
    if len(list(locDict.keys())[0].rsplit('.')[-1]) > 5:
        locParam = True
    elif len(list(locDict.keys())[0].rsplit('.')) > 2:
        locParam = True
    else:
        locParam = False
    # print(locParam)
    for tr in traces:
        if not locParam:
            ns = '%s.%s' % (tr.network, tr.station)
        else:
            ns = '%s.%s%s' % (tr.network, tr.station, tr.location)

        try:
            locDict[ns]
        except KeyError as e:
            print('Missing location for %s' % e)
            continue

        if ns not in staDict:
            STA = StationGMClass(
                network=ns.rsplit('.')[0],
                station=ns.rsplit('.')[1],
                lat=locDict[ns][1],
                lon=locDict[ns][0],
                components={})
        else:
            STA = staDict['%s.%s' % (ns.rsplit('.')[0], ns.rsplit('.')[1])]

        cha = tr.channel
        COMP = ComponentGMClass(
                component=cha[-1])

        COMP.trace = tr
        STA.components[COMP.component] = COMP
        staDict['%s.%s' % (ns.rsplit('.')[0], ns.rsplit('.')[1])] = STA

    return staDict


def get_observation_container(source, wvData, locationDict, mapextent,
                            pyrockoChas, filterfreq, imts, freqs, H2=True, 
                            deleteWvData=True, savepath=None, resample_f=200,
                            tilt_correction='disp', rmStas=[]):

    staDict = create_stationdict_with_traces(wvData, locationDict)
    stationCont = StationContainerObservation(refSource=source, stations=staDict)
    stationCont.validate()

    #############################
    ### Data Check
    #############################
    print('Data for %s observed stations' % (len(stationCont.stations)))
    ### remove Data outside of the area of interest or limits
    stationCont.remove_distant_stations(mapextent[0], mapextent[1])

    print('Remaining data for %s observed stations after distance check.'
        % (len(stationCont.stations)))

    stationCont.remove_short_traces(mint=20.)
    stationCont.remove_notneeded_components(pyrockoChas)
    stationCont.remove_stations_without_components(pyrockoChas)

    stationCont.remove_stations(rmStas)
    print('Remaining data for %s observed stations after manual selection/removal.'
        % (len(stationCont.stations)))

    print('Remaining data for %s observed stations.'
        % (len(stationCont.stations)))

    ##### Waveform modification
    stationCont.modify_waveform(quality_check=True)
    stationCont.create_all_waveforms()
    stationCont.cut_waveforms()
    if savepath:
        stationCont.save_waveform(savepath=savepath + '_disp', mode='disp')
        stationCont.save_waveform(savepath=savepath + '_vel', mode='vel')
        stationCont.save_waveform(savepath=savepath + '_acc', mode='acc')

    # if 'pgd' in imts or 'pgv' in imts:
        # stationCont.remove_tilted_waveforms()
    # 
    
    print('Remaining data after waveform quality check: %s stations.'
        % (len(stationCont.stations)))

    stationCont.convert_to_ZNE(E='2', N='1', Z='0')

    if tilt_correction:
        # stationCont.modify_disp_trace(mode=tilt_correction, plot=True)
        stationCont.modify_disp_trace(mode=tilt_correction, plot=False)
    stationCont.resample_waveform(resample_f=resample_f)
    stationCont.filter_waveform(freqs=filterfreq)
    stationCont.get_gm_from_wv(imts=imts, freqs=freqs,
            H2=H2, delete=False, deleteWvData=deleteWvData)

    return stationCont


def pf(x, consts):
    consts = num.flip(consts)
    y = 0
    for ii in range(len(consts)):
        y += consts[ii] * (x**(ii))

    return y


def own_integrate(intr, intval=1, detrend=False, starttime=None, endtime=None):
    tr = intr.copy()
    # tr.ydata = tr.ydata[500:-500]
    # tr.ydata -= num.mean(tr.ydata)

    # if detrend:
    #   # tr.ydata = signal.detrend(tr.ydata, type='linear')
    #   tr.ydata = obspy.signal.detrend.simple(tr.ydata)

    # tr.ydata = num.pad(tr.ydata, (500, 500), 'linear_ramp', end_values=(0, 0))

    vtr = tr.transfer(0,
                    # cut_off_fading=False,
                    transfer_function=trace.IntegrationResponse(intval))

    if detrend:
        if starttime:
            st = max((starttime - vtr.tmin), 0)
        else:
            st = 1
        upintv = int(st / vtr.deltat)

        if endtime:
            et = max((endtime - vtr.tmin), 0)
        else:
            et = 6
        downintv = int(et / vtr.deltat)

        facs = num.polyfit(vtr.get_xdata()[upintv:downintv] - vtr.tmin,
                        vtr.ydata[upintv:downintv], 1)
        vtr.ydata = vtr.ydata - pf(vtr.get_xdata() - vtr.tmin, facs)

        # vtr.ydata -= num.mean(vtr.ydata[upintv:downintv])

    # print(facs)
    # print(vtr.ydata)
    # vtr.ydata -= num.mean(vtr.ydata[10:100])
    # if detrend:
    
    # vtr.ydata = signal.detrend(vtr.ydata, type='linear')
    # print(vtr.ydata)
    # exit()
    vtr.ydata -= vtr.ydata[0]

    return vtr


def corners2surface(source, corners):

    pUR = Point(corners['UR'][0], corners['UR'][1],
                corners['UR'][2])
    pUL = Point(corners['UL'][0], corners['UL'][1],
                corners['UL'][2])
    pLL = Point(corners['LL'][0], corners['LL'][1],
                corners['LL'][2])
    pLR = Point(corners['LR'][0], corners['LR'][1],
                corners['LR'][2])

    try:
        surface = PlanarSurface.from_corner_points(top_left=pUL, top_right=pUR,
                                            bottom_left=pLL, bottom_right=pLR)
    except ValueError as e:
        print(e)
        return None

    return surface


def convert_surface(source, surface):
    source.surface = surface
    source.form = 'rectangular'
    surfstrike = surface.get_strike()
    surfdip = surface.get_dip()

    if not source.rake:
        source.rake = 0.0
        print('Set rake manually to 0')
    else:
        print('Point source params: Strike %0.2f; Dip %0.2f; Rake %0.2f'
                % (source.strike, source.dip, source.rake))

    print('Fault source params: Strike %0.2f; Dip %0.2f; Rake %0.2f'
                % (surfstrike, surfdip, source.rake))

    source.strike = surfstrike
    source.dip = surfdip
    source.ztor = float(surface.get_top_edge_depth())
    source.area = float(surface.get_area())
    source.width = float(surface.get_width())
    source.length = source.area / source.width

    ### Convert nucleation coordinates
    if source.nucleationCoords:
        nucLat = source.nucleationCoords[1]
        nucLon = source.nucleationCoords[0]
        nucDepth = source.nucleationCoords[2]
    else:
        nucLat = source.lat
        nucLon = source.lon
        nucDepth = source.depth

    upperDepth = source.ztor
    lowerDepth = surface.bottom_right.depth

    midDepth = (upperDepth + lowerDepth) / 2.
    source.nucleation_y = float((nucDepth - midDepth)
                            / (lowerDepth - midDepth))

    d1 = orthodrome.distance_accurate50m_numpy(nucLat, nucLon,
                                            surface.top_left.latitude,
                                            surface.top_left.longitude)
    d2 = orthodrome.distance_accurate50m_numpy(nucLat, nucLon,
                                            surface.top_right.latitude,
                                            surface.top_right.longitude)
    d3 = orthodrome.distance_accurate50m_numpy(surface.top_left.latitude,
                                            surface.top_left.longitude,
                                            surface.top_right.latitude,
                                            surface.top_right.longitude)
    nucX = ((d1**2 - d2**2 - d3**2) / (-2 * d3**2))[0]
    source.nucleation_x = float(-((nucX * 2) - 1))

    # print('1', source.nucleation_x, source.nucleation_y)

    ## Alternative
    # tL = source.surface.top_left
    # tR = source.surface.top_right
    # bL = source.surface.bottom_left
    # bR = source.surface.bottom_right

    # if source.depth <= tL.depth:
    #   source.nucleation_y = -1
    # elif source.depth >= bR.depth:
    #   source.nucleation_y = 1
    # else:
    #   nuclY = (source.depth - tL.depth) / (bR.depth - tL.depth)
    #   source.nucleation_y = (nuclY * 2) - 1

    # distX = geodetic.min_distance_to_segment(
    #                   num.array([bL.longitude, tL.longitude]),
    #                   num.array([bL.latitude, tL.latitude]),
    #                   num.array([source.lon, bR.longitude]),
    #                   num.array([source.lat, bR.latitude]))
    # if distX[0] < 0.:
    #   source.nucleation_x = -1
    # elif distX[0] > distX[1]:
    #   source.nucleation_x = 1
    # else:
    #   nuclX = distX[0] / distX[1]
    #   source.nucleation_x = (nuclX * 2) - 1

    # print('2', source.nucleation_x, source.nucleation_y)

    return source


###
#
###
def get_pyrocko_container(source, coords, pyrockoChas, imts, freqs, filterfreq=None,
                        H2=True, delete=False, deleteWvData=False, resample_f=None,
                        gfpath='/home/lehmann/dr/pyrocko_gf/own_2hz_distant/',
                        savepath=None, timecut=False, only_waveform=False):

    stf = get_stf_with_duration(source)

    t1 = time.time()
    synthTraces, targets = create_synthetic_waveform(
                            gfpath, source, coords, timecut=timecut,
                            stf=stf, chaCodes=pyrockoChas)
    # source.validate()
    print('Py-Time', time.time() - t1)

    synthStaDict = create_stationdict_synthetic(synthTraces, targets)
    pyrockoCont = StationContainerObservation(refSource=source, stations=synthStaDict)

    if savepath:
        pyrockoCont.save_waveform(savepath=savepath)

    if not only_waveform:
        t1 = time.time()
        pyrockoCont.create_all_waveforms_synth()
        print('Wv-Time', time.time() - t1)

        ### Resampling afterwards!!!! Otherwise artifacts come in
        if resample_f is not None:
            t1 = time.time()
            pyrockoCont.resample_waveform(resample_f=resample_f)  # alternative resampling, has the same effect
            print('RS-Time', time.time() - t1)

        pyrockoCont.filter_waveform(freqs=filterfreq)

        t1 = time.time()
        pyrockoCont.get_gm_from_wv(imts=imts, freqs=freqs,
                                H2=H2, delete=delete,
                                deleteWvData=deleteWvData)
        print('GM-Time', time.time() - t1)

    return pyrockoCont


###
# Scaling relations
###
def calc_source_width_length(magnitude, mode='Blaser', typ='scr', rake=0.):

    if mode == 'WC':
        ### wells and copersmith 1994
        # RA = 10**((magnitude - 4.07) / 0.98)
        WD = 10**((magnitude - 4.06) / 2.25)
        # LN = RA / WD
        LN = 10**((magnitude - 4.38) / 1.49)

        ## from blaser 2010
        # LN = 10**(magnitude * 0.59 - 2.44)
        # WD = 10**(magnitude * 0.32 - 1.01)

    elif mode == 'Blaser':
        #### Blaser et al., 2010

        if rake < 135 and rake > 45:
            # print('reverse')
            # wa = -1.86 + num.random.normal(0, 0.12)
            # wb = 0.46 + num.random.normal(0, 0.02)
            ws = 0.17
            wcov = num.matrix([[27.47, -3.77], [-3.77, 0.52]]) * 1e-5
            wa, wb = num.random.multivariate_normal([-1.86, 0.46], wcov)

            # la = -2.37 + num.random.normal(0, 0.13)
            # lb = 0.57 + num.random.normal(0, 0.02)
            ls = 0.18
            lcov = num.matrix([[26.14, -3.67], [-3.67, 0.52]]) * 1e-5
            la, lb = num.random.multivariate_normal([-2.37, 0.57], lcov)

        elif rake > -135 and rake < -45:
            # print('normal')
            # wa = -1.20 + num.random.normal(0, 0.25)
            # wb = 0.36 + num.random.normal(0, 0.04)
            ws = 0.16
            wcov = num.matrix([[264.18, -42.02], [-42.02, 6.73]]) * 1e-5
            wa, wb = num.random.multivariate_normal([-1.20, 0.36], wcov)

            # la = -1.91 + num.random.normal(0, 0.29)
            # lb = 0.52 + num.random.normal(0, 0.04)
            ls = 0.18
            lcov = num.matrix([[222.24, -32.34], [-32.34, 4.75]]) * 1e-5
            la, lb = num.random.multivariate_normal([-1.91, 0.52], lcov)

        else:
            # print('strike-slip')
            # wa = -1.12 + num.random.normal(0, 0.12)
            # wb = 0.33 + num.random.normal(0, 0.02)
            ws = 0.15
            wcov = num.matrix([[13.48, -2.18], [-2.18, 0.36]]) * 1e-5
            wa, wb = num.random.multivariate_normal([-1.12, 0.33], wcov)

            # la = -2.69 + num.random.normal(0, 0.11)
            # lb = 0.64 + num.random.normal(0, 0.02)
            ls = 0.18
            lcov = num.matrix([[12.37, -1.94], [-1.94, 0.31]]) * 1e-5
            la, lb = num.random.multivariate_normal([-2.69, 0.64], lcov)

        # LN = 10**(num.random.normal(magnitude * lb + la, ls**2))
        # WD = 10**(num.random.normal(magnitude * wb + wa, ws**2))

        LN = 10**(num.random.normal(magnitude * lb + la, ls**2))
        WD = 10**(num.random.normal(magnitude * wb + wa, ws**2))

        # print(LN)
        # print(WD)
        # print()

        # LN = LN / 1000
        # WD = WD / 1000

    elif mode == 'Leonard':
        ### Leonard 2010/2014
        m0 = pmt.magnitude_to_moment(magnitude)

        mu = 3.3 * 1e10

        # typ = 'interplate'
        # typ = 'scr'


        # get_truncated_normal(mean=0.64, sd=0.02, lim='std').rvs(1)[0]

        # X1 = get_truncated_normal(mean=2, sd=1, low=1, upp=10)
        # X2 = get_truncated_normal(mean=5.5, sd=1, low=1, upp=10)
        # X3 = get_truncated_normal(mean=8, sd=1, low=1, upp=10)

        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots(3, sharex=True)
        # ax[0].hist(X1.rvs(10000), normed=True)
        # ax[1].hist(X2.rvs(10000), normed=True)
        # ax[2].hist(X3.rvs(10000), normed=True)


        if typ == 'interplate':
            if rake < 135 and rake > 45 or rake > -135 and rake < -45:
                ## dip-slip
                # c1 = 17.5
                c1 = num.random.uniform(12, 25)
                # c2 = 3.8
                c2 = num.random.uniform(1.5, 12)

            else:
                ## strike-slip
                # c1 = 15.
                c1 = num.random.uniform(11, 20)
                # c2 = 3.7
                c2 = num.random.uniform(1.5, 9.0)
        elif typ in ['scr', 'stabel continental regions']:
            if rake < 135 and rake > 45 or rake > -135 and rake < -45:
                ## dip-slip
                # c1 = 13.5
                c1 = num.random.uniform(10, 17)
                # c2 = 7.3
                c2 = num.random.uniform(5.0, 10.0)
            else:
                ## strike-slip
                # c1 = 11.3
                c1 = num.random.uniform(9, 15)
                # c2 = 5.8
                c2 = num.random.uniform(3.0, 11)

        c2 = c2 * 1e-5

        LN = 10**((num.log10(m0) - (1.5 * num.log10(c1)) - num.log10(c2 * mu)) / 2.5)
        WD = 10**((num.log10(m0) + (2.25 * num.log10(c1)) - num.log10(c2 * mu)) / 3.75)

        LN = LN / 1000
        WD = WD / 1000

    # elif mode == 'Leonard2':
    #   ### still in work, no uncertainties and no correction for to large events
    #   ### Leonard 2010/2014
        # from pyrocko import moment_tensor as pmt
        # m0 = pmt.magnitude_to_moment(magnitude)

        # if typ == 'interplate':
        #   if rake < 135 and rake > 45 or rake > -135 and rake < -45:
        #       la = 3.0
        #       lb = 6.098

        #       wa = 3.75
        #       wb = 3.301
        #   else:
        #       ## strike-slip
        #       la = 3.0
        #       lb = 6.087

        #       wa = 3.75
        #       wb = 3.441

        # elif typ in ['scr', 'stabel continental regions']:
        #   if rake < 135 and rake > 45 or rake > -135 and rake < -45:
        #       ## dip-slip
        #       la = 3.0
        #       lb = 6.382

        #       wa = 3.75
        #       wb = 3.84

        #   else:
        #       ## strike-slip
        #       la = 3.0
        #       lb = 6.370

        #       wa = 3.75
        #       wb = 3.966

        # LN = 10**((num.log10(m0) - lb) / la)
        # WD = 10**((num.log10(m0) - wb) / wa)

        # LN = LN / 1000
        # WD = WD / 1000

    else:
        print('Wrong Scaling-Relation  Mode')
        exit()

    # print(mode)
    # print('Area:', LN * WD)
    # print('WD  :', WD, 'log', num.log10(WD))
    # print('LN  :', LN, 'log', num.log10(LN))
    # print()

    ## what about uncertainties?

    return WD, LN


def calc_rupture_duration(source=None, mag=None, moment=None,
                        vr=None, WD=None, LN=None, nucx=None, nucy=None,
                        mode='uncertain'):

    if source:
        if source.rupture_velocity is not None and source.rupture_velocity not in [-99.0, 0.0, 999.0]:
            vr = source.rupture_velocity
        else:
            vs = 5.9 / num.sqrt(3)  # extract from velocity model for crust?
            vr = 0.8 * (vs)
        print('Rupture velocity:', vr)

        if source.length and source.width:
            WD = source.width
            LN = source.length
        else:
            WD, LN = calc_source_width_length(source.magnitude, rake=source.rake)

        nucx = source.nucleation_x
        nucy = source.nucleation_y
    elif vr is not None and WD is not None and LN is not None:
        if nucx is None:
            nucx = 0
        if nucy is None:
            nucy = 0

    if mode == 'own':
        eLN = LN * (0.5 + 0.5 * abs(nucx))
        eWD = WD * (0.5 + 0.5 * abs(nucy))
        diag = num.sqrt((eLN)**2 + (eWD)**2)

        maxlen = float(max(eLN, eWD, diag))
        duration = (maxlen / vr)

    elif mode == 'uncertain':
        eLN = LN * 0.5
        eWD = WD * 0.5
        diag = num.sqrt((eLN)**2 + (eWD)**2)

        maxlen = float(max(eLN, eWD, diag))
        dur = (maxlen / vr)  # Duration from middle
        duration = float(num.random.uniform(dur, 2 * dur))  # Uncertainty

    elif mode == 'pub':
        if mag is not None or moment is not None:
            if mag is not None:
                moment = pmt.magnitude_to_moment(mag)
            # duration = num.sqrt(moment) / 0.5e9
            duration = num.power(moment, 0.33) / 0.25e6
        else:
            print('Magnitude or Moment missing')
            exit()

    else:
        print('Wrong Rupture Duration mode %s' % (mode))
        exit()

    return duration


def calc_rise_time(source=None, mag=None, fac=0.8):
    ### Rise time after Chen Ji 2021; Two Empirical Double-Corner-Frequency Source
    # Spectra and Their Physical Implications

    if mag is None:
        if source is None:
            print('Need to select either source or magnitude')
            exit()
        mag = source.magnitude

    fc2 = 10**(3.250 - (0.5 * mag))
    riseTime = fac / fc2

    # sommerville 1999
    # riseTime2 = 2.03e-9 * (mag ** (1 / 3)) #smth wrong?
    # the same!
    # riseTime3 = 10**(0.5 * mag - 3.34)

    # gusev 2018
    # riseTime4 = 0.5 / fc2

    # wang and day 2017
    # riseTime5 = 1.5 / fc2  # S
    # riseTime5 = 1.8 / fc2  # P

    if riseTime < 1.0:
        riseTime = 1.0

    return riseTime


def get_stf_with_duration(source):

    if hasattr(source, 'stf') and source.form == 'point' and source.stf:
        sstf = source.stf
        try:
            duration = float(sstf['duration'])
        except TypeError:
            duration = float(sstf.duration)
        source.duration = duration
        source.duration_type = 'observed'
        if hasattr(sstf, 'type') and sstf.type == 'triangle':
            stf = gf.TriangularSTF(duration=duration)
        elif hasattr(sstf, 'type') and sstf.type == 'boxcar':
            stf = gf.BoxcarSTF(duration=duration)
        else:
            stf = gf.HalfSinusoidSTF(anchor=0, duration=duration)

    else:
        if source.form == 'point':
            risetime = calc_rupture_duration(source, mode='uncertain')
            # print('RuptureDuration:', risetime)
            source.duration_type = 'calculated'
        else:
            if source.risetime is not None and source.risetime not in [-99.0, 0.0, 999.0]:
                risetime = source.risetime
                source.duration_type = 'observed'
            else:
                risetime = calc_rise_time(source)
                source.duration_type = 'calculated'
            print('RiseTime:', risetime)

        source.duration = risetime

        # stf = None
        # stf = gf.BoxcarSTF(duration=riseTime)
        # stf = gf.TriangularSTF(duration=riseTime)
        stf = gf.HalfSinusoidSTF(anchor=0, duration=risetime)

    return stf


###
# Baseline
###
def trace_baseline_correction(tr, vtr, comp, idx1, idx2, idx3, axes, mode='tilt', plot=True):

    if comp == 'N':
        color = 'green'
        # continue
    elif comp == 'E':
        color = 'orange'
        # continue
    elif comp == 'Z':
        color = 'blue'

    dt = tr.deltat

    if mode == 'tilt':

        if plot:
            axes[-1].set_title('Tilted (?)')

        # facs = num.polyfit(tr.get_xdata() - tr.tmin, tr.ydata, 1)
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin, tr.ydata[:idx1], 1)
        tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)

        rawtr = tr.copy()
        flag = True
        rang = 3
        while flag:
            tr = rawtr.copy()
            if plot:
                ax = axes[3]
                ax.axvline(dt * idx3 + tr.tmin)
                ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

            facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin,
                            tr.ydata[idx3:], rang)
            tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
            if plot:
                ax = axes[4]
                ax.axvline(dt * idx2 + tr.tmin)
                ax.axvline(dt * idx1 + tr.tmin)
                ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

            facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
                            tr.ydata[:idx1], rang)

            tr.ydata[:idx2] = tr.ydata[:idx2] - pf(tr.get_xdata()[:idx2]
                - tr.tmin, facs) + pf(tr.get_xdata()[:idx2] - tr.tmin, facs)[-1]

            tr.ydata -= tr.ydata[0]
            if plot:
                ax = axes[5]
                ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

                ax.axvline(dt * idx2 + tr.tmin)

            flag = False

            # facs = num.polyfit(tr.get_xdata()[idx3:],
            #                 tr.ydata[idx3:], 1)

            # ampval = tr.ydata[idx3:] - pf(tr.get_xdata()[idx3:], facs)

            # err = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin,
            #                 tr.ydata[idx3:], 1, full=True)[1][0]

            # ### Define Rang
            # # cc = num.corrcoef(ampval, tr.ydata[idx3:])
            # # r = cc[0][1]
            # # b = facs[-2]
            # # var = num.var(tr.ydata[idx3:])
            # # # # print(r, b, var)
            # # # # print(b, cc2[0][1])
            # # f = abs(r) / (abs(b) * var)
            # # print(comp, rang, f, r, b, var)
            # print(comp, rang, err)
            # # print(tr.get_xdata()[idx3] - tr.tmin)
            # # if f < 1000. and rang < 4:
            # if err > 0.1 and rang < 3:
            #     rang += 2
            # else:
            #     flag = False
        # exit()

    else:
        if plot:
            axes[-1].set_title('None tilt')

        facs = num.polyfit(vtr.get_xdata() - vtr.tmin, vtr.ydata, 1) # or 2
        vtr.ydata = vtr.ydata - pf(vtr.get_xdata() - vtr.tmin, facs)
        if plot:
            ax = axes[3]
            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)

        tr = own_integrate(vtr, detrend=False, intval=1)
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
                        tr.ydata[:idx1], 1)
        tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
        if plot:
            ax = axes[4]
            ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)
        facs = num.polyfit(tr.get_xdata()[idx1:] - tr.tmin,
                        tr.ydata[idx1:], 1)
        facs = num.polyfit(tr.get_xdata() - tr.tmin, tr.ydata, 3)
        tr.ydata[idx1:] = tr.ydata[idx1:] - pf(tr.get_xdata()[idx1:] - tr.tmin, facs) + pf(tr.get_xdata()[idx1:] - tr.tmin, facs)[0]

        if plot:
            ax = axes[5]
            ax.axvline(dt * idx2 + tr.tmin)
            ax.axvline(dt * idx3 + tr.tmin)
            ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

        tr.ydata = tr.ydata - num.mean(tr.ydata[:idx1])

    return tr



def trace_baseline_correction2(tr, vtr, comp, idx1, idx2, idx3, axes, mode='tilt', plot=True):

    if comp == 'N':
        color = 'green'
        # continue
    elif comp == 'E':
        color = 'orange'
        # continue
    elif comp == 'Z':
        color = 'blue'

    dt = tr.deltat

    if mode == 'tilt':

        if plot:
            axes[-1].set_title('Tilted (?)')

        # facs = num.polyfit(tr.get_xdata() - tr.tmin, tr.ydata, 1)
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin, tr.ydata[:idx1], 1)
        tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)

        rawtr = tr.copy()
        flag = True
        rang = 3
        while flag:
            tr = rawtr.copy()
            if plot:
                ax = axes[3]
                ax.axvline(dt * idx3 + tr.tmin)
                ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

            facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin,
                            tr.ydata[idx3:], rang)
            tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
            if plot:
                ax = axes[4]
                ax.axvline(dt * idx2 + tr.tmin)
                ax.axvline(dt * idx1 + tr.tmin)
                ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

            facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
                            tr.ydata[:idx1], rang)

            tr.ydata[:idx2] = tr.ydata[:idx2] - pf(tr.get_xdata()[:idx2]
                - tr.tmin, facs) + pf(tr.get_xdata()[:idx2] - tr.tmin, facs)[-1]

            tr.ydata -= tr.ydata[0]
            if plot:
                ax = axes[5]
                ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

                ax.axvline(dt * idx2 + tr.tmin)

            flag = False

            # facs = num.polyfit(tr.get_xdata()[idx3:],
            #                 tr.ydata[idx3:], 1)

            # ampval = tr.ydata[idx3:] - pf(tr.get_xdata()[idx3:], facs)

            # err = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin,
            #                 tr.ydata[idx3:], 1, full=True)[1][0]

            # ### Define Rang
            # # cc = num.corrcoef(ampval, tr.ydata[idx3:])
            # # r = cc[0][1]
            # # b = facs[-2]
            # # var = num.var(tr.ydata[idx3:])
            # # # # print(r, b, var)
            # # # # print(b, cc2[0][1])
            # # f = abs(r) / (abs(b) * var)
            # # print(comp, rang, f, r, b, var)
            # print(comp, rang, err)
            # # print(tr.get_xdata()[idx3] - tr.tmin)
            # # if f < 1000. and rang < 4:
            # if err > 0.1 and rang < 3:
            #     rang += 2
            # else:
            #     flag = False
        # exit()

    else:
        if plot:
            axes[-1].set_title('None tilt')

        # facs = num.polyfit(vtr.get_xdata() - vtr.tmin, vtr.ydata, 1) # or 2
        # vtr.ydata = vtr.ydata - pf(vtr.get_xdata() - vtr.tmin, facs)
        # if plot:
        #     ax = axes[3]
        #     ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)

        # tr = own_integrate(vtr, detrend=False, intval=1)
        # facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
        #                 tr.ydata[:idx1], 1)
        # tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
        # if plot:
        #     ax = axes[4]
        #     ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

        # # facs = num.polyfit(tr.get_xdata()[idx1:] - tr.tmin,
        # #                 tr.ydata[idx1:], 1)
        # facs = num.polyfit(tr.get_xdata() - tr.tmin, tr.ydata, 3)
        # tr.ydata[idx1:] = tr.ydata[idx1:] - pf(tr.get_xdata()[idx1:] - tr.tmin, facs) + pf(tr.get_xdata()[idx1:] - tr.tmin, facs)[0]

        # if plot:
        #     ax = axes[5]
        #     ax.axvline(dt * idx2 + tr.tmin)
        #     ax.axvline(dt * idx3 + tr.tmin)
        #     ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

        # tr.ydata = tr.ydata - num.mean(tr.ydata[:idx1])

        vtr.highpass(3, 0.05, demean=False)

        if plot:
            ax = axes[3]
            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)

        tr = own_integrate(vtr, detrend=False, intval=1)
        # tr.highpass(3, 0.05, demean=False)

        if plot:
            ax = axes[4]
            ax.axvline(dt * idx2 + tr.tmin)
            ax.axvline(dt * idx3 + tr.tmin)
            ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

    return tr




def trace_baseline_correction_ebasco(atr, comp, idx1, idx2, idx3, axes, mode='tilt', plot=True):

    atr.ydata -= atr.ydata[0]

    if comp == 'N':
        color = 'green'
        # continue
    elif comp == 'E':
        color = 'orange'
        # continue
    elif comp == 'Z':
        color = 'blue'

    ## does smth
    dt = atr.deltat
    if plot:
        axes[3].plot(atr.get_xdata(), atr.ydata, color=color, label=comp)
        axes[3].axhline(y=0, color='gray', zorder=-10)

    # if mode == 'tilt':
    if True:
        vtr = own_integrate(atr, detrend=False, intval=1)
        facs = num.polyfit(vtr.get_xdata()[:idx1] - vtr.tmin, vtr.ydata[:idx1], 1)
        print(facs)
        Ai = facs[-2]
        
        facs = num.polyfit(vtr.get_xdata()[idx3:] - vtr.tmin, vtr.ydata[idx3:], 1)
        print(facs)
        Af = facs[-2]
        Vf = facs[-1]

        Am = (pf(vtr.get_xdata()[idx3] - vtr.tmin, facs)) / ((dt * idx3) - (dt * idx2))

        print(Am)

        atr.ydata[idx1:idx3] -= Am
        atr.ydata[:idx1] -= Ai
        atr.ydata[idx3:] -= Af

        vtr = own_integrate(atr, detrend=False, intval=1)
        # vtr.ydata -= vtr.ydata[0]
        # vtr.ydata -= num.mean(vtr.ydata[:idx1])
        if plot:
            ax = axes[4]
            ax.axvline(dt * idx3 + vtr.tmin)
            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)

        # facs = num.polyfit(vtr.get_xdata()[idx3:] - vtr.tmin,
        #                 vtr.ydata[idx3:], 3)
        # vtr.ydata[idx3:] = vtr.ydata[idx3:] - pf(vtr.get_xdata()[idx3:] - vtr.tmin,
        #                  facs) + pf(vtr.get_xdata()[idx3:] - vtr.tmin, facs)[0]

        tr = own_integrate(vtr, detrend=False, intval=1)
        tr.ydata -= tr.ydata[0]
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
                        tr.ydata[:idx1], 1)
        tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)

        facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin,
                        tr.ydata[idx3:], 1)
        Di = facs[-2]
        vtr.ydata[idx2:] -= Di
        if plot:
            ax = axes[5]
            ax.axvline(dt * idx3 + vtr.tmin)
            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)

        tr = own_integrate(vtr, detrend=False, intval=1)
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
                        tr.ydata[:idx1], 1)
        tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
        # tr.ydata -= tr.ydata[0]

    # else:
    if mode != 'tilt':

        # vtr = own_integrate(atr, detrend=False, intval=1)
        # tr = own_integrate(atr, detrend=False, intval=2)
        # facs = num.polyfit(tr.get_xdata()[idx1:idx2] - tr.tmin, tr.ydata[idx1:idx2], 1)
        # print(facs)
        # Vi = facs[-2]
        # vtr.ydata[idx1:idx2] -= Vi
        
        facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin, tr.ydata[idx3:], 1)
        print(facs)
        Vf = facs[-2]
        df = facs[-1]

        vtr.ydata[idx3:] -= Vf

        # # atr.ydata -= num.mean(atr.ydata)
        # vtr = own_integrate(atr, detrend=False, intval=1)
        # # facs = num.polyfit(vtr.get_xdata()[:idx1] - vtr.tmin, vtr.ydata[:idx1], 1)
        # facs = num.polyfit(vtr.get_xdata() - vtr.tmin, vtr.ydata, 1)
        # vtr.ydata = vtr.ydata - pf(vtr.get_xdata() - vtr.tmin, facs)
        # vtr.ydata -= vtr.ydata[0]

        # if plot:
        #     ax = axes[6]
        #     ax.axvline(dt * idx3 + vtr.tmin)
        #     ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)
        
        tr = own_integrate(vtr, detrend=False, intval=1)
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin, tr.ydata[:idx1], 1)
        tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)

    return tr


def trace_baseline_correction_ebasco_own(atr, comp, idx1, idx2, idx3, axes, mode='tilt', plot=True):

    if comp == 'N':
        color = 'green'
        # continue
    elif comp == 'E':
        color = 'orange'
        # continue
    elif comp == 'Z':
        color = 'blue'

    ## does smth
    dt = atr.deltat
    if plot:
        axes[3].plot(atr.get_xdata(), atr.ydata, color=color, label=comp)
        axes[3].axhline(y=0, color='gray', zorder=-10)
        # axes[3].set_ylim((-0.2, 0.2))

    vtr = own_integrate(atr, detrend=False, intval=1)

    ## own ebasco
    if mode == 'tilt':
    # if True:
        if plot:
            axes[0].set_title('Tilted (?)')

        facs = num.polyfit(vtr.get_xdata()[:idx1] - vtr.tmin, vtr.ydata[:idx1], 1)
        m1 = facs[-2]

        facs = num.polyfit(vtr.get_xdata()[idx3:] - vtr.tmin, vtr.ydata[idx3:], 1)
        m2 = facs[-2]

        atr.ydata[:idx2] -= m1
        atr.ydata[idx2:] -= m2
        atr.ydata -= atr.ydata[0]
        # atr.ydata -= num.mean(atr.ydata[:idx1])

        vtr = own_integrate(atr, detrend=False, intval=1)
        vtr.ydata -= vtr.ydata[0]


        if plot:
            ax = axes[4]
            # ax.plot(vtr.get_xdata()[:idx1], vtr.ydata[:idx1], color=color, label=comp)
            # ax.plot(atr.get_xdata()[:idx1], atr.ydata[:idx1], color=color, label=comp)

            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)
            # ax.axvline(dt * idx1 + vtr.tmin)


        tr = own_integrate(vtr, detrend=False, intval=1)
        facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin, tr.ydata[:idx1], 1)
        m3 = facs[-2]

        facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin, tr.ydata[idx3:], 1)
        m4 = facs[-2]

        # vtr.ydata[:idx1] -= m3
        # vtr.ydata[idx3:] -= m4
        # vtr.ydata -= num.mean(vtr.ydata[:idx1])
        vtr.ydata -= vtr.ydata[0]

        tr = own_integrate(vtr, detrend=False, intval=1)
        tr.ydata -= tr.ydata[0]

        if plot:
            ax = axes[5]
            # ax = axes[5]
            # ax.axvline(dt * idx2 + vtr.tmin)
            # ax.axvline(dt * idx2n + vtr.tmin)
            ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)
            # ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)
            # ax.plot(vtr.get_xdata()[idx3:], ((vtr.get_xdata()[idx3:] - vtr.tmin) * m2))

        # vtr = own_integrate(atr, detrend=False, intval=1)
        # vtr.ydata -= vtr.ydata[0]
        # facs = num.polyfit(vtr.get_xdata()[:idx1] - vtr.tmin,
        #                 vtr.ydata[:idx1], 1)
        # m1 = facs[-2]

        # facs = num.polyfit(vtr.get_xdata()[idx3:] - vtr.tmin,
        #                 vtr.ydata[idx3:], 1)
        # m2 = facs[-2]

        # print(m1, m2)

        # atr.ydata[:idx2] -= m1
        # atr.ydata[idx2:] -= m2

        

    else:
        if plot:
            axes[0].set_title('None tilt')
        facs = num.polyfit(vtr.get_xdata()[:idx1] - vtr.tmin,
                        vtr.ydata[:idx1], 1)
        vtr.ydata[:idx1] = vtr.ydata[:idx1] - pf(vtr.get_xdata()[:idx1] - vtr.tmin, facs) + pf(vtr.get_xdata()[:idx1] - vtr.tmin, facs)[-1]
        # vtr.ydata[:idx2] = vtr.ydata[:idx2] - pf(vtr.get_xdata()[:idx2] - vtr.tmin, facs) + pf(vtr.get_xdata()[:idx2] - vtr.tmin, facs)[-1]

        vtr.ydata = vtr.ydata - num.mean(vtr.ydata[:idx1])
        
        if plot:
            ax = axes[4]
            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)
        

        facs = num.polyfit(vtr.get_xdata()[idx2:] - vtr.tmin,
                        vtr.ydata[idx2:], 1)
        # vtr.ydata[idx3:] = vtr.ydata[idx3:] - pf(vtr.get_xdata()[idx3:] - vtr.tmin, facs) + pf(vtr.get_xdata()[idx3:] - vtr.tmin, facs)[0]
        vtr.ydata[idx2:] = vtr.ydata[idx2:] - pf(vtr.get_xdata()[idx2:] - vtr.tmin, facs) + pf(vtr.get_xdata()[idx2:] - vtr.tmin, facs)[0]
        
        # vtr.ydata -= num.mean(vtr.ydata[:idx1])
        # facs = num.polyfit(vtr.get_xdata() - vtr.tmin, vtr.ydata, 1)
        # vtr.ydata = vtr.ydata - pf(vtr.get_xdata() - vtr.tmin, facs)

        # vtr.ydata -= vtr.ydata[0]

        if plot:
            ax = axes[5]
            ax.axvline(dt * idx3 + vtr.tmin)
            ax.plot(vtr.get_xdata(), vtr.ydata, color=color, label=comp)

    # vtr.ydata -= vtr.ydata[0]
    # tr = own_integrate(vtr, detrend=False, intval=1)
    # facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin, tr.ydata[:idx1], 1)
    # tr.ydata -= pf(tr.get_xdata() - tr.tmin, facs)

    # facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin, tr.ydata[idx3:], 1)
    # tr.ydata[idx2:] = tr.ydata[idx2:] - pf(tr.get_xdata()[idx2:] - tr.tmin, facs) + pf(tr.get_xdata()[idx2:] - tr.tmin, facs)[0]

    # vtr.ydata[idx2:] -= facs[-2]
    # tr = own_integrate(vtr, detrend=False, intval=1)
    # facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin, tr.ydata[:idx1], 1)
    # tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)

    return tr


def trace_baseline_correction_vel(tr, comp, idx1, idx2, idx3, axes, mode=None, plot=True):

    if comp == 'N':
        color = 'green'
        # continue
    elif comp == 'E':
        color = 'orange'
        # continue
    elif comp == 'Z':
        color = 'blue'

    ## does smth
    dt = tr.deltat
    tr.ydata = tr.ydata - num.mean(tr.ydata[:idx1])
    if plot:
        ax = axes[3]
        ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

    # if mode == 'tilt':
    if True:
        # facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
        #                 tr.ydata[:idx1], 2)
        # tr.ydata[:idx2] = tr.ydata[:idx2] - pf(tr.get_xdata()[:idx2] - tr.tmin, facs) + pf(tr.get_xdata()[:idx2] - tr.tmin, facs)[-1]
        # # tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
        # tr.ydata = tr.ydata - num.mean(tr.ydata[:idx1])
        # if plot:
        #     ax = axes[4]
        #     ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)

        idxs = num.where(num.diff(num.sign(tr.ydata)))[0]

        # print(idxs[idxs > idx2][0])
        # exit()
        # idx2 = idxs[idxs > idx2][0]

        ## removes the trend of the noise (idx3) for the whole trace
        ax = axes[5]
        # facs = num.polyfit(tr.get_xdata()[:idx1] - tr.tmin,
        #                 tr.ydata[:idx1], 3)
        facs = num.polyfit(tr.get_xdata()[idx3:] - tr.tmin,
                        tr.ydata[idx3:], 1)
        # tr.ydata = tr.ydata - pf(tr.get_xdata() - tr.tmin, facs)
        tr.ydata[idx2:] = tr.ydata[idx2:] - pf(tr.get_xdata()[idx2:] - tr.tmin, facs) + pf(tr.get_xdata()[idx2:] - tr.tmin, facs)[0]
        if plot:
            ax.axvline(dt * idx2 + tr.tmin, linewidth=2, color='yellow')
            ax.axvline(dt * idx3 + tr.tmin)
            ax.plot(tr.get_xdata(), tr.ydata, color=color, label=comp)
        # tr.ydata = tr.ydata - GMs.pf(tr.get_xdata() - tr.tmin, facs)

    ### sets the noise to zero
    tr.ydata = tr.ydata - num.mean(tr.ydata[:idx1])
    disptr = own_integrate(tr, detrend=True, intval=1)
    facs = num.polyfit(disptr.get_xdata()[idx2:] - disptr.tmin,
                    disptr.ydata[idx2:], 1)
    disptr.ydata[idx2:] = disptr.ydata[idx2:] - pf(disptr.get_xdata()[idx2:] - disptr.tmin, facs) + pf(disptr.get_xdata()[idx2:] - disptr.tmin, facs)[0]
    
    facs = num.polyfit(disptr.get_xdata()[:idx1] - disptr.tmin,
                    disptr.ydata[:idx1], 1)
    disptr.ydata = disptr.ydata - pf(disptr.get_xdata() - disptr.tmin, facs)
    

    return disptr
    # return tr


def get_tilt_point(accZ, accE, accN, sta=None, eq=None, dist=None, plot=True, mode='all'):
    
    tmin = max(accZ.tmin, accE.tmin, accN.tmin)
    tmax = min(accZ.tmax, accE.tmax, accN.tmax)
    accZ.chop(tmin, tmax, include_last=True)
    accE.chop(tmin, tmax, include_last=True)
    accN.chop(tmin, tmax, include_last=True)
    # dt = accZ.deltat

    jerkZ = own_differentation(accZ, intval=1, chop=False, transfercut=0)
    jerkE = own_differentation(accE, intval=1, chop=False, transfercut=0)
    jerkN = own_differentation(accN, intval=1, chop=False, transfercut=0)

    snapZ = own_differentation(accZ, intval=2, chop=False, transfercut=0)
    snapE = own_differentation(accE, intval=2, chop=False, transfercut=0)
    snapN = own_differentation(accN, intval=2, chop=False, transfercut=0)

    accEne = accZ.ydata**2 + accE.ydata**2 + accN.ydata**2

    jerkEne = jerkZ.ydata**2 + jerkE.ydata**2 + jerkN.ydata**2
    snapEne = snapZ.ydata**2 + snapE.ydata**2 + snapN.ydata**2

    idxEnemax = num.argmax(accEne)
    Enemax = accZ.deltat * idxEnemax + accZ.tmin

    idxjEnemax = num.argmax(jerkEne)
    jEnemax = jerkZ.deltat * idxjEnemax + jerkZ.tmin

    idxsEnemax = num.argmax(snapEne)
    sEnemax = snapZ.deltat * idxsEnemax + snapZ.tmin

    maxindv = 0.
    maxjindv = 0
    maxsindv = 0.
    for comp in ['Z', 'E', 'N']:
        if comp == 'Z':
            tr = accZ
            jtr = jerkZ
            sntr = snapZ

        elif comp == 'E':
            tr = accE
            jtr = jerkE
            sntr = snapE

        elif comp == 'N':
            tr = accN
            jtr = jerkN
            sntr = snapN

        amp = num.max(num.abs(tr.ydata))
        ampjerk = num.max(num.abs(jtr.ydata))
        ampsnap = num.max(num.abs(jtr.ydata))

        if amp > maxindv:
            maxindv = amp
            maxindxidx = num.argmax(num.abs(tr.ydata))
            indvtime = tr.deltat * maxindxidx + tr.tmin

        if ampjerk > maxjindv:
            maxjindv = ampjerk
            maxjindvidx = num.argmax(num.abs(jtr.ydata))
            jindvtime = jtr.deltat * maxjindvidx + jtr.tmin

        if ampsnap > maxsindv:
            maxsindv = ampsnap
            maxsindvidx = num.argmax(num.abs(sntr.ydata))
            sindvtime = sntr.deltat * maxsindvidx + sntr.tmin

    if plot:
        plotdir = '/home/lehmann/dr/plots/baseline/%s/tilt_point/' % (eq)
        if os.path.exists(plotdir):
            pass
        else:
            os.mkdir(plotdir)

        print('Plot Tilt point')
        f, axes = plt.subplots(6, 1, sharex=True, figsize=(16, 9))

        tmin = accN.tmin
        axes[0].set_title('%s\ndist=%.1f' % (sta, dist))
        axes[0].plot(accZ.get_xdata(), accZ.ydata, label='Z', alpha=0.5, color='b')
        axes[0].plot(accE.get_xdata(), accE.ydata, label='E', alpha=0.5, color='orange')
        axes[0].plot(accN.get_xdata(), accN.ydata, label='N', alpha=0.5, color='g')

        axes[0].axvline(indvtime, color='red', linestyle='-', zorder=10)

        #
        axes[1].plot(accN.get_xdata(), accEne / max(accEne), color='k')
        axes[1].plot(accN.get_xdata(), num.cumsum(accEne) / max(num.cumsum(accEne)), alpha=0.5, color='r')
        
        axes[1].axvline(Enemax, color='blue', linestyle='-', zorder=10)

        ##
        axes[2].plot(jerkZ.get_xdata(), jerkZ.ydata, label='Z', alpha=0.5, color='b')
        axes[2].plot(jerkE.get_xdata(), jerkE.ydata, label='E', alpha=0.5, color='orange')
        axes[2].plot(jerkN.get_xdata(), jerkN.ydata, label='N', alpha=0.5, color='g')

        axes[2].axvline(jindvtime, color='red', linestyle='-', zorder=10)

        #
        axes[3].plot(jerkZ.get_xdata(), jerkEne / max(jerkEne), color='k')
        axes[3].plot(jerkZ.get_xdata(), num.cumsum(jerkEne) / max(num.cumsum(jerkEne)), alpha=0.5, color='r')
        
        axes[3].axvline(jEnemax, color='blue', linestyle='-', zorder=10)

        ##
        axes[4].plot(snapZ.get_xdata(), snapZ.ydata, label='Z', alpha=0.5, color='b')
        axes[4].plot(snapE.get_xdata(), snapE.ydata, label='E', alpha=0.5, color='orange')
        axes[4].plot(snapN.get_xdata(), snapN.ydata, label='N', alpha=0.5, color='g')

        axes[4].axvline(sindvtime, color='red', linestyle='-', zorder=10)

        #
        axes[5].plot(snapZ.get_xdata(), snapEne / max(snapEne), color='k')
        axes[5].plot(snapZ.get_xdata(), num.cumsum(snapEne) / max(num.cumsum(snapEne)), alpha=0.5, color='r')
        
        axes[5].axvline(sEnemax, color='blue', linestyle='-', zorder=10)

        for ii in range(6):
            # axes[ii].axvline(smax, color='purple', linestyle='-', zorder=10)
            # axes[ii].axvline(pgatime, color='red', linestyle='-', zorder=10)

            axes[ii].set_xlim((indvtime - 20, indvtime + 30))

        f.savefig('%s%s.png' % (plotdir, sta))

    # print(sta, num.round(indvtime - tmin, 3), num.round(Enemax - tmin, 3))

    if mode == 'indv':
        tilttime = min(indvtime, jindvtime, sindvtime)
    if mode == 'all':
        tiltmaxlist = [Enemax, jEnemax, sEnemax]
        tiltmaxlist = [i for i in tiltmaxlist if i > (accZ.tmin + 5 * accZ.deltat)]
        tilttime = min(tiltmaxlist)

    # print(mode, tilttime)
    # print(indvtime, jindvtime, sindvtime, Enemax, jEnemax, sEnemax)
    # exit()

    return tilttime
