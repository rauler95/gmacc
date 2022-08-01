import numpy as num
import geopandas as gpd
import traceback
import math

from scipy.integrate import cumtrapz

import eqsig
from pyrocko import orthodrome, gf, trace
from pyrocko.guts import Object, StringChoice, Float, String, List, Dict, Choice, load_all

from openquake.hazardlib.geo import geodetic, Point, Mesh, PlanarSurface

#############################
### Constants
#############################
deg = 111.19
G = 9.81  # m/s*s

#############################
### Classes
#############################
class Cloneable(object):

    def __iter__(self):
        return iter(self.T.propnames)

    def __getitem__(self, k):
        if k not in self.keys():
            raise KeyError(k)

        return getattr(self, k)

    def __setitem__(self, k, v):
        if k not in self.keys():
            raise KeyError(k)

        return setattr(self, k, v)

    def clone(self, **kwargs):
        '''
        Make a copy of the object.

        A new object of the same class is created and initialized with the
        parameters of the object on which this method is called on. If
        ``kwargs`` are given, these are used to override any of the
        initialization parameters.
        '''

        d = dict(self)
        for k in d:
            v = d[k]
            if isinstance(v, Cloneable):
                d[k] = v.clone()

        d.update(kwargs)
        return self.__class__(**d)

    @classmethod
    def keys(cls):
        '''
        Get list of the source model's parameter names.
        '''

        return cls.T.propnames


class SourceClass(Object, Cloneable):
    name = String.T()
    depth = Float.T()
    lon = Float.T()
    lat = Float.T()
    magnitude = Float.T()
    time = Float.T(optional=True)

    form = StringChoice.T(choices=['point', 'rectangular', 'pdr'], default='point')
    nucleation_x = Float.T(default=0.)
    nucleation_y = Float.T(default=0.)

    ### optional
    strike = Float.T(optional=True)
    dip = Float.T(optional=True)
    rake = Float.T(optional=True)
    width = Float.T(optional=True)
    length = Float.T(optional=True)
    duration = Float.T(optional=True)
    rupture = Dict.T(optional=True)
    nucleationCoords = List.T(optional=True)
    moment = Float.T(optional=True)
    slip = Float.T(optional=True)
    tensor = Dict.T(optional=True)
    ztor = Float.T(optional=True)

    risetime = Float.T(optional=True)
    rupture_velocity = Float.T(optional=True)

    pyrockoSource = gf.Source.T(optional=True)

    def update(self, **kwargs):
        for (k, v) in kwargs.items():
            self[k] = v

    def create_synthetic_rupture_plain_from_hypo(self):
        nucXfac2 = (self.nucleation_x + 1) / 2  # convert to 0-1 range
        nucXfac1 = 1 - nucXfac2
        nucYfac = ((self.nucleation_y + 1) / 2)  # convert to 0-1 range

        surface_width = self.width * num.cos(num.radians(self.dip))  # length of width projected to surface
        depth_range = self.width * num.sin(num.radians(self.dip))  # length of depth along Z

        dist1 = num.sqrt((nucYfac * surface_width)**2
                        + (self.length * nucXfac1)**2)
        dist2 = num.sqrt((nucYfac * surface_width)**2
                        + (self.length * nucXfac2)**2)

        if nucXfac2 <= 0:
            nucXfac2 = 0.001

        if nucXfac1 <= 0:
            nucXfac1 = 0.001

        azi1 = self.strike - (num.arctan((nucYfac * surface_width)
                        / (self.length * nucXfac1)) * (180. / num.pi))
        azi2 = self.strike - 180. + (num.arctan((nucYfac * surface_width)
                    / (self.length * nucXfac2)) * (180 / num.pi))

        depth1 = self.depth - nucYfac * depth_range
        depth2 = self.depth - nucYfac * depth_range

        lat1, lon1 = orthodrome.azidist_to_latlon(self.lat, self.lon,
                                                  azi1, dist1 / deg)
        lat2, lon2 = orthodrome.azidist_to_latlon(self.lat, self.lon,
                                                  azi2, dist2 / deg)
        p1 = Point(lon1, lat1, depth1)
        p2 = Point(lon2, lat2, depth2)
        p3 = p1.point_at(surface_width, depth_range, self.strike + 90)
        p4 = p2.point_at(surface_width, depth_range, self.strike + 90)

        self.rupture = {'UR': [p1.longitude, p1.latitude, p1.depth],
                        'UL': [p2.longitude, p2.latitude, p2.depth],
                        'LL': [p4.longitude, p4.latitude, p4.depth],
                        'LR': [p3.longitude, p3.latitude, p3.depth]}

        self.form = 'rectangular'

    def create_rupture_surface(self):
        if self.form == 'point':
            surface = Mesh(num.array([self.lon]),
                           num.array([self.lat]),
                           num.array([self.depth])
                           )
            self.surface = surface

        elif self.form == 'rectangular':
            '''
            To do:
            - check if the strike is always correct calculated
            '''

            p1 = Point(self.rupture['UR'][0], self.rupture['UR'][1],
                       self.rupture['UR'][2])
            p2 = Point(self.rupture['UL'][0], self.rupture['UL'][1],
                       self.rupture['UL'][2])
            p3 = Point(self.rupture['LL'][0], self.rupture['LL'][1],
                       self.rupture['LL'][2])
            p4 = Point(self.rupture['LR'][0], self.rupture['LR'][1],
                       self.rupture['LR'][2])

            if p1.depth <= 0:
                p1.depth = 0.01
                print('Point 1 above ground. Set rupture top depth to 10m.')
            if p2.depth <= 0:
                p2.depth = 0.01
                print('Point 2 above ground. Set rupture top depth to 10m.')
            # print(p1)
            # print(p2)
            # print(p3)
            # print(p4)
            # exit()

            surface = PlanarSurface.from_corner_points(top_left=p2,
                                                       top_right=p1,
                                                       bottom_left=p3,
                                                       bottom_right=p4)

            testdist = geodetic.geodetic_distance(p1.longitude, p1.latitude,
                                                  p4.longitude, p4.latitude)
            p5 = geodetic.point_at(p1.longitude, p1.latitude,
                                   surface.get_strike() + 90., testdist)
            valdist = geodetic.geodetic_distance(p5[0], p5[1],
                                                p4.longitude, p4.latitude)
            # print(valdist)
            # dipazi = geodetic.azimuth(p1.longitude, p1.latitude, p4.longitude, p4.latitude)-90
            # print('Azimuth Difference', dipazi - surface.get_strike(), dipazi, surface.get_strike())
            # if abs(dipazi - surface.get_strike()) > 5.:
            if valdist > 1:
                surface = PlanarSurface.from_corner_points(top_left=p1,
                                                           top_right=p2,
                                                           bottom_left=p4,
                                                           bottom_right=p3)

                testdist = geodetic.geodetic_distance(p1.longitude, p1.latitude,
                                                    p4.longitude, p4.latitude)
                p5 = geodetic.point_at(p1.longitude, p1.latitude,
                                       surface.get_strike() + 90., testdist)
                valdist = geodetic.geodetic_distance(p5[0], p5[1],
                                                    p4.longitude, p4.latitude)

                # dipazi = geodetic.azimuth(p1.longitude, p1.latitude, p4.longitude, p4.latitude)-90
                # print('Azimuth Difference', dipazi - surface.get_strike(), dipazi, surface.get_strike())
                # if abs(dipazi - surface.get_strike()) > 5.:
                # print(valdist)
                if valdist > 1:
                    print('Strike: %0.1f; dip: %0.1f; rake: %0.1f'
                        % (surface.get_strike(), surface.get_dip(), self.rake))
                    print(p1)
                    print(p2)
                    print(p3)
                    print(p4)
                    print('Check rupture plain')
                    exit()

                else:
                    # pass
                    print('Switched Rupture points to get:')
                    print('Strike: %0.1f; dip: %0.1f; rake: %0.1f'
                        % (surface.get_strike(), surface.get_dip(), self.rake))

            self.surface = surface
            self.strike = float(self.surface.get_strike())
            self.dip = float(self.surface.get_dip())

            if self.rake is None:
                self.rake = 0.0
                print('Set rake manually to 0')

            self.ztor = float(self.surface.get_top_edge_depth())

            self.area = float(self.surface.get_area())
            self.width = float(self.surface.get_width())
            self.length = self.area / self.width

            ### Convert nucleation coordinates
            if self.nucleationCoords:
                nucLat = self.nucleationCoords[1]
                nucLon = self.nucleationCoords[0]
                nucDepth = self.nucleationCoords[2]
            else:
                nucLat = self.lat
                nucLon = self.lon
                nucDepth = self.depth

            upperDepth = self.ztor
            lowerDepth = surface.bottom_right.depth

            midDepth = (upperDepth + lowerDepth) / 2.
            self.nucleation_y = float((nucDepth - midDepth)
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
            self.nucleation_x = float(-((nucX * 2) - 1))

        return

    def calc_distance(self, lons, lats, distType='rhypo'):

        locPoints = []
        dists = []
        for lon, lat in zip(lons, lats):
            if distType == 'rhypo':
                dists.append(geodetic.distance(self.lon, self.lat, self.depth,
                                        lon, lat, 0.))

            else:
                locPoints.append(Point(lon, lat))

        if hasattr(self, 'surface'):
            if distType == 'rjb':
                dists = self.surface.get_joyner_boore_distance(
                                        Mesh.from_points_list(locPoints))

            elif distType == 'ry0':
                if self.refSource.form == 'rectangular':
                    dists = self.surface.get_ry0_distance(
                                            Mesh.from_points_list(locPoints))
                else:
                    dists = num.zeros(len(locPoints)) * num.nan

            elif distType == 'rx':
                if self.refSource.form == 'rectangular':
                    dists = self.surface.get_rx_distance(
                                            Mesh.from_points_list(locPoints))
                else:
                    dists = num.zeros(len(locPoints)) * num.nan

            elif distType == 'rrup':
                dists = self.surface.get_min_distance(
                                        Mesh.from_points_list(locPoints))

        return dists

    def calc_azimuth(self, lons, lats):

        azimuths = geodetic.azimuth(
                        self.lon, self.lat,
                        lons, lats)
        return azimuths

    def calc_rupture_azimuth(self, lons, lats):

        mesh = Mesh(num.array(lons), num.array(lats))

        rup_azimuth = self.surface.get_azimuth_of_closest_point(mesh)
        centre_azimuth = self.surface.get_azimuth(mesh)

        return rup_azimuth, centre_azimuth


class GMClass(Object, Cloneable):
    name = String.T()
    value = Float.T()
    unit = String.T(optional=True)


# class TraceClass(trace.Object):
#   tr = trace.Object.D(optional=False)


class ComponentGMClass(Object, Cloneable):
    component = String.T()
    gms = Dict.T(
        optional=True,
        content_t=GMClass.T())

    traces = Dict.T()
    # trace = Any.T()
    # trace = TraceClass.T(optional=True)
    # trace.Object.D(
    #   optional=True)


class StationGMClass(Object, Cloneable):
    network = String.T()
    station = String.T()
    lon = Float.T()
    lat = Float.T()

    components = Dict.T(
        optional=True,
        content_t=ComponentGMClass.T())

    azimuth = Float.T(optional=True)
    rup_azimuth = Float.T(optional=True)
    centre_azimuth = Float.T(optional=True)
    rhypo = Float.T(optional=True)
    rjb = Float.T(optional=True)
    ry0 = Float.T(optional=True)
    rx = Float.T(optional=True)
    rrup = Float.T(optional=True)
    vs30 = Float.T(optional=True)

    azimuth = Float.T(optional=True)
    rupture_azimuth = Float.T(optional=True)
    centre_azimuth = Float.T(optional=True)


class StationContainer(Object, Cloneable):
    stations = Dict.T(
        content_t=StationGMClass.T())

    refSource = Choice.T([
                SourceClass.T()],
                optional=True)

    def calc_distances(self, distTypes=['rhypo', 'rjb', 'rx', 'ry0', 'rrup']):
        locPoints = []
        for ii in self.stations:
            if 'rhypo' in distTypes:
                rhypo = geodetic.distance(self.refSource.lon, self.refSource.lat,
                                        self.refSource.depth,
                                        self.stations[ii].lon, self.stations[ii].lat,
                                        0.)
                self.stations[ii].rhypo = float(rhypo)

            locPoints.append(Point(self.stations[ii].lon, self.stations[ii].lat))

        if hasattr(self.refSource, 'surface'):

            if 'rjb' in distTypes:
                rjbs = self.refSource.surface.get_joyner_boore_distance(
                                        Mesh.from_points_list(locPoints))

            if 'ry0' in distTypes:
                if self.refSource.form == 'rectangular':
                    ry0s = self.refSource.surface.get_ry0_distance(
                                            Mesh.from_points_list(locPoints))
                else:
                    ry0s = num.zeros(len(locPoints)) * num.nan

            if 'rx' in distTypes:
                if self.refSource.form == 'rectangular':
                    rxs = self.refSource.surface.get_rx_distance(
                                            Mesh.from_points_list(locPoints))
                else:
                    rxs = num.zeros(len(locPoints)) * num.nan

            if 'rrup' in distTypes:
                rrups = self.refSource.surface.get_min_distance(
                                        Mesh.from_points_list(locPoints))

            for nn, ii in enumerate(self.stations):
                if 'rrup' in distTypes:
                    self.stations[ii].rrup = float(rrups[nn])
                if 'rx' in distTypes:
                    self.stations[ii].rx = float(rxs[nn])
                if 'ry0' in distTypes:
                    self.stations[ii].ry0 = float(ry0s[nn])
                if 'rjb' in distTypes:
                    self.stations[ii].rjb = float(rjbs[nn])

        return

    def get_distances(self, distTypes=['rhypo', 'rjb', 'rx', 'ry0', 'rrup']):
        distDict = {}
        for sta, staDict in self.stations.items():
            for disttyp in distTypes:
                if disttyp not in distDict:
                    distDict[disttyp] = []
                distDict[disttyp].append(staDict[disttyp])

        return distDict

    def calc_azimuths(self, aziTypes=['azimuth', 'rupture_azimuth', 'centre_azimuth']):

        lons = []
        lats = []
        for ii in self.stations:
            lons.append(self.stations[ii].lon)
            lats.append(self.stations[ii].lat)

        if 'azimuth' in aziTypes:
            azimuth = geodetic.azimuth(
                            self.refSource.lon, self.refSource.lat,
                            lons, lats)

        mesh = Mesh(num.array(lons), num.array(lats))
        if hasattr(self.refSource, 'surface'):
            if 'rupture_azimuth' in aziTypes:
                rupAzis = self.refSource.surface.get_azimuth_of_closest_point(mesh)
            
            if 'centre_azimuth' in aziTypes:
                centroidAzis = self.refSource.surface.get_azimuth(mesh)

        for nn, ii in enumerate(self.stations):
            if 'azimuth' in aziTypes:
                self.stations[ii].azimuth = float(azimuth[nn])
            if hasattr(self.refSource, 'surface'):
                if 'rupture_azimuth' in aziTypes:
                    self.stations[ii].rupture_azimuth = float(rupAzis[nn])
                if 'centre_azimuth' in aziTypes:
                    self.stations[ii].centre_azimuth = float(centroidAzis[nn])

        return

    def get_azimuths(self, aziTypes=['azimuth', 'rupture_azimuth', 'centre_azimuth']):
        aziDict = {}
        for sta, staDict in self.stations.items():
            for azitype in aziTypes:
                if azitype not in aziDict:
                    aziDict[azitype] = []
                aziDict[azitype].append(staDict[azitype])

        return aziDict

    def get_gm_values(self):
        gmdict = {}
        for sta in self.stations:
            for comp in self.stations[sta].components:
                for gm in self.stations[sta].components[comp].gms:
                    ims = '%s_%s' % (comp, gm)
                    if ims not in gmdict:
                        gmdict[ims] = []
                    val = self.stations[sta].components[comp].gms[gm].value
                    gmdict[ims].append(val)

        return gmdict

    def get_gm_from_wv(self, imts=['pga', 'pgv'], freqs=[0.3, 1.0, 3.0], H2=False,
                       delete=False, deleteWvData=True):
        '''
        To do:
        - clean
        - stuetzstellen einfuegen mit sinc-interpolation (hochsamplen /resamplen)


        seee https://github.com/emolch/wafe/blob/master/src/measure.py
        '''

        tfade = 2

        nfreqs = []
        for freq in freqs:
            nfreqs.append(float('%.3f' % float(freq)))
        freqs = num.array(nfreqs)

        for sta in self.stations:

            ### Horizontal Vector Sum component
            if H2:
                cha = 'H'

                if 'E' in self.stations[sta].components.keys() \
                    and 'N' in self.stations[sta].components.keys():

                    tracesE = self.stations[sta].components['E'].traces
                    trDE = tracesE['disp'].copy()
                    trVE = tracesE['vel'].copy()
                    trAE = tracesE['acc'].copy()

                    tracesN = self.stations[sta].components['N'].traces
                    trDN = tracesN['disp'].copy()
                    trVN = tracesN['vel'].copy()
                    trAN = tracesN['acc'].copy()

                    COMP = ComponentGMClass(component=cha, gms={})
                    flagshort = False

                    if len(trAE.ydata) < 500. or len(trAN.ydata) < 500.:
                        print('One of the horizontal components is too short')
                        print('E:', len(trAE.ydata), 'N:', len(trAN.ydata))
                        flagshort = True

                    else:
                        tmin = max(trAE.tmin, trAN.tmin)
                        tmax = min(trAE.tmax, trAN.tmax)
                        trAE.chop(tmin, tmax, include_last=True)
                        trAN.chop(tmin, tmax, include_last=True)

                        tmin = max(trVE.tmin, trVN.tmin)
                        tmax = min(trVE.tmax, trVN.tmax)
                        trVE.chop(tmin, tmax, include_last=True)
                        trVN.chop(tmin, tmax, include_last=True)

                        tmin = max(trDE.tmin, trDN.tmin)
                        tmax = min(trDE.tmax, trDN.tmax)
                        trDE.chop(tmin, tmax, include_last=True)
                        trDN.chop(tmin, tmax, include_last=True)

                        deltaT = trAE.deltat
                        dataAH = num.abs(num.sqrt(trAE.ydata**2 + trAN.ydata**2))
                        dataVH = num.abs(num.sqrt(trVE.ydata**2 + trVN.ydata**2))
                        dataDH = num.abs(num.sqrt(trDE.ydata**2 + trDN.ydata**2))

                    sa_freqs = []
                    for gm in imts:
                        if flagshort:
                            val = 0.000001
                            unit = 'NaN'

                        elif 'sigdur' == gm:
                            dur = eqsig.im.calc_sig_dur_vals(dataAH,
                                                            dt=deltaT)
                            val = dur
                            unit = 's'

                        elif 'ai' == gm:
                            ai = arias_intensity(dataAH, dt=deltaT)
                            val = num.log10(ai.max())
                            unit = 'm/s'

                        elif 'pga' == gm:
                            pga = num.abs(dataAH).max()
                            val = num.log10((pga / G) * 100.)  # in g%
                            unit = 'g%'

                            # rotsPga = [num.max(abs(tsE.values*num.cos(num.pi*ang/180) + tsN.values*num.sin(num.pi*ang/180))) for ang in num.linspace(0,180,181) ] 
                            # print('%.2f, %.3f %.3f || %.3f %.3f %.3f' %(pga/num.sqrt(max(tsE.values)*max(tsN.values)), pga, num.sqrt(max(tsE.values)*max(tsN.values)), num.max(rotsPga), num.median(rotsPga), num.min(rotsPga)))

                        elif 'vi' == gm:
                            vi = integrated_velocity(dataVH, dt=deltaT)
                            val = num.log10(vi.max())
                            unit = 'm'

                        elif 'pgv' == gm:
                            pgv = num.abs(dataVH).max()
                            val = num.log10(pgv * 100.)  # in cm/s
                            unit = 'cm/s'

                        elif 'pgd' == gm:
                            pgd = num.abs(dataDH).max()
                            val = num.log10(pgd * 100.)  # in cm
                            unit = 'cm'

                        elif 'SA' in gm:
                            sa_freqs.append(float(gm.rsplit('_')[-1]))
                            continue

                        GM = GMClass(
                            name=gm,
                            value=float(val),
                            unit=unit)

                        COMP.gms[GM.name] = GM

                    if len(sa_freqs) > 0:
                        TrAcc = eqsig.single.AccSignal(dataAH, deltaT)
                        TrAcc.generate_response_spectrum(sa_freqs)
                        for f, sa in zip(sa_freqs, TrAcc.s_a):
                            val = num.log10((sa / G) * 100.)
                            unit = 'UKN'
                            GM = GMClass(
                                    name='SA_%s' % f,
                                    value=float(val),
                                    unit=unit)
                            COMP.gms[GM.name] = GM

                    if freqs.size != 0:
                        if flagshort:
                            for frq in freqs:
                                val = num.nan
                                unit = 'NaN'

                                GM = GMClass(
                                    name='f_%s' % frq,
                                    value=float(val),
                                    unit='nan')

                                COMP.gms[GM.name] = GM

                        else:
                            spec = get_spectra(dataAH, deltaT, tfade)
                            vals = eqsig.functions.calc_smooth_fa_spectrum(
                                                spec[1], spec[0],
                                                num.array(freqs), band=40)

                            for frq, val in zip(freqs, vals):
                                GM = GMClass(
                                    name='f_%s' % frq,
                                    value=float(num.log10(val)),
                                    unit='m/s?')

                                COMP.gms[GM.name] = GM

                    self.stations[sta].components[COMP.component] = COMP

                    if delete:
                        del self.stations[sta].components['E']
                        del self.stations[sta].components['N']

            ### Standard components
            for cha in self.stations[sta].components:
                if cha == 'H':
                    continue

                COMP = ComponentGMClass(
                            component=cha,
                            gms={})

                traces = self.stations[sta].components[cha].traces
                trD = traces['disp'].copy()
                trV = traces['vel'].copy()
                trA = traces['acc'].copy()

                deltaT = trA.deltat
                dataA = trA.ydata
                dataV = trV.ydata
                dataD = trD.ydata

                sa_freqs = []
                for gm in imts:
                    if 'sigdur' == gm:
                        dur = eqsig.im.calc_sig_dur_vals(dataA,
                                                        dt=deltaT)
                        val = dur
                        unit = 's'

                    elif 'ai' == gm:
                        ai = arias_intensity(dataA, deltaT)
                        val = num.log10(ai.max())
                        unit = 'm/s'

                    elif 'pga' == gm:
                        pga = num.abs(dataA).max()
                        val = num.log10((pga / G) * 100.)
                        unit = 'g%'

                    elif 'vi' == gm:
                        vi = integrated_velocity(dataV, deltaT)
                        val = num.log10(vi.max())
                        unit = 'm'

                    elif 'pgv' == gm:
                        pgv = num.abs(dataV).max()
                        val = num.log10(pgv * 100.)
                        unit = 'cm/s'

                    elif 'pgd' == gm:
                        pgd = num.abs(dataD).max()
                        val = num.log10(pgd * 100.)
                        unit = 'cm'

                    elif 'SA' in gm:
                        sa_freqs.append(float(gm.rsplit('_')[-1]))
                        continue

                    GM = GMClass(
                                name=gm,
                                value=float(val),
                                unit=unit)
                    COMP.gms[GM.name] = GM

                if len(sa_freqs) > 0:
                    TrAcc = eqsig.single.AccSignal(dataA, deltaT)
                    TrAcc.generate_response_spectrum(sa_freqs)
                    for f, sa in zip(sa_freqs, TrAcc.s_a):
                        val = num.log10((sa / G) * 100.)
                        unit = 'UKN'
                        GM = GMClass(
                                name='SA_%s' % f,
                                value=float(val),
                                unit=unit)
                        COMP.gms[GM.name] = GM

                if freqs.size != 0:
                    spec = get_spectra(dataA, deltaT, tfade)
                    vals = eqsig.functions.calc_smooth_fa_spectrum(
                                        spec[1], spec[0],
                                        num.array(freqs), band=40)
                    for frq, val in zip(freqs, vals):
                        GM = GMClass(
                            name='f_%s' % frq,
                            value=float(num.log10(val)),
                            unit='m/s?')

                        COMP.gms[GM.name] = GM

                if deleteWvData:
                    pass
                else:
                    COMP.traces = self.stations[sta].components[cha].traces

                self.stations[sta].components[COMP.component] = COMP

        return

    def to_dictionary(self):
        Dict = {}
        for sta, staDict in self.stations.items():
            for comp, compDict in staDict.components.items():
                if len(comp) > 3:
                    pass
                else:
                    comp = comp[-1]

                for gm, gmDict in compDict.gms.items():
                    if gm not in Dict:
                        Dict[gm] = {}

                    if comp not in Dict[gm]:
                        Dict[gm][comp] = {'vals': [], 'lons': [], 'lats': []}

                    Dict[gm][comp]['vals'].append(gmDict['value'])
                    Dict[gm][comp]['lons'].append(staDict['lon'])
                    Dict[gm][comp]['lats'].append(staDict['lat'])

        return Dict

    def to_geodataframe(self):
        Dict = {}
        lons = []
        lats = []
        for sta, staDict in self.stations.items():
            lons.append(staDict['lon'])
            lats.append(staDict['lat'])
            for comp, compDict in staDict.components.items():
                if len(comp) > 3:
                    pass
                else:
                    comp = comp[-1]

                for gm, gmDict in compDict.gms.items():
                    chagm = '%s_%s' % (comp, gm)
                    if chagm not in Dict:
                        Dict[chagm] = []

                    Dict[chagm].append(gmDict['value'])

        Dict['st_lon'] = lons
        Dict['st_lat'] = lats

        gdf = gpd.GeoDataFrame(Dict, geometry=gpd.points_from_xy(lons, lats))

        return gdf

    def create_all_waveforms_synth(self, disp=True, vel=True, acc=True):

        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                trD = self.stations[sta].components[comp].trace.copy()

                del self.stations[sta].components[comp].trace

                if disp:
                    self.stations[sta].components[comp].traces['disp'] = trD

                if vel:
                    trV = own_differentation(trD, 1)
                    self.stations[sta].components[comp].traces['vel'] = trV

                if acc:
                    trA = own_differentation(trD, 2)
                    self.stations[sta].components[comp].traces['acc'] = trA

    def resample_waveform(self, resample_f=200, resample_fac=1.):
        for sta in self.stations:
            for comp in self.stations[sta].components.keys():
                for key, reftr in self.stations[sta].components[comp].traces.items():
                    tr = reftr.copy()

                    # tr.ydata -= num.mean(tr.ydata)
                    if resample_f:
                        if (1 / resample_f) != tr.deltat:
                            tr.resample(1 / resample_f)
                    elif resample_fac != 1:
                        tr.resample(tr.deltat * resample_fac)

                    ## cut 1.5s before and after, due to artifacts
                    tr.chop(tr.tmin + 1.5, tr.tmax - 1.5, include_last=True)  # cleaner way
                    self.stations[sta].components[comp].traces[key] = tr


#############################
### Support functions	
#############################
def merge_StationContainer(staCont1, staCont2):
    if staCont1.refSource != staCont2.refSource:
        print()
        print('EXIT\nSources are not the same for both Containers:')
        print('Source1:', staCont1.refSource)
        print('Source2:', staCont2.refSource)
        exit()

    newStaCont = StationContainer(
                    refSource=staCont1.refSource,
                    stations={})

    for sta1 in staCont1.stations:
        if sta1 not in staCont2.stations:
            newStaCont.stations[sta1] = staCont1.stations[sta1]

    for sta2 in staCont2.stations:
        if sta2 not in staCont1.stations:
            newStaCont.stations[sta2] = staCont2.stations[sta2]

    for sta1 in staCont1.stations:
        for sta2 in staCont2.stations:
            if sta1 != sta2:
                continue

            # if staCont1.stations[sta1].lon != staCont2.stations[sta2].lon or \
            #   staCont1.stations[sta1].lat != staCont2.stations[sta2].lat:
            #   continue

            newStaCont.stations[sta1] = staCont1.stations[sta1]

            for comp2 in staCont2.stations[sta2].components:
                if comp2 in newStaCont.stations[sta1]:
                    for gm2 in staCont2.stations[sta2].components[comp2]:
                        if gm2 in newStaCont.stations[sta1].components[comp2].gms:
                            if newStaCont.stations[sta1].components[comp2].gms[gm2] == staCont2.stations[sta2].components[comp2].gms[gm2]:
                                continue
                            else:
                                newStaCont.stations[sta1].components[comp2].gms[str(gm2) + '_2'] = staCont2.stations[sta2].components[comp2].gms[gm2] 
                        else:
                            newStaCont.stations[sta1].components[comp2].gms[gm2] = staCont2.stations[sta2].components[comp2].gms[gm2] 
                else:
                    newStaCont.stations[sta1].components[comp2] = staCont2.stations[sta2].components[comp2]

    return newStaCont


def integrated_velocity(array, dt):
    ai = cumtrapz(array ** 2, dx=dt, initial=0)
    return ai


def arias_intensity(array, dt):
    ### from eqsig copied
    # ai = eqsig.im.calc_arias_intensity(tr)
    ai = num.pi / (2 * 9.81) * cumtrapz(array ** 2,
                                    dx=dt,
                                    initial=0)
    return ai


def create_stationdict_synthetic(traces, wvtargets):
    staDict = {}
    for tr, wvtarget in zip(traces, wvtargets):
        ns = '%s.%s' % (tr.network, tr.station)

        cha = tr.channel
        COMP = ComponentGMClass(
                component=cha)
        COMP.trace = tr

        if ns not in staDict:
            STA = StationGMClass(
                network=tr.network,
                station=tr.station,
                lat=float(wvtarget.lat),
                lon=float(wvtarget.lon),
                components={})

            STA.components[COMP.component] = COMP
            staDict['%s.%s' % (STA.network, STA.station)] = STA

        else:
            staDict['%s.%s'
            % (tr.network, tr.station)].components[COMP.component] = COMP

    return staDict


def get_spectra(acc, dt, tfade, minlen=200.):
    # acc = signal.detrend(acc, type='constant')
    acc -= num.mean(acc)
    # acc = signal.detrend(acc)
    # acc = detrend.simple(acc)
    # print(acc)

    ndata = acc.size
    ### from pyrocko
    acc = acc * costaper(0., tfade,
                    dt * (ndata - 1) - tfade,
                    dt * ndata, ndata, dt)

    tlen = len(acc)
    minlen = minlen / dt
    if tlen < minlen:
        tdiff = minlen - tlen
        acc = num.pad(acc, (int(tdiff / 2), int(tdiff / 2)),
                    'linear_ramp', end_values=(0, 0))

    # elif tlen > minlen:
    #   acc = acc[:int(minlen)]

    # ts = AccSignal(acc, tr.deltat)
    # spec = eqsig.functions.calc_fa_spectrum(ts)

    # return spec

    # Tlen = len(acc)

    points = int(len(acc) / 2)
    fa = num.fft.fft(acc)
    famp = fa[range(points)] * dt  # * Tlen
    freqs = num.arange(points) / (2 * points * dt)

    return abs(famp), freqs


def own_differentation(intr, intval=1, chop=True, transfercut=1):
    tr = intr.copy()

    vtr = tr.transfer(transfercut,
                    # cut_off_fading=False,
                    transfer_function=trace.DifferentiationResponse(intval))

    if chop:
        vtr.chop(vtr.tmin + 1, vtr.tmax - 1, include_last=True)  # cleaner way

    #### überarbeiten!!
    # fint = int(3 / tr.deltat)
    # lint = -fint
    # vtr.ydata = vtr.ydata[fint:lint]
    return vtr


def get_distances(lons, lats, source, distType='hypo'):

    hypoLat = source.lat
    hypoLon = source.lon
    hypoDepth = source.depth / 1000.

    dists = []
    points = []
    for ii in range(len(lons)):
        if distType == 'hypo':
            dist = geodetic.distance(hypoLon, hypoLat, hypoDepth,
                                     lons[ii], lats[ii], 0.)
            dists.append(dist)
        else:
            points.append(Point(lons[ii], lats[ii]))

    if distType == 'hypo':
        del points
    else:
        surface = source.surface

        if distType == 'rjb':
            dists = surface.get_joyner_boore_distance(Mesh.from_points_list(points))
        elif distType == 'ry0':
            dists = surface.get_ry0_distance(Mesh.from_points_list(points))
        elif distType == 'rx':
            dists = surface.get_rx_distance(Mesh.from_points_list(points))
        elif distType == 'rrup':
            dists = surface.get_min_distance(Mesh.from_points_list(points))
        else:
            print('Wrong distType')
            exit()

    dists = num.array(dists)

    return dists


def get_azimuths(lons, lats, source, aziType='hypo'):

    if aziType == 'hypo':
        hypoLat = source.lat
        hypoLon = source.lon
        azimuths = orthodrome.azimuth_numpy(num.array(hypoLat), num.array(hypoLon),
                                            num.array(lats), num.array(lons))

    elif aziType == 'rup':
        mesh = Mesh(num.array(lons), num.array(lats))
        azimuths = source.surface.get_azimuth_of_closest_point(mesh)
    
    elif aziType == 'centre':
        mesh = Mesh(num.array(lons), num.array(lats))
        azimuths = source.surface.get_azimuth(mesh) 
    
    azimuths[azimuths > 180] = azimuths[azimuths > 180] - 360.

    return azimuths


def from_mtsource_to_own_source(src):
    from pyrocko import moment_tensor as pmt
    mt = pmt.MomentTensor(mnn=src.mnn, mee=src.mee, mdd=src.mdd,
            mne=src.mne, mnd=src.mnd, med=src.med)

    source = SourceClass(
        name='Synthetic',
        form='point',
        time=src.time,
        lon=float(src.lon),  # hypo
        lat=float(src.lat),  # hypo
        depth=float(src.depth),  # hypo
        magnitude=float(mt.moment_magnitude()),
        strike=float(mt.strike1),
        dip=float(mt.dip1),
        rake=float(mt.rake1),
        tensor=dict(mnn=src.mnn, mee=src.mee, mdd=src.mdd,
            mne=src.mne, mnd=src.mnd, med=src.med))

    return source


def from_rectsource_to_own_source(src):
    cp = src.outline(cs='latlondepth')

    rupture = {'UR': [cp[1][1], cp[1][0], cp[1][2] / 1000.],
               'UL': [cp[0][1], cp[0][0], cp[0][2] / 1000.],
               'LL': [cp[3][1], cp[3][0], cp[3][2] / 1000.],
               'LR': [cp[2][1], cp[2][0], cp[2][2] / 1000.]}

    ## get nucleation point full coordinates
    p2 = Point(rupture['UL'][0], rupture['UL'][1], rupture['UL'][2])

    nucXfac = (src.nucleation_x + 1) / 2  # convert to 0-1 range
    nucYfac = ((src.nucleation_y + 1) / 2)  # convert to 0-1 range | 0 top; 1 down

    surface_width = (src.width / 1000.) * num.cos(num.radians(src.dip))     # length of width projected to surface
    depth_range = (src.width / 1000.) * num.sin(num.radians(src.dip))       # length of width projected to vertical
    plm = p2.point_at(surface_width * nucYfac, depth_range * nucYfac, src.strike + 90)
    phypo = plm.point_at((src.length / 1000.) * nucXfac, 0, src.strike)

    hypLon = phypo.longitude
    hypLat = phypo.latitude
    hypDepth = phypo.depth

    # hypLat, hypLon = src.get_nucleation_abs_coord(cs='latlon')
    # print(hypLat, hypLon)

    ownSource = SourceClass(
            name='Synthetic',
            form='rectangular',
            lon=float(hypLon),  # hypo
            lat=float(hypLat),  # hypo
            depth=float(hypDepth),  # hypo
            magnitude=float(src.magnitude),
            nucleation_x=float(src.nucleation_x),  # (-1 = left edge, +1 = right edge)
            nucleation_y=float(src.nucleation_y),  # (-1 = upper edge, +1 = lower edge)
            strike=float(src.strike),
            dip=float(src.dip),
            rake=float(src.rake),
            rupture=rupture,
            width=float(src.width) / 1000.,
            length=float(src.length) / 1000.,
            time=src.time)

    return ownSource


# def from_rectsource_to_own_source_old(rectsource):
#     '''
#     Here the input event depth is the depth of anchor point
#     '''

#     src = copy.deepcopy(rectsource)

#     if src.anchor == 'top_left':
#         p2 = Point(src.lon, src.lat, src.depth / 1000.)
#         p1 = p2.point_at(src.length / 1000., 0., src.strike)

#         surface_width = (src.width / 1000.) * num.cos(num.radians(src.dip))     # length of width projected to surface
#         depth_range = (src.width / 1000.) * num.sin(num.radians(src.dip))       # length of width projected to vertical

#         p3 = p1.point_at(surface_width, depth_range, src.strike + 90)
#         p4 = p2.point_at(surface_width, depth_range, src.strike + 90)

#         rupture = {'UR': [p1.longitude, p1.latitude, p1.depth],
#                    'UL': [p2.longitude, p2.latitude, p2.depth],
#                    'LL': [p4.longitude, p4.latitude, p4.depth],
#                    'LR': [p3.longitude, p3.latitude, p3.depth]}

#         nucXfac = (src.nucleation_x + 1) / 2  # convert to 0-1 range
#         # nucXfac1 = 1 - nucXfac

#         nucYfac = ((src.nucleation_y + 1) / 2)  # convert to 0-1 range | 0 top; 1 down

#         plm = p2.point_at(surface_width * nucYfac, depth_range * nucYfac, src.strike + 90)
#         phypo = plm.point_at((src.length / 1000.) * nucXfac, 0, src.strike)

#         hypLon = phypo.longitude
#         hypLat = phypo.latitude
#         hypDepth = phypo.depth
#     else:
#         print('Source Anchor not defined yet. Set to center.')

#         surface_width = (src.width / 1000.) * num.cos(num.radians(src.dip))     # length of width projected to surface
#         depth_range = (src.width / 1000.) * num.sin(num.radians(src.dip))       # length of width projected to vertical
        
#         pc = Point(src.lon, src.lat, src.depth / 1000.)
#         p2 = pc.point_at(-0.5 * src.length / 1000., -0.5 * depth_range, src.strike)
#         p1 = p2.point_at(src.length / 1000., 0., src.strike)
#         p3 = p1.point_at(surface_width, depth_range, src.strike + 90)
#         p4 = p2.point_at(surface_width, depth_range, src.strike + 90)

#         rupture = {'UR': [p1.longitude, p1.latitude, p1.depth],
#                    'UL': [p2.longitude, p2.latitude, p2.depth],
#                    'LL': [p4.longitude, p4.latitude, p4.depth],
#                    'LR': [p3.longitude, p3.latitude, p3.depth]}

#         nucXfac = (src.nucleation_x + 1) / 2  # convert to 0-1 range
#         # nucXfac1 = 1 - nucXfac

#         nucYfac = ((src.nucleation_y + 1) / 2)  # convert to 0-1 range | 0 top; 1 down

#         plm = p2.point_at(surface_width * nucYfac, depth_range * nucYfac, src.strike + 90)
#         phypo = plm.point_at((src.length / 1000.) * nucXfac, 0, src.strike)

#         hypLon = phypo.longitude
#         hypLat = phypo.latitude
#         hypDepth = phypo.depth

#     ownSource = SourceClass(
#             name='Synthetic',
#             form='rectangular',
#             lon=float(hypLon),  # hypo
#             lat=float(hypLat),  # hypo
#             depth=float(hypDepth),  # hypo
#             magnitude=float(src.magnitude),
#             nucleation_x=float(src.nucleation_x),  # (-1 = left edge, +1 = right edge)
#             nucleation_y=float(src.nucleation_y),  # (-1 = upper edge, +1 = lower edge)
#             strike=float(src.strike),
#             dip=float(src.dip),
#             rake=float(src.rake),
#             rupture=rupture,
#             width=float(src.width) / 1000.,
#             length=float(src.length) / 1000.,
#             time=src.time)

#     return ownSource

# #############################
### Misc
#############################
def snapper(nmax, delta, snapfun=math.ceil):
    def snap(x):
        return max(0, min(int(snapfun(x / delta)), nmax))
    return snap


def costaper(a, b, c, d, nfreqs, deltaf):
    ### from pyrocko trace
    hi = snapper(nfreqs, deltaf)
    tap = num.zeros(nfreqs)
    tap[hi(a):hi(b)] = 0.5 \
        - 0.5 * num.cos((deltaf * num.arange(hi(a), hi(b)) - a) / (b - a) * num.pi)
    tap[hi(b):hi(c)] = 1.
    tap[hi(c):hi(d)] = 0.5 \
        + 0.5 * num.cos((deltaf * num.arange(hi(c), hi(d)) - c) / (d - c) * num.pi)

    return tap


def get_gdf_Pyrocko(ii, args, src, engine, waveform_targets, srcsDict):

    print('Starting: %s' % (ii))
    try:
        #############################
        ### Get GMs for every Source
        #############################
        response = engine.process(src, waveform_targets)
        synthTraces = response.pyrocko_traces()

        synthStaDict = create_stationdict_synthetic(synthTraces,
                        waveform_targets)
        pyrockoCont = StationContainer(refSource=src, stations=synthStaDict)
        pyrockoCont.create_all_waveforms_synth()
        pyrockoCont.resample_waveform(resample_f=20)

        # pyrockoCont.filter_waveform(freqs=filterfreq)

        pyrockoCont.get_gm_from_wv(imts=args.imts, freqs=args.freqs,
                                H2=args.rotd100, delete=True,
                                deleteWvData=True)

        gdf = pyrockoCont.to_geodataframe()

        srcsDict[ii] = gdf

        print('Finished: %s' % (ii))

    except Exception:
        traceback.print_exc()
        return (ii, traceback.format_exc())
