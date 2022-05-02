import numpy as num
import random

from pyrocko import orthodrome
from pyrocko import moment_tensor as pmt

#### 
# Read in ensemble file
####


###############
# smth else
###############
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
        if hasattr(source, 'rupture_velocity') and source.rupture_velocity is not None and source.rupture_velocity not in [-99.0, 0.0, 999.0]:
        # if source.rupture_velocity is not None and source.rupture_velocity not in [-99.0, 0.0, 999.0]:
            vr = source.rupture_velocity
        elif hasattr(source, 'velocity'):
            vr = source.velocity
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

#############################
'''
Mapping functions

Create grid coordinates around the (Hypo)center of the given source.
'''
#############################
def quasirandom_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.0):
    import chaospy as cpy

    lats, lons = cpy.create_halton_samples(ncoords**2, 2)

    relfac = 0.01
    lons += num.random.uniform(-relfac * mapextent[0],
                               relfac * mapextent[0], len(lons))
    lats += num.random.uniform(-relfac * mapextent[1],
                               relfac * mapextent[1], len(lats))

    lons = lons * 2 * mapextent[0] - mapextent[0] + source.lon
    lats = lats * 2 * mapextent[1] - mapextent[1] + source.lat

    prefacLat = (2 * mapextent[1]) / (max(lats) - min(lats))
    lats = (prefacLat * (lats - num.mean(lats))) + num.mean([source.lat - mapextent[1],
                        source.lat + mapextent[1]])

    prefacLon = (2 * mapextent[0]) / (max(lons) - min(lons))
    lons = (prefacLon * (lons - num.mean(lons))) + num.mean([source.lon - mapextent[0],
                        source.lon + mapextent[0]])

    coords = []
    for lon, lat in zip(lons, lats):
        dist = orthodrome.distance_accurate50m_numpy(
                            source.lat, source.lon, lat, lon) / 1000.
        if abs(dist) > rmin:
            coords.append([lon, lat])
        else:
            pass

    coords = num.array(coords)
    return coords


def random_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.0):
    coords = []

    lats = num.random.uniform(source.lat - mapextent[1],
                        source.lat + mapextent[1], ncoords**2)
    lons = num.random.uniform(source.lon - mapextent[0],
                        source.lon + mapextent[0], ncoords**2)

    for lon, lat in zip(lons, lats):
        dist = orthodrome.distance_accurate50m_numpy(
                            source.lat, source.lon, lat, lon) / 1000.
        if abs(dist) > rmin:
            coords.append([lon, lat])
        else:
            pass

    coords = num.array(coords)
    return coords


def rectangular_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.0):
    coords = []

    lats = num.linspace(source.lat - mapextent[1],
                        source.lat + mapextent[1], ncoords)
    lons = num.linspace(source.lon - mapextent[0],
                        source.lon + mapextent[0], ncoords)

    for lon in lons:
        for lat in lats:
            dist = orthodrome.distance_accurate50m_numpy(
                                source.lat, source.lon, lat, lon) / 1000.
            if abs(dist) > rmin:
                coords.append([lon, lat])
            else:
                pass

    coords = num.array(coords)
    return coords


def circular_mapping(source, mapextent=[1, 1], ncoords=10, rmin=0.05):
    coords = []
    dcor = min(mapextent[1], mapextent[0])

    r = num.logspace(num.log10(rmin), num.log10(dcor), int(ncoords / 1.25))
    theta = num.linspace(0, 2 * num.pi, int(ncoords * 1.25)) \
        + random.random() * 2 * num.pi

    R, Theta = num.meshgrid(r, theta)
    lons = R * num.cos(Theta) + source.lon
    lats = R * num.sin(Theta) + source.lat

    for lon, lat in zip(lons.flatten(), lats.flatten()):
        coords.append([lon, lat])

    coords = num.array(coords)

    return coords
