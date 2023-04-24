import time
import os
from multiprocessing import Pool

import numpy as num
import pandas as pd

from pyrocko import gf, util
from pyrocko import moment_tensor as pmt

import gmacc.gmeval.observation as GMobs

import gmacc.gmeval.sources as GMs
import gmacc.gmeval.util as GMu
import gmacc.dbsynth.synthetic_database as GMdb


def get_database_and_wv(args):
    # catch if settings are the same, when using append
    args.outdir = os.path.join(args.outdir, '%s_%s' % (args.sourcemode, args.mapmode))
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.outputfile = os.path.join(args.outdir, 'database.csv') 
    if os.path.exists(args.outputfile) and not args.append:
        print('File "%s" exists already.\nEither delete it or use "append" mode.' % args.outputfile)
        exit()

    if not args.append:
        pass
    else:
        if os.path.exists(args.outputfile):
            pass
        else:
            print('Output file %s does not exist.\nDo not use --append mode' % (args.outputfile))
            exit()

    #############################
    ### Processing
    #############################

    ### Regarding append-mode, knowing which sources to calculate
    nlist = num.arange(0, args.srccnt)

    if args.append:
        olddata = pd.read_csv(args.outputfile)
        li = []
        print(args.outputfile)
        print(olddata)
        for name in list(set(olddata['evID'])):
            print(name)
            nn = int(name.rsplit('_')[1])
            li.append(nn)
        li = num.array(sorted(li))
        print(li)
        nlist = num.delete(nlist, li)
        del olddata

    if args.mp > 1:
        print('Starting multiprocessing')
        p = Pool(processes=args.mp)

    for ii in nlist:
        if args.mp > 1:
            p.apply_async(calculate_wv, args=(ii, args, args.mapextent,
                                            args.mappoints, args.mapmode))
        else:
            calculate_wv(ii, args, args.mapextent, args.mappoints, args.mapmode)
    
    if args.mp > 1:
        p.close()
        p.join()


# def create_random_source(sourcemode, ii):

#     num.random.seed(seed=ii)

#     mag = num.random.randint(5000, 7501) / 1000.
#     #strike = float(num.random.randint(0., 360.))
#     #dip = float(num.random.randint(1, 89))
#     #rake = float(num.random.randint(-180, 180))
   
#     (strike, dip, rake) = pmt.random_strike_dip_rake()

#     lat = num.random.uniform(-10., 10.)
#     lon = num.random.uniform(-10., 10.)
#     depth = num.random.uniform(0.1, 20.)

#     ## MT ##
#     if sourcemode == 'MT':
#         mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mag)

#         src = gf.MTSource(lat=lat, lon=lon,
#                     depth=depth * 1000.,
#                     mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
#                     mne=mt.mne, mnd=mt.mnd, med=mt.med)
#         src.validate()

#         source = GMs.SourceClass(
#                 name='Synth',
#                 form='point',
#                 time=src.time,
#                 lon=float(lon),  # hypo
#                 lat=float(lat),  # hypo
#                 depth=float(depth),  # hypo
#                 magnitude=float(mag),
#                 strike=float(strike),
#                 dip=float(dip),
#                 rake=float(rake),
#                 tensor=dict(mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
#                     mne=mt.mne, mnd=mt.mnd, med=mt.med))

#     # ## RS ##
#     # elif sourcemode == 'RS':
#     #     ## depth corresponds to top of rupture depth/depth of anchor point

#     #     # nucx = num.random.uniform(-1, 1)  # (-1 = left edge, +1 = right edge)
#     #     # nucy = num.random.uniform(-1, 1)  # (-1 = upper edge, +1 = lower edge)

#     #     # if mag < 6.:
#     #     #     nucfac = 3
#     #     # elif mag < 7:
#     #     #     nucfac = 5
#     #     # else:
#     #     #     nucfac = 7
#     #     nucfac = 9

#     #     anchor = 'top_left'
#     #     nucx = float(num.random.choice(num.linspace(-1, 1, nucfac)))
#     #     # (-1 = left edge, +1 = right edge)
        
#     #     nucy = float(num.random.choice(num.linspace(-1, 1, nucfac))) 
#     #     # (-1 = upper edge, +1 = lower edge)
        
#     #     width, length = GMu.calc_source_width_length(mag, mode='Blaser', rake=rake)

#     #     src = gf.RectangularSource(time=util.str_to_time('1995-01-29 13:00:00.0'),
#     #                         lat=lat, lon=lon, depth=depth * 1000., anchor=anchor,
#     #                         strike=strike, dip=dip, rake=rake,
#     #                         width=width * 1000., length=length * 1000.,
#     #                         nucleation_x=nucx,
#     #                         nucleation_y=nucy,
#     #                         decimation_factor=1,
#     #                         magnitude=mag)
#     #     src.validate()
#     #     source = GMs.from_rectsource_to_own_source(src)
#     #     source.create_rupture_surface()
#     else:
#         print('Wrong mode')
#         exit()

#     source.update(name='%s_%s' % (source.name, ii))
#     source.validate()

#     return source


def calculate_wv(ii, args, mapextent, ncords, mapping):
    wvdir = 'synth_wv'

    reftime = time.time()
    source = GMdb.create_random_source(args.sourcemode, ii=ii)

    print('Starting %s; Mag: %0.1f, Depth: %0.2f, NucX: %0.2f, NucY %0.2f'
        % (ii, source.magnitude, source.depth,
        source.nucleation_x, source.nucleation_y))

    ### Generate Map points/coords
    coordinates = []
    if mapping == 'random':
        mapCoords = GMu.quasirandom_mapping(source, mapextent, ncords, rmin=0.)
    elif mapping == 'rectangular':
        mapCoords = GMu.rectangular_mapping(source, mapextent,
                                            ncords, rmin=0.)
    elif mapping == 'circular':
        mapCoords = GMu.circular_mapping(source, mapextent,
                                        ncords, rmin=0.05)
    elif mapping == 'random_circular':
        mapCoords = GMu.random_circular_mapping(source, mapextent,
                                                ncords, rmin=0.05)
    elif mapping == 'downsampling':
        mapCoords = GMu.downsampling_mapping(source,
            mapextent=mapextent, ncoords=ncords, log=True)
    elif mapping == 'mixed':
        rng = num.random.rand(1)[0]
        threshold = 0.85
        if rng > threshold:
            mapCoords = GMu.quasirandom_mapping(source, mapextent=mapextent, ncoords=ncords)
        elif rng < threshold:
            mapCoords = GMu.random_circular_mapping(source, mapextent=mapextent, rmin=0.05, ncoords=ncords, log=True)
        else:
            print('rnd doesnt not func')
            exit()
    
    else:
        print('Wrong mapping: %s' % (mapping))
        exit()

    coordinates.append(mapCoords)
    coords = coordinates[0]

    #############################
    ### Calculation of Synthetic WV with Pyrocko
    #############################
    staContainer = GMobs.get_pyrocko_container(source, coords, args.comps,
                    timecut=args.timecut, gfpath=args.gf,
                    only_waveform=True,
                    savepath=os.path.join(args.outdir, wvdir, source.name))

    #####
    ## Results
    #####
    staContainer.calc_distances()
    staContainer.calc_azimuths()
    if args.sourcemode == 'RS':
        staContainer.calc_rupture_azimuths()

    ############
    ## Print to file

    ### alternative approach (maybe faster/better?)
    # st_gdf = stationCont.to_geodataframe()
    # py_gdf = pyrockoCont.to_geodataframe()

    # cols = py_gdf.columns
    # allGDF = st_gdf.copy(deep=True)

    # for col in cols:
    #     if col in ['geometry', 'st_lat', 'st_lon']:
    #         continue
    #     else:
    #         allGDF[str(col) + '_lowfr'] = py_gdf[col]

    resultDictraw = {
        'evID': source.name,
        'magnitude': '%0.3f' % source.magnitude,
        'ev_lat': '%0.3f' % source.lat,
        'ev_lon': '%0.3f' % source.lon,
        'ev_depth': '%0.3f' % source.depth,
        'strike': '%0.3f' % source.strike,
        'dip': '%0.3f' % source.dip,
        'rake': '%0.3f' % source.rake,
        'src_duration': '%0.3f' % source.duration,
    }

    if args.sourcemode == 'RS':
        resultDictraw['width'] = '%0.3f' % source.width
        resultDictraw['length'] = '%0.3f' % source.length
        resultDictraw['nucleation_x'] = '%0.3f' % source.nucleation_x
        resultDictraw['nucleation_y'] = '%0.3f' % source.nucleation_y
    
    for ns, station in staContainer.stations.items():
        resultDict = resultDictraw.copy()

        resultDict['ns'] = '%s.%s' % (station.network, station.station)
        resultDict['st_lat'] = '%0.3f' % station.lat
        resultDict['st_lon'] = '%0.3f' % station.lon
        resultDict['azimuth'] = '%0.3f' % station.azimuth
        resultDict['rhypo'] = '%0.3f' % (station.rhypo)
        if args.sourcemode == 'RS':
            resultDict['rjb'] = '%0.3f' % (station.rjb)
            resultDict['ry0'] = '%0.3f' % (station.ry0)
            resultDict['rx'] = '%0.3f' % (station.rx)
            resultDict['rrup'] = '%0.3f' % (station.rrup)
            resultDict['rup_azimuth'] = '%0.3f' % (station.rup_azimuth)
            resultDict['centre_azimuth'] = '%0.3f' % (station.centre_azimuth)

        resultDict['waveform_path'] = os.path.join(wvdir, '%s.mseed' % (source.name))

        if not os.path.exists(args.outputfile):
            with open(args.outputfile, "w") as file:
                header = ''
                for nn, key in enumerate(resultDict.keys()):
                    if nn >= len(resultDict.keys()) - 1:
                        header += '%s\n' % key
                    else:
                        header += '%s,' % key

                file.write(header)

        with open(args.outputfile, "a") as file:
            line = ''
            for nn, val in enumerate(resultDict.values()):
                if nn >= len(resultDict.values()) - 1:
                    line += '%s\n' % val
                else:
                    line += '%s,' % val

            file.write(line)

    print('Finished %s in %0.4fs' % (ii, time.time() - reftime))
