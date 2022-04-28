import sys
import time
import os

import numpy as num
import pandas as pd

from pyrocko import gf, util
from pyrocko import moment_tensor as pmt

# from openquake.hazardlib.geo import geodetic

import ewrica.gm.sources as GMs
import ewrica.gm.util as GMu
# import mole.gm.misc as GMm
# import mole_ext.openquake as GMopenquake

from multiprocessing import Pool


sys.path.insert(1, '../gm_test/')
import observation as GMobs


def create(args):

    # catch if settings are the same, when using append

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    args.outputfile = args.outdir + '/%s_database.csv' % (args.sourcemode)
    if os.path.exists(args.outputfile) and not args.append:
        print('File "%s" exists already.\nEither delete it or use "append" mode.' % args.outputfile)
        exit()

    if not args.append:
        if args.rotd100 and ('E' in args.comps and 'N' in args.comps):
            pass
        elif args.rotd100:
            print('rotd100 not possible due to E and/or N not in comps list.')
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
            p.apply_async(extract_gm, args=(ii, args, args.mapextent,
                                            args.mappoints, args.mapmode))
        else:
            extract_gm(ii, args, args.mapextent, args.mappoints, args.mapmode)
    
    if args.mp > 1:
        p.close()
        p.join()


def extract_gm(ii, args, mapextent, ncords, mapping):

    reftime = time.time()
    num.random.seed(seed=ii)

    mag = num.random.randint(5000, 7501) / 1000.
    strike = float(num.random.randint(0., 360.))
    dip = float(num.random.randint(1, 89))
    rake = float(num.random.randint(-180, 180))
    
    lat = num.random.uniform(-10., 10.)
    lon = num.random.uniform(-10., 10.)
    depth = num.random.uniform(0.1, 10.)

    ## MT ##
    if args.sourcemode == 'MT':
        mt = pmt.MomentTensor(strike=strike, dip=dip, rake=rake, magnitude=mag)

        src = gf.MTSource(lat=lat, lon=lon,
                    depth=depth * 1000.,
                    mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
                    mne=mt.mne, mnd=mt.mnd, med=mt.med)
        src.validate()

        source = GMs.SourceClass(
                name='Synthetic',
                form='point',
                time=src.time,
                lon=float(lon),  # hypo
                lat=float(lat),  # hypo
                depth=float(depth),  # hypo
                magnitude=float(mag),
                strike=float(strike),
                dip=float(dip),
                rake=float(rake),
                tensor=dict(mnn=mt.mnn, mee=mt.mee, mdd=mt.mdd,
                    mne=mt.mne, mnd=mt.mnd, med=mt.med))

        # print(GMs.from_mtsource_to_own_source(src))

    ## RS ##
    elif args.sourcemode == 'RS':
        ## depth corresponds to top of rupture depth/depth of anchor point

        # nucx = num.random.uniform(-1, 1)  # (-1 = left edge, +1 = right edge)
        # nucy = num.random.uniform(-1, 1)  # (-1 = upper edge, +1 = lower edge)

        # if mag < 6.:
        #     nucfac = 3
        # elif mag < 7:
        #     nucfac = 5
        # else:
        #     nucfac = 7
        nucfac = 9

        anchor = 'top_left'
        nucx = float(num.random.choice(num.linspace(-1, 1, nucfac)))
        # (-1 = left edge, +1 = right edge)
        
        nucy = float(num.random.choice(num.linspace(-1, 1, nucfac))) 
        # (-1 = upper edge, +1 = lower edge)
        
        width, length = GMu.calc_source_width_length(mag, mode='Blaser', rake=rake)

        src = gf.RectangularSource(time=util.str_to_time('1995-01-29 13:00:00.0'),
                            lat=lat, lon=lon, depth=depth * 1000., anchor=anchor,
                            strike=strike, dip=dip, rake=rake,
                            width=width * 1000., length=length * 1000.,
                            nucleation_x=nucx,
                            nucleation_y=nucy,
                            decimation_factor=1,
                            magnitude=mag)
        src.validate()
        source = GMs.from_rectsource_to_own_source(src)
        source.create_rupture_surface()
    else:
        print('Wrong mode')
        exit()

    source.update(name='%s_%s' % (source.name, ii))
    source.validate()
    # print(source)

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
                                        ncords, rmin=0.1)
    else:
        print('Wrong mapping: %s' % (mapping))
        exit()

    coordinates.append(mapCoords)
    coords = coordinates[0]

    #############################
    ### Calculation of Synthetic WV with Pyrocko
    #############################

    staContainer = GMobs.get_pyrocko_container(source, coords, args.comps,
                    args.imts, args.freqs, resample_f=20,
                    deleteWvData=True, H2=args.rotd100, gfpath=args.gf,
                    savepath='%s/%s_synthetic_waveforms/%s' % (args.outdir, args.sourcemode, source.name))

    #############################
    ### Calculation of GMPE with openquake
    #############################
    if args.gmpes:
        print('GMPEs currently not implemented')
        exit()
        # gmpes = []
        # gmpe_list = get_available_gsims()

        # for g in args.gmpes:
        #     gmpes.append(gmpe_list[g]())

        # for gmpe in gmpes:
        #     # Each GMPE has the following information
        #     print()
        #     print(str(gmpe))
        #     print(gmpe.REQUIRES_RUPTURE_PARAMETERS)
        #     print(gmpe.REQUIRES_DISTANCES)
        #     print(gmpe.REQUIRES_SITES_PARAMETERS)
        #     print()

        # args.chaCodes = gmpes

        # vs30 = num.zeros(len(mapCoords)) + 800.
        # rctx, sctx, dctx = GMopenquake.gmpe_contexts(source, coords, vs30)

        # staContainer = GMs.StationContainer(
        #                     refSource=source,
        #                     stations={})

        # for gmpe in args.chaCodes:
        #     staContainer = GMopenquake.gmpe_gm(gmpe, staContainer,
        #                                         rctx, sctx, dctx,
        #                                         args.imts, args.sdofs,
        #                                         mapCoords)

    #####
    ## Results
    #####

    staContainer.calc_distances()
    staContainer.calc_azimuths()
    if args.sourcemode == 'RS':
        staContainer.calc_rupture_azimuths()

    comps = args.comps.copy()
    if args.rotd100 and ('E' in args.comps and 'N' in args.comps):
        comps.append('H')
    elif args.rotd100:
        print('rotd100 not possible due to E and/or N not in comps list.')

    ############
    ## Print to file

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

        resultDict['waveform_path'] = 'synthetic_waveforms/%s.mseed' % (source.name)

        for mm, comp in enumerate(comps):

            if args.gmpes:
                comp = str(comp).replace('[', '').replace(']', '')

            for nn, im in enumerate(args.imts):
                resultDict['%s_%s' % (comp, im)] = '%0.3f' \
                    % (station.components[comp].gms[im].value)

            for nn, im in enumerate(args.freqs):
                im = 'f_%s' % (im)
                resultDict['%s_%s' % (comp, im)] = '%0.3f' \
                    % (station.components[comp].gms[im].value)

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