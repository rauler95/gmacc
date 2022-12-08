import json
import os

import numpy as num

from pyrocko import orthodrome, io
from pyrocko import moment_tensor as pmt

from openquake.hazardlib.geo import Point, Mesh, PlanarSurface,\
                                    GriddedSurface

from gmacc.gmeval.sources import SourceClass

###
# Inventory
###
def get_station_data(path):
    if os.path.isdir(path):
        locationDict = create_locationDict_from_invdir(path)
        print('Read from inventory directory')

    elif os.path.isfile(path):
        try:
            locationDict = create_locationDict_from_xml(path)
            print('Read from inventory xml')

        except:
            try:
                locationDict = create_locationDict(path)
                print('Read from location txt')
            except:
                print('Station file %s could not be load.' % path)

    else:
        print('Either directory or file  \'%s\' does not exist.' % path)
        exit()

    return locationDict


def create_locationDict(fileLocation):
    locDict = {}
    with open(fileLocation, 'r') as f:
        for line in f:
            line = line.rsplit()

            if len(line) == 1:
                line = line[0].rsplit(',')

            ns = line[0].replace(',', '')
            lon = float(line[1].replace(',', ''))
            lat = float(line[2].replace(',', ''))
            locDict[ns] = [lon, lat]

    return locDict


def create_locationDict_from_invdir(directory):
    from obspy import read_inventory
    
    locDict = {}
    for file in os.listdir(directory):
        fileLocation = directory + '/' + file
        try:
            inv = read_inventory(fileLocation)
        except AttributeError as e:
            print(e)
            print(file)
            print()
        for net in inv:
            for sta in net:
                ns = '%s.%s' % (net.code, sta.code)
                lon = float(sta.longitude)
                lat = float(sta.latitude)
                locDict[ns] = [lon, lat]

    return locDict


def create_locationDict_from_xml(fileLocation):
    from obspy import read_inventory

    locDict = {}

    inv = read_inventory(fileLocation)
    for net in inv:
        for sta in net:
            ns = '%s.%s' % (net.code, sta.code)
            lon = float(sta.longitude)
            lat = float(sta.latitude)
            locDict[ns] = [lon, lat]

    return locDict

###
# Waveform
###
def get_waveform_data(path):
    if os.path.isdir(path):
        print(path)

        waveformdata = []
        for ii, file in enumerate(os.listdir(path)):
            # print(ii, path, file)
            wvfile = os.path.join(path, file)
            if os.path.exists(wvfile) and not os.stat(wvfile).st_size < 1:
                waveformdata.append(*io.load(wvfile))

    elif os.path.isfile(path):
        if os.path.exists(path) and not os.stat(path).st_size < 1:
            waveformdata = io.load(path)
        else:
            print('No MSEED data available or empty')
            exit()

    else:
        print('Path %s is not existing.' % path)
        exit()

    return waveformdata

###
# Event information
###


def get_event_data(file):
    if os.path.exists(file) and not os.stat(file).st_size < 1:
        pass
    else:
        print('File \'%s\' is either empty or does not exists.' % file)
        return None

    source = convert_quakeml_to_source(file)

    if hasattr(source, 'moment'):
        if source.moment:
            newmag = float(pmt.moment_to_magnitude(source['moment']))
            if abs(source.magnitude - newmag) > 0.5:
                print('Moment wrong, newmag would be: %0.2f; oldmag %s' % (newmag, source.magnitude))
                print(source.moment)
                source.moment = None

            else:
                # print(source.magnitude, pmt.moment_to_magnitude(source['moment']))
                source.magnitude = newmag 

    return source


def get_finte_fault_data(faultFile, source):
    if os.path.exists(faultFile) and not os.stat(faultFile).st_size < 1:
        pass
    else:
        print('File \'%s\' is either empty or does not exists.' % faultFile)
        exit()

        source = read_fsp(faultFile, source)

    try:
        source = read_fsp(faultFile, source)
    except:
        try:
            corners = read_esm_faultfile(faultFile)
            source.rupture = corners
            surface = corners2surface(source, corners)
            if surface:
                source = convert_surface(source, surface)
            print('Fault from ESM')
        except:
            try:
                corners = read_usgs_faultfile(faultFile)
                source.rupture = corners
                surface = corners2surface(source, corners)
                if surface:
                    source = convert_surface(source, surface)
                print('Fault from USGS')
            except:
                try:
                    corners = read_json_faultfile(faultFile, source)
                    print('Fault from .json')
                except:
                    print('Could not load %s file.' % faultFile)
                    exit()

    source.create_rupture_surface()
    if source.risetime is None or source.risetime < 1.:
        source.risetime = 1.

    return source

###
# Load external source files
###
def json_to_source(eventFile):

    with open(eventFile) as f:
        ev = json.load(f)

        evpro = ev['properties']

        if len(str(evpro['time'])) < 12:
            time = float(evpro['time'])
        else:
            time = float(evpro['time']) / 1000.

        source = SourceClass(
            name=str(ev['id']),
            lon=float(ev['geometry']['coordinates'][0]),
            lat=float(ev['geometry']['coordinates'][1]),
            depth=abs(float(ev['geometry']['coordinates'][2])),
            magnitude=float(evpro['mag']),
            time=time,
            form='point')

        if 'focal-mechanism' in evpro['products']:
            fms = evpro['products']['focal-mechanism']
            reftime = 0
            for nn, ifm in enumerate(fms):
                utime = ifm['updateTime']
                # or use preferredWeight
                # or evaluation-status == 'reviewed'
                if utime > reftime:
                    idx = nn

            FM = fms[idx]['properties']
            source.strike = float(FM['nodal-plane-1-strike'])
            source.dip = float(FM['nodal-plane-1-dip'])
            source.rake = float(FM['nodal-plane-1-rake'])
            # print(strike, dip, rake)

        if "moment-tensor" in evpro['products']:
            mts = evpro['products']["moment-tensor"]

            reftime = 0
            for nn, imt in enumerate(mts):
                utime = float(imt['updateTime'])
                if utime > reftime:
                    reftime = utime
                    idx = nn

            MT = mts[idx]['properties']
            try:
                source.strike = float(MT['nodal-plane-1-strike'])
                source.dip = float(MT['nodal-plane-1-dip'])
                source.rake = float(MT['nodal-plane-1-rake'])

                # source.strike = float(MT['nodal-plane-2-strike'])
                # source.dip = float(MT['nodal-plane-2-dip'])
                # source.rake = float(MT['nodal-plane-2-rake'])
            except KeyError:
                pass

            # print(source.strike, source.dip, source.rake)
            # exit()

            source.moment = float(MT['scalar-moment'])
            source.stf = {}
            try:
                source.stf['duration'] = float(MT['sourcetime-duration'])
            except KeyError:
                pass

            source.tensor = {'mdd': float(MT["tensor-mrr"]),
                    'mnn': float(MT["tensor-mtt"]),
                    'mee': float(MT["tensor-mpp"]),
                    'mnd': float(MT["tensor-mrt"]),
                    'med': -float(MT["tensor-mrp"]),
                    'mne': -float(MT["tensor-mtp"])}

    return source


def obspy_to_source(eventFile):
    from obspy.core.event import read_events
    from obspy.core.utcdatetime import UTCDateTime
    cat = read_events(eventFile)
    ev = cat[0]

    prefOr = ev.preferred_origin()
    if prefOr.longitude:
        pass
    else:
        for org in ev.origins:
            if org.evaluation_mode == 'manual':
                prefOr = org
                break
            else:
                prefOr = org

    mag = ev.preferred_magnitude()
    if mag:
        pass
    else:
        for evmag in ev.magnitudes:
            if evmag.evaluation_mode == 'manual':
                if evmag.magnitude_type in ['Mw', 'mw']:
                    mag = evmag
                    break
                else:
                    mag = evmag

    time = UTCDateTime.strftime(prefOr.time, format='%Y-%m-%d %H:%M:%S')
    tmin = util.str_to_time(time, format='%Y-%m-%d %H:%M:%S')

    source = SourceClass(
        name=str(ev.resource_id),
        lon=float(prefOr.longitude),
        lat=float(prefOr.latitude),
        depth=float(prefOr.depth) / 1000.,
        magnitude=float(mag.mag),
        time=tmin,
        form='point')

    prefFM = ev.preferred_focal_mechanism()

    if prefFM:
        fm = prefFM
    else:
        fm = None

    if (hasattr(fm, 'moment_tensor')
            and hasattr(fm.moment_tensor, 'tensor')
            and fm.moment_tensor.tensor is not None)\
            or (hasattr(fm, 'nodal_planes')
                and fm.nodal_planes is not None
                and hasattr(fm.nodal_planes, 'nodal_plane_1')
                and fm.nodal_planes.nodal_plane_1.strike is not None):
        FM = prefFM
        # print(fm.nodal_planes)
        # print(hasattr(fm.nodal_planes, 'nodal_plane_1'))
        # print(fm.nodal_planes.nodal_plane_1.strike)
        print('Using preferred FM')

    else:
        if len(ev.focal_mechanisms) < 1:
            FM = None
        FMmt = None
        FMnd = None

        for fm in ev.focal_mechanisms:
            if hasattr(fm, 'moment_tensor')\
                    and hasattr(fm.moment_tensor, 'tensor')\
                    and fm.moment_tensor.tensor is not None\
                    and hasattr(fm, 'nodal_planes')\
                    and fm.nodal_planes is not None\
                    and hasattr(fm.nodal_planes, 'nodal_plane_1')\
                    and fm.nodal_planes.nodal_plane_1.strike is not None:
                FM = fm
                # print('Both')
                break

            elif hasattr(fm, 'moment_tensor')\
                    and hasattr(fm.moment_tensor, 'tensor')\
                    and fm.moment_tensor.tensor is not None:
                FMmt = fm
                # print('MT')
                # break
            elif hasattr(fm, 'nodal_planes') and fm.nodal_planes is not None\
                    and hasattr(fm.nodal_planes, 'nodal_plane_1') and\
                    fm.nodal_planes.nodal_plane_1.strike is not None:
                FMnd = fm
                # print('ND')
            else:
                pass

            # print()
            if FMmt is not None:
                FM = FMmt
            elif FMnd is not None:
                FM = FMnd

    if FM:
        mt = FM.moment_tensor
        if mt:
            source.moment = mt.scalar_moment
            source.stf = mt.source_time_function
            tn = mt.tensor
            if tn:
                source.tensor = {'mdd': tn.m_rr,
                        'mnn': tn.m_tt,
                        'mee': tn.m_pp,
                        'mnd': tn.m_rt,
                        'med': -tn.m_rp,
                        'mne': -tn.m_tp}
        try:
            nd1 = FM.nodal_planes.nodal_plane_1
            source.strike = float(nd1.strike)
            source.dip = float(nd1.dip)
            source.rake = float(nd1.rake)
        except (AttributeError, TypeError) as e:
            print('No Strike in FM;', e)

    return source


def pyrocko2_to_source(eventFile):
    from pyrocko.io import quakeml
    qml = quakeml.QuakeML.load_xml(filename=eventFile)
    # print(qml.event_parameters.event_list[0].focal_mechanism_list)
    # exit()
    evs = qml.get_pyrocko_events()
    if len(evs) > 1:
        print('WARNING event.xml includes more than 1 EVENTS')
        exit()
    ev = evs[0]

    source = SourceClass(
        name=str(ev.name),
        lon=float(ev.lon),
        lat=float(ev.lat),
        depth=float(ev.depth) / 1000.,
        magnitude=float(ev.magnitude),
        time=ev.time,
        form='point')

    mt = ev.moment_tensor
    if mt:
        source.moment = mt.moment
        try:
            source.stf = mt.source_time_function
        except AttributeError:
            print('No STF found')
            pass
        source.tensor = {'mdd': mt.mdd,
                    'mnn': mt.mnn,
                    'mee': mt.mee,
                    'mnd': mt.mnd,
                    'med': mt.med,
                    'mne': mt.mne}

        source.strike = float(mt.strike1)
        source.dip = float(mt.dip1)
        source.rake = float(mt.rake1)

    if not hasattr(source, 'stf'):
        qml = quakeml.QuakeML.load_xml(filename=eventFile)
        ev = qml.event_parameters.event_list[0]
        prefMagID = ev.preferred_magnitude_id
        prefFM = False
        for fm in ev.focal_mechanism_list:
            if prefMagID.rsplit('/')[-2:] == fm.public_id.rsplit('/')[-2:]:
                prefFM = fm
                break
            else:
                prefFM = fm

        if prefFM:
            if hasattr(prefFM, 'moment_tensor_list'):
                for mt in prefFM.moment_tensor_list:
                    source.moment = mt.scalar_moment.value
                    source.stf = mt.source_time_function

    return source


def pyrocko_to_source(eventFile):
    from pyrocko.io import quakeml
    qml = quakeml.QuakeML.load_xml(filename=eventFile)
    ev = qml.event_parameters.event_list[0]
    if len(qml.event_parameters.event_list) > 1:
        print('WARNING event.xml includes more than 1 EVENTS')
        exit()

    prefOrID = ev.preferred_origin_id
    prefMagID = ev.preferred_magnitude_id
    prefFMID = ev.preferred_focal_mechanism_id

    for origin in ev.origin_list:
        if origin.public_id == prefOrID:
            prefOr = origin

    for mag in ev.magnitude_list:
        if mag.public_id != prefMagID:
            continue
        magnitude = mag.mag.value

    source = SourceClass(
        name=str(str(ev.public_id).rsplit('=')[-1]),
        lon=float(prefOr.longitude.value),
        lat=float(prefOr.latitude.value),
        depth=float(prefOr.depth.value) / 1000.,
        magnitude=float(magnitude),
        time=prefOr.time.value,
        form='point')

    if not hasattr(ev, 'focal_mechanism_list'):
        GMm.pwarning('No FM found.')
        return source

    if len(ev.focal_mechanism_list) < 1:
        GMm.pwarning('No FM found.')
        return source

    prefFM = False

    for fm in ev.focal_mechanism_list:
        # print(fm)
        if prefFMID:
            GMm.psuccess('\n\nFM ID FOUND!!! NEEDS TO BE IMPLEMENTED\n\n')
            GMm.pwarning('\n\nFM ID FOUND!!! NEEDS TO BE IMPLEMENTED\n\n')
            GMm.pfail('\n\nFM ID FOUND!!! NEEDS TO BE IMPLEMENTED\n\n')
            # print(fm)
            # print(fm.public_id)
            # print(fm.triggering_origin_id)
            # print(prefOrID)
            # print(prefMagID)
            # print(fm.method_id)
            # print(prefFMID)
            # print()
        else:
            if prefMagID.rsplit('/')[-2:] == fm.public_id.rsplit('/')[-2:]:
                prefFM = fm
                break
            else:
                if fm.triggering_origin_id == prefOrID:
                    prefFM = fm
                else:
                    prefFM = fm

    if prefFM:
        if hasattr(prefFM, 'moment_tensor_list'):
            for mt in prefFM.moment_tensor_list:
                source.moment = mt.scalar_moment.value
                source.stf = mt.source_time_function

                tn = mt.tensor
                try:
                    source.tensor = {'mdd': tn.mrr.value,
                            'mnn': tn.mtt.value,
                            'mee': tn.mpp.value,
                            'mnd': tn.mrt.value,
                            'med': -tn.mrp.value,
                            'mne': -tn.mtp.value}
                except AttributeError:
                    pass
        else:
            print('None MT found')

        nd1 = prefFM.nodal_planes.nodal_plane1
        source.strike = float(nd1.strike.value)
        source.dip = float(nd1.dip.value)
        source.rake = float(nd1.rake.value)
        nd2 = prefFM.nodal_planes.nodal_plane2
        source.strike = float(nd2.strike.value)
        source.dip = float(nd2.dip.value)
        source.rake = float(nd2.rake.value)

    return source


def convert_quakeml_to_source(eventFile):

    # source = pyrocko_to_source(eventFile)
    # source = pyrocko2_to_source(eventFile)
    # source = obspy_to_source(eventFile)
    # exit()
    if '.json' in eventFile:
        source = json_to_source(eventFile)
        flag = 'JSON'
    else:
        try:
            source = pyrocko_to_source(eventFile)
            flag = 'Pyrocko_own'
        except:
            try:
                source = pyrocko2_to_source(eventFile)
                flag = 'Pyrocko2'

            # except (guts.ValidationError, guts.ArgumentError) as e:
            #     print('Pyrocko in-read error:', e)
            except:
                try:
                    source = obspy_to_source(eventFile)
                    flag = 'Obspy'
                except:
                    print('Not possible to read-in %s' % (eventFile.rsplit('/')[-1]))
                    # exit()
                    flag = ''
                    source = None

    if flag != '':
        print('Read source with %s' % flag)
    return source


def read_fsp(faultFile, source):
    flons = []
    flats = []
    with open(faultFile) as f:
        for line in f:
            line = line.rsplit()
            if len(line) == 1:
                continue

            if line == []:
                continue

            if line[1] == 'Loc':
                lat = float(line[5])
                lon = float(line[8])
                depth = line[11]

            if line[1] == 'Size':
                LN = line[5]
                WD = line[9]
                magnitude = line[13]
                moment = line[16]

            if line[1] == 'Mech':
                strike = line[5]
                dip = line[8]
                rake = line[11]
                ztor = line[14]

            if line[1] == 'Rupt':
                hypx = float(line[5])
                hypy = float(line[9])
                tr = line[13]
                vr = line[17]

            # if line[1] == 'Nsbfs':
            #   flag = 'newsegment'

            # # if line[0] != '%' and flag == 'newsegment':
            # #     if pdepth < float(line[4]):
            # #         continue
            # #     plat = float(line[0])
            # #     plon = float(line[1])
            # #     pdepth = float(line[4])
            # #     flag = None
            if line[0] not in ['%']:
                flons.append(float(line[0]))
                flats.append(float(line[1]))

    # print(flons)
    # print(flats)
    # print(min(flons), max(flons))
    # print(min(flats), max(flats))
    # print

    # exit()

    ## starting top left than going along strike than down-dip
    source.lat = float(lat)
    source.lon = float(lon)
    source.depth = float(depth)
    source.ztor = float(ztor)

    source.length = float(LN)
    source.width = float(WD)
    source.magnitude = float(magnitude)
    source.moment = float(moment)

    source.strike = float(strike)
    source.dip = float(dip)

    if float(rake) in [999.0, -99.0]:
        if source.rake != 0.:
            pass
        else:
            source.rake = 0.
    else:
        source.rake = float(rake)

    source.risetime = float(tr)
    source.rupture_velocity = float(vr)

    ## with the assumption hypx and hypy are from top-left
    source.nucleation_x = ((hypx / float(LN)) * 2) - 1
    source.nucleation_y = ((hypy / float(WD)) * 2) - 1

    ### calc rectangular points
    pHypo = Point(longitude=source.lon, latitude=source.lat, depth=source.depth)
    pTMP = pHypo.point_at(
            hypx,
            0,
            source.strike + 180.)
    pUL = pTMP.point_at(
            hypy * num.cos(num.radians(source.dip)),
            - hypy * num.sin(num.radians(source.dip)),
            source.strike - 90)
    pUR = pUL.point_at(source.length, 0, source.strike)
    pLR = pUR.point_at(source.width * num.cos(num.radians(source.dip)),
                        source.width * num.sin(num.radians(source.dip)),
                        source.strike + 90.)
    pLL = pUL.point_at(source.width * num.cos(num.radians(source.dip)),
                        source.width * num.sin(num.radians(source.dip)),
                        source.strike + 90.)

    corners = {'UL': [pUL.longitude, pUL.latitude, pUL.depth],
                'UR': [pUR.longitude, pUR.latitude, pUR.depth],
                'LR': [pLR.longitude, pLR.latitude, pLR.depth],
                'LL': [pLL.longitude, pLL.latitude, pLL.depth]}

    source.rupture = corners
    source.form = 'rectangular'

    return source


def read_esm_faultfile(faultFile):
    plainDict = {}
    with open(faultFile) as f:
        for ii, line in enumerate(f):
            line = line.rsplit()
            if ii == 1:
                keyname = 'UL'
            elif ii == 2:
                keyname = 'UR'
            elif ii == 3:
                keyname = 'LR'
            elif ii == 4:
                keyname = 'LL'
            else:
                continue

            plainDict[keyname] = [float(line[1]), float(line[0]), float(line[2])]

    return plainDict


def read_usgs_faultfile(faultFile):
    plainDict = {}
    with open(faultFile) as f:
        for ii, line in enumerate(f):
            line = line.rsplit()
            if ii == 0:
                keyname = 'UL'
            elif ii == 1:
                keyname = 'UR'
            elif ii == 2:
                keyname = 'LR'
            elif ii == 3:
                keyname = 'LL'
            else:
                continue

            plainDict[keyname] = [float(line[0]), float(line[1]), float(line[2])]

    return plainDict


def read_usgs_finitefaultfile(faultFile):
    plainDict = {}
    print(faultFile)
    with open(faultFile) as f:
        ff = json.load(f)

    ps = []
    lons = []
    lats = []
    depths = []
    minlat = 0
    maxlat = 0
    minlon = 0
    maxlon = 0
    mindepth = 0
    maxdepth = 0
    for patch in ff['features']:
        print(patch)
        # print(patch['properties']['x==ew'])
        continue
        sps = []
        for ii, sp in enumerate(patch['geometry']['coordinates'][0]):
            if ii == 4:
                continue

            lon = sp[0]
            lat = sp[1]
            depth = sp[2] / 1000.

            sps.append(Point(longitude=lon, latitude=lat, depth=depth))

            lons.append(lon)
            lats.append(lat)
            depths.append(depth)

            if depth <= mindepth:
                mindepth = depth
                # if lat 

        ms = PlanarSurface.from_corner_points(top_left=sps[0],
                                            top_right=sps[1],
                                            bottom_left=sps[3],
                                            bottom_right=sps[2])
        ps.append(ms.get_middle_point())

    surface = GriddedSurface.from_points_list(ps)
    print(surface)
    print(surface.get_surface_boundaries_3d())
    print(max(lons), min(lons))
    print(max(lats), min(lats))
    print(max(depths), min(depths))
    print()
    print(surface.surface_nodes)
    print(surface.get_bounding_box())
    # print(ff)
    # with open(faultFile) as f:
    #   for ii, line in enumerate(f):
    #       line = line.rsplit()
    #       if ii == 0:
    #           keyname = 'UL'
    #       elif ii == 1:
    #           keyname = 'UR'
    #       elif ii == 2:
    #           keyname = 'LR'
    #       elif ii == 3:
    #           keyname = 'LL'
    #       else:
    #           continue

    #       plainDict[keyname] = [float(line[0]), float(line[1]), float(line[2])]

    # print(plainDict)
    exit()
    return plainDict


def read_json_faultfile(faultFile, source):

    with open(faultFile) as f:
        ff = json.load(f)
    # print(ff)
    lat = ff['metadata']['lat']
    lon = ff['metadata']['lon']
    depth = ff['metadata']['depth']
    mag = ff['metadata']['mag']

    source.lat = float(lat)
    source.lon = float(lon)
    source.depth = float(depth)
    source.magnitude = float(mag)

    if len(ff['features']) > 1:
        print('Multi patch rupture found. Exit!')
        exit()

    if ff['features'][0]['geometry']['type'] == 'Point':
        print('Point source')
        return source

    patch = ff['features'][0]['geometry']['coordinates'][0][0]
    corners = {}
    for ii, line in enumerate(patch):
        # print(ii, line)
        if ii == 0:
            keyname = 'UL'
        elif ii == 1:
            keyname = 'UR'
        elif ii == 2:
            keyname = 'LR'
        elif ii == 3:
            keyname = 'LL'
        else:
            continue

        corners[keyname] = [float(line[0]), float(line[1]), float(line[2])]

    source.rupture = corners
    surface = corners2surface(source, corners)
    if surface:
        source = convert_surface(source, surface)

    return source


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


def check_for_fault_plain(directory, source):
    for ending in ['srcmod.fsp',
                   'fault_esm.txt',
                   'finite_fault_usgs.fsp',
                   'fault_usgs.txt']:  # order for hierarchic
        faultFile = directory + '/' + ending
        if os.path.exists(faultFile) and not os.stat(faultFile).st_size < 1:
            print(ending)
            if 'srcmod.fsp' in ending:
                source = read_fsp(faultFile, source)
                print('Fault from SRCMOD')

            elif 'fault_esm.txt' == ending:
                corners = read_esm_faultfile(faultFile)
                source.rupture = corners
                surface = corners2surface(source, corners)
                if surface:
                    source = convert_surface(source, surface)
                print('Fault from ESM')

            elif '.fsp' in ending:
                source = read_fsp(faultFile, source)
                # plainDict = read_usgs_finitefaultfile(faultFile)
                print('Fault from USGS finite fault')

            elif 'fault_usgs.txt' == ending:
                corners = read_usgs_faultfile(faultFile)
                source.rupture = corners
                surface = corners2surface(source, corners)
                if surface:
                    source = convert_surface(source, surface)
                print('Fault from USGS')

            break

    if source.form == 'point':
        surface = Mesh(num.array([source.lon]),
                       num.array([source.lat]),
                       num.array([source.depth]))

    return source
