# GPLv3
# The Developers, 21st Century
import os
import logging
import time
from multiprocessing import Pool, Manager

import pandas as pd
import numpy as num

from pyrocko import gf

import gmacc.util as GMu
import gmacc.sources as GMs
import gmacc.nn as GMn
import gmacc.config as GMcfg
import gmacc.interactive_gm_plot as GMint

logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'gm.log'),
    level=logging.DEBUG, filemode='w')  # do not delete this command
# logging.basicConfig(filename='gm.log', level=logging.DEBUG, filemode='w')
logger = logging.getLogger('ewricagm.gm_mapping')


def gm_mapping_read(ensemblefile, dest_dir, config_path=None, html=False):
    '''
    helper function for gm_mapping -
    Read a ensemble file of pyrocko source and
    processes it further (see function: gm_mapping)

    ensemblefile: path to the file containing ewrica sources
    dest_dir: directory to store the output
    config_path: path to the config file
    html: bool to enable plotting afterwards a HTML page/file    
    '''
    import siria.sources as ews
    srcs = ews.load_sources(filename=ensemblefile)

    if config_path is None:
        args = None
    else:
        args = GMcfg.GroundMotionConfig(config_path=config_path).get_config()

    gm_mapping(srcs, dest_dir, args, html=html)


def gm_mapping(srcs, dest_dir, args=None, html=False):
    '''
    ewricagm main function

    Starts the ground motion processing of a ewrica source ensemble.
    The processing can be done either with pre-trained Neural Networks or
    with a modified version of the Pyrocko chain for ground motion extraction.
    Produces a json file, containing location specific GM values,
    per source.
    And if desired, produces a HTML page showing those values.
    
    Input:
    srcs: list of ewrica sources (see ewrica-siria package)
    dest_dir: directory to store the output
    args: further arguments, if None, using values stored in
        default-config.yaml
    html: bool to enable plotting a HTML page/file afterwards

    Output:
    Create a .json in dest_dir.
    If html is True, generate a .html in dest_dir. 
    '''

    ### Can be deleted!

    print('Len all sources:', len(srcs))
    delsrcs = []
    for idx, src in enumerate(srcs):
        mt = src.pyrocko_moment_tensor()
        src.magnitude = mt.moment_magnitude()

        if src.magnitude > 7.5:
            delsrcs.append(idx)
        elif src.depth > 10000:
            delsrcs.append(idx)

    srcs = [i for j, i in enumerate(srcs) if j not in delsrcs]
    print(len(srcs))

    # exit()
    # srcs = srcs[:10]
    # print('Len all sources:', len(srcs))
    # probs = []
    # idxs = []
    # for idx in range(len(srcs)):
    #     sprob = srcs[idx].misfit
    #     probs.append(sprob)
    #     idxs.append(idx)
    # numb = 500
    # maxidxs = num.argpartition(probs, 0)[:numb]
    # srcs = num.array(srcs)[maxidxs]
    # print('Len after misfit:', len(srcs))
    ###

    if args is None:
        default_config = os.path.join(os.path.dirname(__file__),
            'configs/default_config.yaml')
        args = GMcfg.GroundMotionConfig(config_path=default_config).get_config()
        args.nndir = os.path.join(os.path.dirname(__file__), args.nndir)

    logger.info('\n###############\nStarting run')
    logger.info('Number of found sources: %s' % (len(srcs)))

    if len(srcs) == 0:
        logger.info('WARNING: number of sources: %s' % (len(srcs)))
        raise ValueError('No sources found.')

    prob = num.inf
    for ii in range(len(srcs)):
        sprob = srcs[ii].misfit
        if sprob < prob:
            prob_src = srcs[ii]
            prob = sprob

    print(prob_src)

    # Initialize weights/misfits
    weights = []
    misfits = []
    for src in srcs:
        try:
            weights.append(src.probability)
        except AttributeError:
            weights.append(0.0)
        misfits.append(src.misfit)

    if max(weights) == 0.0:
        misfits = num.array(misfits)
        weights = 1. / (misfits * sum(1. / misfits))

    # Create coordinates
    coords = None
    if args.mapmode == 'rectangular':
        coords = GMu.rectangular_mapping(prob_src, args.mapextent,
                                         args.mappoints, rmin=1.0)
    elif args.mapmode == 'circular':
        coords = GMu.circular_mapping(prob_src, args.mapextent,
                                      args.mappoints, rmin=0.1)
    
    if coords is None:
        logger.warning('Wrong Mapping: \'%s\'' % (args.mapmode))
        exit()

    if args.mp and args.mp > 1:
        logger.info('Starting multiprocessing')
        p = Pool(processes=args.mp)
        manager = Manager()
        srcs_dict = manager.dict()
    else:
        srcs_dict = {}
    calctime = time.time()

    #######################################
    # Neural Networks
    if args.method == 'NN':
       
        lons = list(coords.T[0])
        lats = list(coords.T[1])

        nndf = GMn.get_gdf_NN_together(srcs, args, lons, lats)
        nndf['lon'] = lons * len(srcs)
        nndf['lat'] = lats * len(srcs)

    # Pyrocko
    elif args.method == 'Pyrocko':
        logger.info('Pyrocko GF arguments: %s' % args.gf)
        engine = gf.LocalEngine(store_superdirs=[args.gf],
                                store_dirs=[args.gf])

        # Create Waveform targets
        waveform_targets = [
            gf.Target(
                quantity='displacement',
                lat=coords[ii][1], lon=coords[ii][0],
                store_id=str(args.gf.rsplit('/')[-2]),
                codes=('PR', 'STA' + str(ii), 'LOC', channel_code))
            for channel_code in args.comps
            for ii in range(len(coords))]

        # Processing
        xs = []
        for ii, src in enumerate(srcs):
            src.validate()
            if args.mp and args.mp > 1:
                x = p.apply_async(
                    GMs.get_gdf_Pyrocko,
                    args=(ii, args, src, engine, waveform_targets, srcs_dict))
                xs.append(x)
            else:
                GMs.get_gdf_Pyrocko(ii, args, src, engine,
                                    waveform_targets, srcs_dict)

    # Finish multiprocessing
    except_flag = False
    if args.mp and args.mp > 1:
        p.close()
        p.join()
        logger.info('multiprocessing done')

        for x in xs:
            if x.get():
                except_flag = True
                logger.info('''
For source number: {}
{}
{}'''.format(x.get()[0], x.get()[1], srcs[x.get()[0]]))

        logger.info('''
Processing time ({} sources, {} points, {} processors):
    {} s'''.format(len(srcs), len(coords), args.mp, time.time() - calctime))
    else:
        logger.info('''
Processing time ({} sources, {} points, {} processors):
    {} s'''.format(len(srcs), len(coords), 1, time.time() - calctime))

    if except_flag:
        exit()

    '''
    To do:
    - check for weighted mean and std, probably falsely calculated wstd!
    - which mean: log or non-log?
    '''
    sorttime = time.time()
    ignorecols = ['st_lon', 'st_lat', 'lon', 'lat', 'lons', 'lats', 'geometry']

    if args.method == 'Pyrocko':
        # Create dataframe with all values
        chagms = []
        # srcs_dict = pd.DataFrame(srcs_dict)
        alldf = pd.DataFrame()
        for srccnt, gdf in srcs_dict.items():
            # probability
            for im in gdf.columns:
                if im == 'geometry':
                    continue
                if im in ignorecols:
                    if srccnt == 0:
                        if im == 'st_lon':
                            oim = 'lon'
                        elif im == 'st_lat':
                            oim = 'lat'
                        else:
                            oim
                        alldf[oim] = gdf[im]
                    continue

                chagms.append(im)
                alldf['%s_%s' % (im, srccnt)] = gdf[im]

        chagms = list(set(chagms))  # important!

        # Calc (weighted) mean and std per comp and gm/im
        for chagm in chagms:
            cols = [col for col in alldf.columns if chagm in col]
            df = alldf[cols].copy(deep=True)

            alldf['%s_mean' % chagm] = df.mean(axis=1)
            alldf['%s_std' % chagm] = df.std(axis=1)

            wmean = num.average(df, axis=1, weights=weights)
            wvar = num.average(
                df.sub(wmean, axis='index')**2,
                axis=1,
                weights=weights)
            alldf['%s_wmean' % chagm] = wmean
            alldf['%s_wstd' % chagm] = num.sqrt(wvar)
            targets = chagms

    elif args.method == 'NN':

        #### nngdf sorting
        alldf = {}
        nncols = nndf.columns
        targets = [x for x in nncols if x not in ignorecols]
        numsrcs = len(srcs)
        numsamples = len(lons)

        for ss in range(numsrcs):
            for col in nncols:
                if col in ignorecols:
                    if ss == 0:
                        vals = nndf[col].iloc[ss * numsamples:
                            (ss + 1) * numsamples]
                        alldf[col] = vals
                    continue

                vals = nndf[col].iloc[ss * numsamples:
                    (ss + 1) * numsamples]
                alldf['%s_%s' % (col, ss)] = vals.to_numpy()

        for chagm in targets:
            df = {key: alldf[key] for key in alldf.keys() if chagm in key}
            df = pd.DataFrame(df)

            alldf['%s_mean' % chagm] = df.mean(axis=1)
            alldf['%s_std' % chagm] = df.std(axis=1)

            wmean = num.average(df, axis=1, weights=weights)
            wvar = num.average(
                df.sub(wmean, axis='index')**2,
                axis=1,
                weights=weights)
            alldf['%s_wmean' % chagm] = wmean
            alldf['%s_wstd' % chagm] = num.sqrt(wvar)

    logger.info('Sorting done in {} s'.format(time.time() - sorttime))

    ######## new
    presavetime = time.time()
    alldf = pd.DataFrame(alldf)

    for target in targets:
        mergecols = [col for col in alldf.columns if (len(col.rsplit('_')) == 3) and (col.rsplit('_')[-1].isdigit()) and (target in col)]
        alldf[target] = alldf[mergecols].values.tolist()
        alldf = alldf.drop(columns=mergecols)
    logger.info('Presave done in {} s'.format(time.time() - presavetime))
    resortalldf = alldf.T

    savetime = time.time()
    fn_json = os.path.join(dest_dir, 'predicted_data_%s.json' % args.method)
    resortalldf.to_json(fn_json)
    logger.info('Saved JSON as {} in {} s'.format(
        fn_json, time.time() - savetime))

    if html:
        GMint.plot_interactive_map(resortalldf, dest_dir, args)
