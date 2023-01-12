import numpy as num

from matplotlib import cm
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.tri as tri
from matplotlib.colors import Normalize
import matplotlib.patches as patches

from mpl_toolkits import basemap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from pyrocko import trace
from pyrocko.plot import beachball
from pyrocko import moment_tensor as pmt

import gmacc.gmeval.sources as GMs

#############################
### Plot functions
#############################
def owncolorbar(mappable, fig, ax, label=[], ticks=[], side='right'):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(side, size="5%", pad=0.05)

    if ticks != [] and len(ticks) < 20:
        cbar = fig.colorbar(mappable, cax=cax, label=label, ticks=ticks)
    else:
        cbar = fig.colorbar(mappable, cax=cax, label=label)

    if side == 'left':
        cbar.ax.yaxis.set_ticks_position('left')
        cbar.ax.yaxis.set_label_position('left')

    return



def plot_gm_map_alternative(source, predDict, obsDict=[], resDict=[], mapextent=[1, 1],
                savename='gm_map', figtitle=None, figtitlesize=16,
                cmapname='afmhot_r',
                # cmapdiffname='twilight_shifted',
                cmapdiffname='bwr',
                markersize=200., reslevel=None,
                predPlotMode='area', smoothfac=False, alpha=1,
                showcbar=True, shmLevels=None,
                minmax=False, plotgmvise=False, plotindi=False,
                valmode='log'):

    gm1 = next(iter(predDict.keys()))
    outRow = len(predDict.keys())
    outCol = len(predDict[gm1].keys())

    figWidth = outCol * 7
    figLength = outRow * 7

    if not plotgmvise and not plotindi:
        fig = plt.figure(figsize=(figWidth, figLength))
        outer = gridspec.GridSpec(outRow, outCol, wspace=0., hspace=0.)

    compCnt = -1
    for gm, gmParams in predDict.items():
        if minmax:
            if obsDict:
                if gm not in obsDict:
                    continue
                valMax = max(
                            max([max(num.array(vals['vals']))
                                for comp, vals in obsDict[gm].items()]),
                            max([max(num.array(vals['vals']))
                                for comp, vals in predDict[gm].items()]))
                valMin = min(
                            min([min(num.array(vals['vals']))
                                for comp, vals in obsDict[gm].items()]),
                            min([min(num.array(vals['vals']))
                                for comp, vals in predDict[gm].items()]))
            else:
                valMax = max([max(num.array(vals['vals'])) for comp, vals in predDict[gm].items()])
                valMin = min([min(num.array(vals['vals'])) for comp, vals in predDict[gm].items()])
        if plotgmvise and not plotindi:
            fig = plt.figure(figsize=(figWidth, 10))
            outer = gridspec.GridSpec(1, outCol, wspace=0., hspace=0.)
            compCnt = -1

        compCnt += 1
        n = -1
        for comp, vals in gmParams.items():
            n += 1

            if obsDict:
                if comp in obsDict[list(obsDict.keys())[0]]:
                    obsComp = comp
                elif 'H' in obsDict[list(obsDict.keys())[0]]:
                    obsComp = 'H'
                else:
                    obsComp = list(obsDict[list(obsDict.keys())[0]].keys())[0]

            if plotindi:
                fig = plt.figure(figsize=(10, 10))
                outer = gridspec.GridSpec(1, 1, wspace=0., hspace=0.)
                compCnt = 0
                n = 0

                if minmax:
                    if obsDict:
                        valMax = max(max(obsDict[gm][obsComp]['vals']), max(vals['vals']))
                        valMin = min(min(obsDict[gm][obsComp]['vals']), min(vals['vals']))
                    else:
                        valMax = max(vals['vals'])
                        valMin = min(vals['vals'])

            if len(comp) > 1 and gm in ['PGD', 'pgd']:
                titlestr = 'None'
                print('Smth with titlestring')
                continue

            ax = fig.add_subplot(outer[compCnt * len(gmParams) + (n)])
            ### data
            lons = vals['lons']
            lats = vals['lats']
            lons = num.array(lons)
            lats = num.array(lats)
            data = num.array(vals['vals'])

            if valmode == 'log':
                pass
            elif valmode in ['abs', 'true']:
                data = 10**data
            else:
                print('Wrong valmode:', valmode)

            if type(data) == float:
                print('Data type is float')
                print('Error')
                exit()

            if shmLevels is not None and minmax is not True:
                pass

            elif minmax:
                damp = 0.2
                shmLevels = num.arange(num.floor(valMin * 10.) / 10.,
                                       (num.ceil(valMax * 10.) / 10.) + damp,
                                       damp)

            elif gm in ['sigdur']:
                shmLevels = num.array(
                    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

            elif gm in ['ai']:
                shmLevels = num.log10(num.array(
                        [0.01, 0.02, 0.05,
                        0.1, 0.2, 0.5,
                        1, 2, 5,
                        10, 20, 50,
                        100, 200])) - 3

            elif ':' in gm:
                shmLevels = num.array([-2.0, -1.75, -1.5, -1.25,
                            -1.0, -0.75, -0.5, -0.25,
                            0.25, 0.5, 0.75,
                            1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
                cmapname = cmapdiffname

            else:
                # shmLevels = num.log10(num.array(
                #         [
                #         # 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 
                #         0.01, 0.02, 0.05,
                #         0.1, 0.2, 0.5,
                #         1, 2, 5,
                #         10, 20, 50,
                #         100, 200]))
                # shmLevels = num.log10(num.logspace(-2, 2, 100))
                shmLevels = num.log10(num.logspace(num.log10(0.01), num.log10(200), 25))

            cmap = cm.get_cmap(cmapname)  #, len(shmLevels))
            # norm = BoundaryNorm(shmLevels, ncolors=cmap.N, clip=False)

            norm = Normalize(clip=False,
                vmin=shmLevels.min(), vmax=shmLevels.max())

            ### map
            lowerLat = source.lat - mapextent[1]
            upperLat = source.lat + mapextent[1]
            lowerLon = source.lon - mapextent[0]
            upperLon = source.lon + mapextent[0]

            dLat = abs((lowerLat - upperLat) / (len(set(lats)) - 1))
            dLon = abs((lowerLon - upperLon) / (len(set(lons)) - 1))

            m = basemap.Basemap(projection='gall', ax=ax,
                                llcrnrlat=lowerLat - (dLat / 2), urcrnrlat=upperLat + (dLat / 2),
                                llcrnrlon=lowerLon - (dLon / 2), urcrnrlon=upperLon + (dLon / 2),
                                resolution='h', epsg=3857)  # 1 - 4326, 2 - 3857
            try:
                m.drawcoastlines()
            except ValueError as e:
                print(e)

            m.drawcountries(linewidth=2.0)
            # m.etopo()
            # m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1000, verbose=True)

            roundval = 2
            if upperLon - lowerLon > 2.:
                mers = num.arange(
                        num.floor(lowerLon),
                        num.ceil(upperLon) + dLon, 1)
            else:
                mers = [num.round(lowerLon + dLon, roundval),
                        num.round(source.lon, roundval),
                        num.round(upperLon - dLon, roundval)]
            
            if upperLat - lowerLat > 2.:
                paras = num.arange(
                        num.floor(lowerLat),
                        num.ceil(upperLat) + dLat, 1)
            else:
                paras = [num.round(lowerLat + dLat, roundval), 
                        num.round(source.lat, roundval),
                        num.round(upperLat - dLat, roundval)]

            if compCnt + 1 == len(predDict) or plotgmvise or plotindi:
                m.drawmeridians(mers,
                                labels=[0, 0, 0, 1])  # , rotation=90.)
            else:
                m.drawmeridians(mers,
                                labels=[0, 0, 0, 0])

            if n == 0:
                m.drawparallels(paras,
                                labels=[1, 0, 0, 0], rotation=90.)
                ax.set_ylabel(str(gm).upper(), labelpad=25, fontsize=20)
            else:
                m.drawparallels(paras,
                                labels=[0, 0, 0, 0])

            '''
            Plotting source
            '''
            xEvt, yEvt = m(source.lon, source.lat)

            if predPlotMode in ['tiles', 'area', 'tri']:
                rupcolor = 'white'
                # antirupcolor = 'black'
            else:
                rupcolor = 'black'
                # antirupcolor = 'white'
            # m.plot(xEvt, yEvt, 'r*', markersize=10, zorder=300.)

            if source.form in ['rectangular', 'pdr']:
                m.scatter(xEvt, yEvt, marker='X', s=100, c='black', edgecolors='white', zorder=300, linewidth=1)

                x1, y1 = m(source.rupture['UR'][0], source.rupture['UR'][1])
                x2, y2 = m(source.rupture['UL'][0], source.rupture['UL'][1])
                x3, y3 = m(source.rupture['LL'][0], source.rupture['LL'][1])
                x4, y4 = m(source.rupture['LR'][0], source.rupture['LR'][1])

                if predPlotMode in ['tiles', 'area', 'tri']:
                    rupcolor = 'white'
                else:
                    rupcolor = 'black'

                m.plot([x1, x2], [y1, y2], '-', color=rupcolor, linewidth=4., zorder=299)
                m.plot([x1, x4], [y1, y4], ':', color=rupcolor, linewidth=2., zorder=298.)
                m.plot([x2, x3], [y2, y3], ':', color=rupcolor, linewidth=2., zorder=298.)
                m.plot([x3, x4], [y3, y4], ':', color=rupcolor, linewidth=2., zorder=298.)

            if source.tensor is not None:
                tn = source['tensor']
                mt = pmt.MomentTensor(mnn=tn['mnn'], mee=tn['mee'], mdd=tn['mdd'],
                                      mne=tn['mne'], mnd=tn['mnd'], med=tn['med'])
            else:
                mt = pmt.MomentTensor(strike=source.strike, dip=source.dip,
                                rake=source.rake)

            if source.form in ['rectangular', 'pdr'] or valmode in ['abs', 'true']:
                # subax = plt.axes([0., 0., 0.2, 0.2], facecolor='y')
                subax = ax.inset_axes([0., 0., 0.2, 0.2], facecolor='None',
                            zorder=300)
                # plt.plot(t[:len(r)], r)
                # plt.title('Impulse response')
                # plt.xlim(0, 0.2)
                subax.set_xticks([])
                subax.set_yticks([])
                subax.axis('off')
                beachball.plot_beachball_mpl(mt, subax,
                                            position=(0.5, 0.5),
                                            size=20,
                                            zorder=300,
                                            color_t='black')
            else:
                beachball.plot_beachball_mpl(mt, ax,
                                            position=(xEvt, yEvt),
                                            color_t='black',
                                            zorder=300,
                                            size=20)

            '''
            Plotting predicted data
            '''
            x, y = m(lons, lats)

            if predPlotMode in ['point', 'scatter']:
                cs = m.scatter(x, y, s=markersize, c=data,
                            cmap=cmap, alpha=alpha,
                            edgecolor='white', zorder=2., linewidth=1,
                            vmin=shmLevels.min(), vmax=shmLevels.max())

            elif predPlotMode in ['contour']:

                lsize = len(list(set(lons)))
                xx = num.reshape(x, (lsize, lsize))
                yy = num.reshape(y, (lsize, lsize))
                zz = num.reshape(data, (lsize, lsize))
                # print(data)
                cs = m.contour(xx, yy, zz, levels=shmLevels, linewidths=5., 
                    zorder=2., cmap=cmap)
                try:
                    plt.clabel(cs, shmLevels, fmt='%1.1f')
                except:
                    print('No contours in range')
                    plt.clabel(cs, list(set(data)), fmt='%1.1f')

            elif predPlotMode in ['tiles']:

                cs = None
                cs = m.scatter(x, y, s=0, c=data,
                            cmap=cmap, alpha=alpha,
                            zorder=-2000.,
                            vmin=shmLevels.min(), vmax=shmLevels.max())
                
                for ii in range(len(lons)):
                    x1, y1 = m(lons[ii] - dLon / 2, lats[ii] - dLat / 2)
                    x2, y2 = m(lons[ii] + dLon / 2, lats[ii] + dLat / 2)
                    xdist = abs(x2 - x1)
                    ydist = abs(y2 - y1)

                    rect = patches.Rectangle((x1, y1), xdist, ydist,
                                        linewidth=1, color=cmap(norm(data[ii])))
                    ax.add_patch(rect)
    
            elif predPlotMode in ['tri', 'area']:
                try:
                    if smoothfac:
                        triang = tri.Triangulation(x, y)
                        # triang.set_mask(num.hypot(x[triang.triangles].mean(axis=1),
                        #                    y[triang.triangles].mean(axis=1))
                        #                    < min_radius)
                        refiner = tri.UniformTriRefiner(triang)
                        tri_refi, z_test_refi = refiner.refine_field(data,
                                                                subdiv=smoothfac)
                        cs = plt.tricontourf(tri_refi, z_test_refi, extend='both',
                                            cmap=cmap, levels=shmLevels,
                                            alpha=alpha,
                                            vmin=shmLevels.min(), vmax=shmLevels.max())
                    else:
                        cs = plt.tricontourf(x, y, data, extend='both',
                                cmap=cmap, alpha=alpha,
                                levels=shmLevels,
                                vmin=shmLevels.min(), vmax=shmLevels.max()
                                )
                except ValueError as e:
                    print(e)
                    print('Error in:', gm, comp)
                    exit()

            elif predPlotMode in ['resArea', 'resScatter']:
                pass
            else:
                print('### Wrong plotMode: %s' % predPlotMode)
                exit()

            if predPlotMode in ['resArea', 'resScatter']:
                pass
            else:
                # if n % outCol == outCol - 1:#
                if n == 0:
                    if gm == 'pga':
                        label = '% g in log10'
                    elif gm == 'pgv':
                        label = 'cm/s in log10'
                    elif gm == 'pgd':
                        label = 'cm in log10'
                    elif gm == 'sigdur':
                        label = 's'
                    elif gm == 'ai':
                        label = 'cm/s in log10'
                    elif ':' in gm:
                        label = 'Log(Ratio)'
                    else:
                        label = 'UKN'
                    if cs is not None and showcbar:
                        owncolorbar(cs, fig=fig, ax=ax, ticks=shmLevels, label=label)

            '''
            Observed Data
            '''
            if obsDict:
                if gm not in obsDict:
                    continue
                obsLon = obsDict[gm][obsComp]['lons']
                obsLat = obsDict[gm][obsComp]['lats']
                obsData = num.array(obsDict[gm][obsComp]['vals'])
                if valmode == 'log':
                    pass
                elif valmode in ['abs', 'true']:
                    obsData = 10**obsData

                if resDict:
                    '''
                    Plotting residuum between
                    '''
                    if reslevel is None:
                        if gm in ['sigdur']:
                            reslevel = num.linspace(-20, 20, 21)
                            ticks = num.linspace(-20, 20, 5)
                        elif gm in ['ai']:
                            reslevel = num.linspace(-1, 1, 21)
                            ticks = num.linspace(-1, 1, 5)
                        else:
                            reslevel = num.linspace(-1, 1, 21)
                            ticks = num.linspace(-1, 1, 5)
                    else:
                        ticks = reslevel

                    resData = num.array(resDict[gm][comp]['vals'])
                    if valmode == 'log':
                        pass
                    elif valmode in ['abs', 'true']:
                        resData = 10**resData

                    residuum = resData - obsData
                    refx, refy = m(obsLon, obsLat)

                    if valmode in ['abs', 'true']:
                        # absres = max(abs(residuum))
                        # reslevel = num.linspace(-absres, absres, 21)
                        stdres = num.std(residuum)
                        reslevel = num.linspace(-stdres, stdres, 21)
                        ticks = reslevel

                        if gm == 'pga':
                            label = '% g'
                        elif gm == 'pgv':
                            label = 'cm/s'
                        elif gm == 'pgd':
                            label = 'cm'
                        elif gm == 'sigdur':
                            label = 's'
                        elif gm == 'ai':
                            label = 'cm/s'
                        elif ':' in gm:
                            label = 'Ratio'
                        else:
                            label = 'UKN'
                        label = 'Difference ' + label
                    
                    else:
                        if gm == 'pga':
                            label = '% g in log10'
                        elif gm == 'pgv':
                            label = 'cm/s in log10'
                        elif gm == 'pgd':
                            label = 'cm in log10'
                        elif gm == 'sigdur':
                            label = 's'
                        elif gm == 'ai':
                            label = 'cm/s in log10'
                        elif ':' in gm:
                            label = 'Log(Ratio)'
                        else:
                            label = 'UKN'
                        label = 'Difference ' + label

                    if predPlotMode in ['resArea']:
                        residuum = num.nan_to_num(residuum)

                        if smoothfac:
                            triang = tri.Triangulation(refx, refy)
                            # triang.set_mask(num.hypot(x[triang.triangles].mean(axis=1),
                            #                    y[triang.triangles].mean(axis=1))
                            #                    < min_radius)
                            refiner = tri.UniformTriRefiner(triang)
                            tri_refi, z_test_refi = refiner.refine_field(residuum,
                                                                    subdiv=smoothfac)
                            sc = plt.tricontourf(tri_refi, z_test_refi, extend='both',
                                                cmap=cm.seismic, alpha=alpha,
                                                levels=reslevel,
                                                vmin=min(reslevel), vmax=max(reslevel))
                        else:
                            sc = plt.tricontourf(refx, refy, residuum, extend='both',
                                    cmap=cmapdiffname, alpha=alpha,
                                    levels=reslevel,
                                    vmin=min(reslevel), vmax=max(reslevel))

                        if n == 0 and showcbar:
                            owncolorbar(sc, fig=fig, ax=ax,
                                        label=label,
                                        ticks=ticks,
                                        side='right')

                    else:
                        # size = ((num.abs(residuum) + 0.1) * 300.)
                        # size = 4**(num.abs(residuum) + 3.)
                        maxsize = markersize * 1.25
                        minsize = markersize / 5.
                        size = (num.abs(residuum) * maxsize)
                        size[size > maxsize] = maxsize
                        size[size < minsize] = minsize
                        # print(num.abs(residuum))
                        # print(size)
                        sc = m.scatter(refx, refy,
                                # s=markersize,
                                s=size,
                                c=residuum,
                                cmap=cmapdiffname, zorder=20.,
                                vmin=min(reslevel), vmax=max(reslevel))

                        if n == 0 and showcbar:
                            owncolorbar(sc, fig=fig, ax=ax,
                                        label='Difference [log10]',
                                        ticks=ticks,
                                        side='right')

                    titlestr = 'Difference-Plot'

                else:
                    '''
                    Plotting observed data
                    '''
                    refx, refy = m(obsLon, obsLat)
                    m.scatter(refx, refy, s=markersize, c=obsData,
                            cmap=cmap, zorder=20.,  # alpha=0.1,
                            edgecolor='white', linewidth=1,
                            vmin=shmLevels.min(), vmax=shmLevels.max())
            if figtitle is None:
                titlestr = '%s' % (source.name)
                if hasattr(source, 'region'):
                    titlestr += ', %s' % (source.region)
                titlestr += '\nMag: %0.1f, Depth: %0.1f' \
                    % (source.magnitude, source.depth)

                if hasattr(source, 'form') and source.form == 'rectangular':
                    titlestr += ' Strike: %0.1f, Dip: %0.1f, Rake: %0.1f' \
                        % (source.strike, source.dip, source.rake)
                    titlestr += '\nNucX: %.1f, NucY: %.1f' \
                        % (source.nucleation_x, source.nucleation_y)

                elif source.tensor:
                    tn = source['tensor']
                    titlestr += '\nmnn: %0.1e, mee: %0.1e, mdd: %0.1e\nmne: %0.1e, mnd: %0.1e, med: %0.1e,' \
                        % (tn['mnn'], tn['mee'], tn['mdd'], tn['mne'], tn['mnd'], tn['med'])

                elif source.strike is not None and source.dip is not None \
                    and source.rake is not None:
                    titlestr += ' Strike: %0.1f, Dip: %0.1f, Rake: %0.1f' \
                        % (source.strike, source.dip, source.rake)

                else:
                    titlestr += '\nExplosion source'
            else:
                titlestr = figtitle

            if compCnt == 0:
                if obsDict:
                    # ax.set_title('%s vs %s_obs' % (comp, obsComp), fontsize=30)
                    ax.set_title('%s' % (comp), fontsize=30)
                else:
                    ax.set_title('%s' % (comp), fontsize=30)

            fig.add_subplot(ax)
            if plotindi:
                plt.suptitle(titlestr, fontsize=figtitlesize)
                plt.tight_layout()
                if savename != [] and savename != '':
                    fig.savefig('%s_%s_%s.png' % (savename, gm, comp))

        if plotgmvise and not plotindi:
            # outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
            plt.suptitle(titlestr, fontsize=figtitlesize)
            plt.tight_layout()
            if savename != [] and savename != '':
                fig.savefig('%s_%s.png' % (savename, gm))
                # plt.close()

    if not plotgmvise and not plotindi:
        outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
        plt.suptitle(titlestr, fontsize=figtitlesize)
        if savename != [] and savename != '':
            fig.savefig('%s.png' % (savename))
        else:
            return m
    else:
        # plt.close('all')
        return m



def plot_1d_alternative(source, obsDict, resDict, mode='dist', distType='hypo', aziType='hypo',
            savename='gm_diagram', figtitle=None, valmode='log',
            plotgmvise=False, plotindi=False):

    gm1 = next(iter(obsDict.keys()))
    outRow = len(obsDict.keys())
    outCol = len(obsDict[gm1].keys())

    figWidth = outCol * 7
    figLength = outRow * 5

    if not plotgmvise and not plotindi:
        fig = plt.figure(figsize=(figWidth, figLength))
        outer = gridspec.GridSpec(outRow, outCol, wspace=0., hspace=0.)

    if mode in ['dist', 'distance']:
        if hasattr(source, 'surface'):
            if distType == 'hypo':
                distStr = 'Hypocentral'
            elif distType == 'rrup':
                distStr = 'Minimal Rupture'
            elif distType == 'rx':
                distStr = 'Perpendicular-to-Strike (rx)'
            elif distType == 'rjb':
                distStr = 'Joyner-Boore'
            elif distType == 'ry0':
                distStr = 'Parallel-to-Strike (ry0)'
        else:
            distType = 'hypo'
            distStr = 'Hypocentral'

    if mode in ['azi', 'azimuth']:
        if hasattr(source, 'surface'):
            if aziType == 'hypo':
                aziStr = 'Hypocentral'
            elif aziType == 'rup':
                aziStr = 'Minimal Rupture'
            elif aziType == 'centre':
                aziStr = 'Ruptrue Centre'
        else:
            aziType = 'hypo'
            aziStr = 'Hypocentral'

    compCnt = -1
    for gm, gmParams in resDict.items():

        if plotgmvise and not plotindi:
            compCnt = -1
            fig = plt.figure(figsize=(figWidth, 10))
            outer = gridspec.GridSpec(1, outCol, wspace=0., hspace=0.)

        compCnt += 1
        n = -1
        valMax = max(
                    max([max(num.array(vals['vals']))
                        for comp, vals in obsDict[gm].items()]),
                    max([max(num.array(vals['vals']))
                        for comp, vals in resDict[gm].items()]))
        valMin = min(
                    min([min(num.array(vals['vals']))
                        for comp, vals in obsDict[gm].items()]),
                    min([min(num.array(vals['vals']))
                        for comp, vals in resDict[gm].items()]))

        for comp, vals in gmParams.items():
            n += 1

            if plotindi:
                # fig = plt.figure(figsize=(10, 5))
                fig = plt.figure(figsize=(8, 4))
                outer = gridspec.GridSpec(1, 1, wspace=0., hspace=0.)
                compCnt = 0
                n = 0

            inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[compCnt * len(gmParams) + (n)],
                    wspace=0., hspace=0.05,
                    width_ratios=[1], height_ratios=[1.5, 1])

            ax1 = plt.Subplot(fig, inner[0])
            ax2 = plt.Subplot(fig, inner[1])

            obsmarker = 'o'
            # obscolor = 'darkgreen'
            # obscolor = 'dimgrey'
            # obscolor = 'mediumseagreen'
            obscolor = 'black'
            # ax1color = 'orangered'
            # ax1color = 'black'
            ax1color = 'mediumseagreen'
            # ax1color = 'goldenrod'
            ax1marker = 'x'
            if len(comp) > 1:
                ax1color = 'goldenrod'
                ax1marker = '+'
                if gm in ['PGD', 'pgd']:
                    continue

            ax2color = 'darkgreen'
            # ax2color = 'black'
            ax2marker = '+'

            ### data
            lons = vals['lons']
            lats = vals['lats']
            data = num.array(vals['vals'])

            if comp in obsDict[list(obsDict.keys())[0]]:
                obsComp = comp
            elif 'H' in obsDict[list(obsDict.keys())[0]]:
                obsComp = 'H'
            else:
                obsComp = list(obsDict[list(obsDict.keys())[0]].keys())[0]
            obsLons = obsDict[gm][obsComp]['lons']
            obsLats = obsDict[gm][obsComp]['lats']
            obsData = num.array(obsDict[gm][obsComp]['vals'])
            
            if valmode == 'log':
                pass
            elif valmode in ['abs', 'true']:
                data = 10**data
                obsData = 10**obsData

            if mode in ['dist', 'distance']:
                '''
                Distance
                '''
                xdata = GMs.get_distances(lons, lats, source, distType=distType)
                xdataRef = GMs.get_distances(obsLons, obsLats, source, distType=distType)
                xdata[xdata <= 0.] = 0.1
                xdataRef[xdataRef <= 0.] = 0.1
                ax1.set_xscale('log')
                ax2.set_xscale('log')
                mindata = min(xdata) * 0.9
                maxdata = max(xdata) * 1.1
                ax1.set_xlim(mindata, maxdata)
                ax1.set_xticklabels([])
                ax2.set_xlim(mindata, maxdata)

                if compCnt + 1 == len(resDict) or plotgmvise or plotindi:
                    ax2.set_xlabel('%s - Distance [km]' % (distStr))

            elif mode in ['azi', 'azimuth']:
                '''
                Azimuth
                '''
                xdata = GMs.get_azimuths(lons, lats, source, aziType=aziType)
                xdataRef = GMs.get_azimuths(obsLons, obsLats, source, aziType=aziType)
                mindata = -180.
                maxdata = 180.
                ax1.set_xlim((mindata, maxdata))
                ax1.set_xticklabels([])
                ax2.set_xlim((mindata, maxdata))

                if compCnt + 1 == len(resDict) or plotgmvise or plotindi:
                    ax2.set_xlabel('%s - Azimuth [Degree]' % (aziStr))

            else:
                print('Wrong mode')
                exit()

            if n == 0:
                if valmode == 'log':
                    if gm == 'pga':
                        label = '% g in log10'
                    elif gm == 'pgv':
                        label = 'cm/s in log10'
                    elif gm == 'pgd':
                        label = 'cm in log10'
                    elif gm == 'sigdur':
                        label = 's'
                    elif gm == 'ai':
                        label = 'cm/s in log10'
                    elif ':' in gm:
                        label = 'Log(Ratio)'
                    else:
                        label = 'UKN'
                    ax1.set_ylabel('%s [%s]' % (str(gm).upper(), label))
                    ax2.set_ylabel('Difference [log10]', color=ax2color)
                else:
                    if gm == 'pga':
                        label = '% g'
                    elif gm == 'pgv':
                        label = 'cm/s'
                    elif gm == 'pgd':
                        label = 'cm'
                    elif gm == 'sigdur':
                        label = 's'
                    elif gm == 'ai':
                        label = 'cm/s'
                    elif ':' in gm:
                        label = 'Ratio'
                    else:
                        label = 'UKN'
                    ax1.set_ylabel('%s [%s]' % (str(gm).upper(), label))
                    ax2.set_ylabel('Difference', color=ax2color)

            '''
            Plotting
            '''
            ## Primary plot, e.g. Azimuth or distance
            ax1.plot(xdata, data, marker=ax1marker, color=ax1color,
                linestyle='None', label='Predicted')
            ax1.plot(xdataRef, obsData, marker=obsmarker, color=obscolor,
                linestyle='None', label='Observed', fillstyle='none')

            ax1.set_xticklabels([])
            ax1.tick_params(which='both', direction='in', bottom=True, top=True,
                            left=True, right=True)
            
            ## verbindung zwischen den correspondieren punkten
            # for nn in range(len(ydata)):
            #     color = 'gray'
            #     ax1.plot([ydata[nn], ydata[nn]],
            #             [obsData[nn], data[nn]], ':', zorder=-100, linewidth=0.5,
            #             color=color)
            if valmode == 'log':
                ax1.set_ylim((valMin - num.log10(2.), valMax + num.log10(5.)))
            else:
                pass
                # ax1.set_ylim((10**valMin, 10**valMax + num.log10(5.)))

            ## Secondary plot, residual
            residuum = data - obsData
            nn_mean = num.mean(residuum)
            nn_std = num.std(residuum)
            ax2.plot(xdata, residuum, marker=ax2marker, color=ax2color,
                linestyle='None')#, label='log10-Res')

            if valmode == 'log':
               
                gmpe_std = 0.3
                gmpep = ax2.fill_between(num.linspace(mindata, maxdata),
                    gmpe_std, -gmpe_std, color='grey',
                    alpha=0.3, label='μ=0; σ=%.1f' % gmpe_std)

            mup = ax2.plot((mindata, maxdata), (nn_mean, nn_mean),
                color='black', marker='+', linestyle='--', label='μ-PWS', zorder=-2)
            sigmap = ax2.plot((mindata, maxdata), (nn_mean + nn_std, nn_mean + nn_std),
                color='black', marker='*', linestyle=':', label='σ-PWS', zorder=-2)
            ax2.plot((mindata, maxdata), (nn_mean -nn_std, nn_mean -nn_std),
                color='black', linestyle=':', zorder=-2)
            textstr = 'μ=%0.2f; σ=%0.2f' % (nn_mean, nn_std)
            ax2.annotate(textstr, (0.98, 0.98), xycoords='axes fraction',
                        ha='right', va='top', fontsize=12, color=ax2color)

            ax2.tick_params(which='both', direction='in', bottom=True, top=True,
                            left=True, right=True)

            # else:
            #     mup = ax2.plot((mindata, maxdata), (0, 0),
            #         color='black', linestyle='--', zorder=-2)
            # maxval = num.ceil((max(ydata) / 10.)) * 10.
            # minval = num.floor((min(ydata) / 10.)) * 10.
            # # print(minval, maxval)

            # binwidth = int((maxval - minval) / 20)
            # # print(binwidth)
            # datmus = []
            # resmus = []
            # for lim1 in range(int(minval), int(maxval), binwidth):
            #     lim2 = lim1 + 1.5 * binwidth
            #     lim1 = lim1 - 1.5 * binwidth

            #     dat = ydata[(ydata < lim2) & (ydata > lim1)]
            #     res = residuum[(ydata < lim2) & (ydata > lim1)]

            #     if len(dat) == 0:
            #         continue
            #     datmus.append(num.mean(dat))
            #     resmus.append(num.mean(res))

            # ax2.plot(datmus, resmus, color=ax2color, linestyle='-',
            #         linewidth=1, zorder=-100)

            # coef = num.polyfit(ydata, residuum, 1)
            # poly1d_fn = num.poly1d(coef)
            # dummyX = num.linspace(min(ydata), max(ydata))
            # ax2.plot(dummyX, poly1d_fn(dummyX), '--', color=ax2color)

            if valmode == 'log':
                tmpmax = max(1.1, max(residuum))
                tmpmin = min(-1.1, min(residuum))
            elif valmode in ['abs', 'true']:
                tmpmax = 1.1 * max(abs(residuum))
                tmpmin = -tmpmax

            ax2.set_ylim((tmpmin, tmpmax))
            # ax2.axhline(y=0, color=ax2color, linestyle=':', alpha=0.3)
            # ax2.axhline(y=1, color=ax2color, linestyle=':', alpha=0.3)
            # ax2.axhline(y=-1, color=ax2color, linestyle=':', alpha=0.3)

            ax2.tick_params(axis='y', colors=ax2color)

            # ax1.set_title('%s\n%s vs. %s' % (gm, comp, obsComp))
            if compCnt == 0:
                # if obsCont:
                #     ax1.set_title('%s vs %s_obs' % (comp, obsComp), fontsize=30)
                # else:
                # ax1.set_title('%s' % (comp), fontsize=30)
                ax1.set_title('%s' % (comp), fontsize=15)

            fig.add_subplot(ax1)
            fig.add_subplot(ax2)

            if plotindi:
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, prop={'size': 8})
                plt.suptitle(figtitle)
                plt.tight_layout()
                if savename != [] and savename != '':
                    fig.savefig('%s_%s_%s.png' % (savename, gm, comp))

        if plotgmvise and not plotindi:
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2)
            plt.suptitle(figtitle)
            outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
            if savename != [] and savename != '':
                fig.savefig('%s_%s.png' % (savename, gm))

    if not plotgmvise and not plotindi:
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines + lines2, labels + labels2)

        ax1.legend(fontsize=10)
        # from matplotlib.legend_handler import HandlerTuple
        from matplotlib.legend_handler import HandlerBase

        class AnyObjectHandler(HandlerBase):
            def create_artists(self, legend, orig_handle,
                               x0, y0, width, height, fontsize, trans):
                l1 = plt.Line2D([x0, y0 + width], [0.75 * height, 0.75 * height], 
                    linestyle=orig_handle[1], color=orig_handle[0])
                l2 = plt.Line2D([x0, y0 + width], [0.25 * height, 0.25 * height],
                    linestyle=orig_handle[3], color=orig_handle[2])
                return [l1, l2]
        if valmode == 'log':        
            ax2.legend([(mup[0].get_color(), mup[0].get_linestyle(),
                    sigmap[0].get_color(), sigmap[0].get_linestyle()),
                    gmpep], ['PWS μ, σ', 'μ=0; σ=0.3'],
                loc='upper left',
                handler_map={tuple: AnyObjectHandler()}, fontsize=10)

        plt.suptitle(figtitle)
        outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
        if savename != [] and savename != '':
            fig.savefig('%s.png' % (savename))




def plot_gm_map(predCont, obsCont=[], resCont=[], mapextent=[1, 1],
                savename='gm_map', figtitle=None, figtitlesize=16,
                cmapname='afmhot_r', fontsize=10,
                # cmapdiffname='twilight_shifted',
                cmapdiffname='bwr',
                markersize=200., reslevel=None,
                predPlotMode='area', smoothfac=False, alpha=1,
                showcbar=True, shmLevels=None,
                minmax=False, plotgmvise=False, plotindi=False,
                valmode='log'):

    source = predCont.refSource
    # print(predCont)
    sta1 = next(iter(predCont.stations))
    comp1 = next(iter(predCont.stations[sta1].components))

    outCol = len(predCont.stations[sta1].components)
    outRow = len(predCont.stations[sta1].components[comp1].gms)
    figWidth = outCol * 5
    figLength = outRow * 5

    predDict = predCont.to_dictionary()

    if obsCont:
        obsDict = obsCont.to_dictionary()

        if resCont:
            resDict = resCont.to_dictionary()

    if not plotgmvise and not plotindi:
        fig = plt.figure(figsize=(figWidth, figLength))
        outer = gridspec.GridSpec(outRow, outCol, wspace=0., hspace=0.)

    compCnt = -1
    for gm, gmParams in predDict.items():
        if minmax:
            if obsCont:
                if gm not in obsDict:
                    continue
                valMax = max(
                            max([max(num.array(vals['vals']))
                                for comp, vals in obsDict[gm].items()]),
                            max([max(num.array(vals['vals']))
                                for comp, vals in predDict[gm].items()]))
                valMin = min(
                            min([min(num.array(vals['vals']))
                                for comp, vals in obsDict[gm].items()]),
                            min([min(num.array(vals['vals']))
                                for comp, vals in predDict[gm].items()]))
            else:
                valMax = max([max(num.array(vals['vals'])) for comp, vals in predDict[gm].items()])
                valMin = min([min(num.array(vals['vals'])) for comp, vals in predDict[gm].items()])
        if plotgmvise and not plotindi:
            fig = plt.figure(figsize=(figWidth, 10))
            outer = gridspec.GridSpec(1, outCol, wspace=0., hspace=0.)
            compCnt = -1

        compCnt += 1
        n = -1
        for comp, vals in gmParams.items():
            n += 1

            if obsCont:
                if comp in obsDict[list(obsDict.keys())[0]]:
                    obsComp = comp
                elif 'H' in obsDict[list(obsDict.keys())[0]]:
                    obsComp = 'H'
                else:
                    obsComp = list(obsDict[list(obsDict.keys())[0]].keys())[0]

            if plotindi:
                fig = plt.figure(figsize=(10, 10))
                outer = gridspec.GridSpec(1, 1, wspace=0., hspace=0.)
                compCnt = 0
                n = 0

                if minmax:
                    if obsCont:
                        valMax = max(max(obsDict[gm][obsComp]['vals']), max(vals['vals']))
                        valMin = min(min(obsDict[gm][obsComp]['vals']), min(vals['vals']))
                    else:
                        valMax = max(vals['vals'])
                        valMin = min(vals['vals'])

            if len(comp) > 1 and gm in ['PGD', 'pgd']:
                titlestr = 'None'
                print('Smth with titlestring')
                continue

            ax = fig.add_subplot(outer[compCnt * len(gmParams) + (n)])
            ### data
            lons = vals['lons']
            lats = vals['lats']
            lons = num.array(lons)
            lats = num.array(lats)
            data = num.array(vals['vals'])

            if valmode == 'log':
                pass
            elif valmode in ['abs', 'true']:
                data = 10**data
            else:
                print('Wrong valmode:', valmode)

            if type(data) == float:
                print('Data type is float')
                print('Error')
                exit()

            if shmLevels is not None and minmax is not True:
                pass

            elif minmax:
                damp = 0.2
                shmLevels = num.arange(num.floor(valMin * 10.) / 10.,
                                       (num.ceil(valMax * 10.) / 10.) + damp,
                                       damp)

            elif gm in ['sigdur']:
                shmLevels = num.array(
                    [0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

            elif gm in ['ai']:
                shmLevels = num.log10(num.array(
                        [0.01, 0.02, 0.05,
                        0.1, 0.2, 0.5,
                        1, 2, 5,
                        10, 20, 50,
                        100, 200])) - 3

            elif ':' in gm:
                shmLevels = num.array([-2.0, -1.75, -1.5, -1.25,
                            -1.0, -0.75, -0.5, -0.25,
                            0.25, 0.5, 0.75,
                            1.0, 1.25, 1.5, 1.75, 2.0, 2.25])
                cmapname = cmapdiffname

            else:
                # shmLevels = num.log10(num.array(
                #         [
                #         # 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 
                #         0.01, 0.02, 0.05,
                #         0.1, 0.2, 0.5,
                #         1, 2, 5,
                #         10, 20, 50,
                #         100, 200]))
                # shmLevels = num.log10(num.logspace(-2, 2, 100))
                shmLevels = num.log10(num.logspace(num.log10(0.01), num.log10(200), 25))

            cmap = cm.get_cmap(cmapname)  #, len(shmLevels))
            # norm = BoundaryNorm(shmLevels, ncolors=cmap.N, clip=False)

            norm = Normalize(clip=False,
                vmin=shmLevels.min(), vmax=shmLevels.max())

            ### map
            lowerLat = source.lat - mapextent[1]
            upperLat = source.lat + mapextent[1]
            lowerLon = source.lon - mapextent[0]
            upperLon = source.lon + mapextent[0]

            dLat = abs((lowerLat - upperLat) / (len(set(lats)) - 1))
            dLon = abs((lowerLon - upperLon) / (len(set(lons)) - 1))

            m = basemap.Basemap(projection='gall', ax=ax,
                                llcrnrlat=lowerLat - (dLat / 2), urcrnrlat=upperLat + (dLat / 2),
                                llcrnrlon=lowerLon - (dLon / 2), urcrnrlon=upperLon + (dLon / 2),
                                resolution='h', epsg=3857)  # 1 - 4326, 2 - 3857
            try:
                m.drawcoastlines()
            except ValueError as e:
                print(e)

            m.drawcountries(linewidth=2.0)
            # m.etopo()
            # m.arcgisimage(service='ESRI_Imagery_World_2D', xpixels = 1000, verbose=True)

            roundval = 2
            if upperLon - lowerLon > 2.:
                mers = num.arange(
                        num.floor(lowerLon),
                        num.ceil(upperLon) + dLon, 1)
            else:
                mers = [num.round(lowerLon + dLon, roundval),
                        num.round(source.lon, roundval),
                        num.round(upperLon - dLon, roundval)]
            
            if upperLat - lowerLat > 2.:
                paras = num.arange(
                        num.floor(lowerLat),
                        num.ceil(upperLat) + dLat, 1)
            else:
                paras = [num.round(lowerLat + dLat, roundval), 
                        num.round(source.lat, roundval),
                        num.round(upperLat - dLat, roundval)]

            if compCnt + 1 == len(predDict) or plotgmvise or plotindi:
                m.drawmeridians(mers,
                                labels=[0, 0, 0, 1])  # , rotation=90.)
            else:
                m.drawmeridians(mers,
                                labels=[0, 0, 0, 0])

            if n == 0:
                m.drawparallels(paras,
                                labels=[1, 0, 0, 0], rotation=90.)
                ax.set_ylabel(str(gm).upper(), labelpad=25)
            else:
                m.drawparallels(paras,
                                labels=[0, 0, 0, 0])

            '''
            Plotting source
            '''
            xEvt, yEvt = m(source.lon, source.lat)

            if predPlotMode in ['tiles', 'area', 'tri']:
                rupcolor = 'white'
                # antirupcolor = 'black'
            else:
                rupcolor = 'black'
                # antirupcolor = 'white'
            # m.plot(xEvt, yEvt, 'r*', markersize=10, zorder=300.)

            if source.form in ['rectangular', 'pdr']:
                m.scatter(xEvt, yEvt, marker='X', s=100, c='black', edgecolors='white', zorder=300, linewidth=1)

                x1, y1 = m(source.rupture['UR'][0], source.rupture['UR'][1])
                x2, y2 = m(source.rupture['UL'][0], source.rupture['UL'][1])
                x3, y3 = m(source.rupture['LL'][0], source.rupture['LL'][1])
                x4, y4 = m(source.rupture['LR'][0], source.rupture['LR'][1])

                if predPlotMode in ['tiles', 'area', 'tri']:
                    rupcolor = 'white'
                else:
                    rupcolor = 'black'

                m.plot([x1, x2], [y1, y2], '-', color=rupcolor, linewidth=4., zorder=299)
                m.plot([x1, x4], [y1, y4], ':', color=rupcolor, linewidth=2., zorder=298.)
                m.plot([x2, x3], [y2, y3], ':', color=rupcolor, linewidth=2., zorder=298.)
                m.plot([x3, x4], [y3, y4], ':', color=rupcolor, linewidth=2., zorder=298.)

            if source.tensor is not None:
                tn = source['tensor']
                mt = pmt.MomentTensor(mnn=tn['mnn'], mee=tn['mee'], mdd=tn['mdd'],
                                      mne=tn['mne'], mnd=tn['mnd'], med=tn['med'])
            else:
                mt = pmt.MomentTensor(strike=source.strike, dip=source.dip,
                                rake=source.rake)

            if source.form in ['rectangular', 'pdr'] or valmode in ['abs', 'true']:
                # subax = plt.axes([0., 0., 0.2, 0.2], facecolor='y')
                subax = ax.inset_axes([0., 0., 0.2, 0.2], facecolor='None',
                            zorder=300)
                # plt.plot(t[:len(r)], r)
                # plt.title('Impulse response')
                # plt.xlim(0, 0.2)
                subax.set_xticks([])
                subax.set_yticks([])
                subax.axis('off')
                beachball.plot_beachball_mpl(mt, subax,
                                            position=(0.5, 0.5),
                                            size=20,
                                            zorder=300,
                                            color_t='black')
            else:
                beachball.plot_beachball_mpl(mt, ax,
                                            position=(xEvt, yEvt),
                                            color_t='black',
                                            zorder=300,
                                            size=20)

            '''
            Plotting predicted data
            '''
            x, y = m(lons, lats)

            if predPlotMode in ['point', 'scatter']:
                cs = m.scatter(x, y, s=markersize, c=data,
                            cmap=cmap, alpha=alpha,
                            edgecolor='white', zorder=2., linewidth=1,
                            vmin=shmLevels.min(), vmax=shmLevels.max())

            elif predPlotMode in ['contour']:

                lsize = len(list(set(lons)))
                xx = num.reshape(x, (lsize, lsize))
                yy = num.reshape(y, (lsize, lsize))
                zz = num.reshape(data, (lsize, lsize))
                # print(data)
                cs = m.contour(xx, yy, zz, levels=shmLevels, linewidths=5., 
                    zorder=2., cmap=cmap)
                try:
                    plt.clabel(cs, shmLevels, fmt='%1.1f')
                except:
                    print('No contours in range')
                    plt.clabel(cs, list(set(data)), fmt='%1.1f')

            elif predPlotMode in ['tiles']:

                cs = None
                cs = m.scatter(x, y, s=0, c=data,
                            cmap=cmap, alpha=alpha,
                            zorder=-2000.,
                            vmin=shmLevels.min(), vmax=shmLevels.max())
                
                for ii in range(len(lons)):
                    x1, y1 = m(lons[ii] - dLon / 2, lats[ii] - dLat / 2)
                    x2, y2 = m(lons[ii] + dLon / 2, lats[ii] + dLat / 2)
                    xdist = abs(x2 - x1)
                    ydist = abs(y2 - y1)

                    rect = patches.Rectangle((x1, y1), xdist, ydist,
                                        linewidth=1, color=cmap(norm(data[ii])))
                    ax.add_patch(rect)
    
            elif predPlotMode in ['tri', 'area']:
                try:
                    if smoothfac:
                        triang = tri.Triangulation(x, y)
                        # triang.set_mask(num.hypot(x[triang.triangles].mean(axis=1),
                        #                    y[triang.triangles].mean(axis=1))
                        #                    < min_radius)
                        refiner = tri.UniformTriRefiner(triang)
                        tri_refi, z_test_refi = refiner.refine_field(data,
                                                                subdiv=smoothfac)
                        cs = plt.tricontourf(tri_refi, z_test_refi, extend='both',
                                            cmap=cmap, levels=shmLevels,
                                            alpha=alpha,
                                            vmin=shmLevels.min(), vmax=shmLevels.max())
                    else:
                        cs = plt.tricontourf(x, y, data, extend='both',
                                cmap=cmap, alpha=alpha,
                                levels=shmLevels,
                                vmin=shmLevels.min(), vmax=shmLevels.max()
                                )
                except ValueError as e:
                    print(e)
                    print('Error in:', gm, comp)
                    exit()

            elif predPlotMode in ['resArea', 'resScatter']:
                pass
            else:
                print('### Wrong plotMode: %s' % predPlotMode)
                exit()

            if predPlotMode in ['resArea', 'resScatter']:
                pass
            else:
                # if n % outCol == outCol - 1:#
                if n == 0:
                    if gm == 'pga':
                        label = '% g in log10'
                    elif gm == 'pgv':
                        label = 'cm/s in log10'
                    elif gm == 'pgd':
                        label = 'cm in log10'
                    elif gm == 'sigdur':
                        label = 's'
                    elif gm == 'ai':
                        label = 'cm/s in log10'
                    elif ':' in gm:
                        label = 'Log(Ratio)'
                    else:
                        label = 'UKN'
                    if cs is not None and showcbar:
                        owncolorbar(cs, fig=fig, ax=ax, ticks=shmLevels, label=label)

            '''
            Observed Data
            '''
            if obsCont:
                if gm not in obsDict:
                    continue
                obsLon = obsDict[gm][obsComp]['lons']
                obsLat = obsDict[gm][obsComp]['lats']
                obsData = num.array(obsDict[gm][obsComp]['vals'])
                if valmode == 'log':
                    pass
                elif valmode in ['abs', 'true']:
                    obsData = 10**obsData

                if resCont:
                    '''
                    Plotting residuum between
                    '''
                    if reslevel is None:
                        if gm in ['sigdur']:
                            reslevel = num.linspace(-20, 20, 21)
                            ticks = num.linspace(-20, 20, 5)
                        elif gm in ['ai']:
                            reslevel = num.linspace(-1, 1, 21)
                            ticks = num.linspace(-1, 1, 5)
                        else:
                            reslevel = num.linspace(-1, 1, 21)
                            ticks = num.linspace(-1, 1, 5)
                    else:
                        ticks = reslevel

                    resData = num.array(resDict[gm][comp]['vals'])
                    if valmode == 'log':
                        pass
                    elif valmode in ['abs', 'true']:
                        resData = 10**resData

                    residuum = resData - obsData
                    refx, refy = m(obsLon, obsLat)

                    if valmode in ['abs', 'true']:
                        # absres = max(abs(residuum))
                        # reslevel = num.linspace(-absres, absres, 21)
                        stdres = num.std(residuum)
                        reslevel = num.linspace(-stdres, stdres, 21)
                        ticks = reslevel

                        if gm == 'pga':
                            label = '% g'
                        elif gm == 'pgv':
                            label = 'cm/s'
                        elif gm == 'pgd':
                            label = 'cm'
                        elif gm == 'sigdur':
                            label = 's'
                        elif gm == 'ai':
                            label = 'cm/s'
                        elif ':' in gm:
                            label = 'Ratio'
                        else:
                            label = 'UKN'
                        label = 'Difference ' + label
                    
                    else:
                        if gm == 'pga':
                            label = '% g in log10'
                        elif gm == 'pgv':
                            label = 'cm/s in log10'
                        elif gm == 'pgd':
                            label = 'cm in log10'
                        elif gm == 'sigdur':
                            label = 's'
                        elif gm == 'ai':
                            label = 'cm/s in log10'
                        elif ':' in gm:
                            label = 'Log(Ratio)'
                        else:
                            label = 'UKN'
                        label = 'Difference ' + label

                    if predPlotMode in ['resArea']:
                        residuum = num.nan_to_num(residuum)

                        if smoothfac:
                            triang = tri.Triangulation(refx, refy)
                            # triang.set_mask(num.hypot(x[triang.triangles].mean(axis=1),
                            #                    y[triang.triangles].mean(axis=1))
                            #                    < min_radius)
                            refiner = tri.UniformTriRefiner(triang)
                            tri_refi, z_test_refi = refiner.refine_field(residuum,
                                                                    subdiv=smoothfac)
                            sc = plt.tricontourf(tri_refi, z_test_refi, extend='both',
                                                cmap=cm.seismic, alpha=alpha,
                                                levels=reslevel,
                                                vmin=min(reslevel), vmax=max(reslevel))
                        else:
                            sc = plt.tricontourf(refx, refy, residuum, extend='both',
                                    cmap=cmapdiffname, alpha=alpha,
                                    levels=reslevel,
                                    vmin=min(reslevel), vmax=max(reslevel))

                        if n == 0 and showcbar:
                            owncolorbar(sc, fig=fig, ax=ax,
                                        label=label,
                                        ticks=ticks,
                                        side='right')

                    else:
                        # size = ((num.abs(residuum) + 0.1) * 300.)
                        # size = 4**(num.abs(residuum) + 3.)
                        maxsize = markersize * 1.25
                        minsize = markersize / 5.
                        size = (num.abs(residuum) * maxsize)
                        size[size > maxsize] = maxsize
                        size[size < minsize] = minsize
                        # print(num.abs(residuum))
                        # print(size)
                        sc = m.scatter(refx, refy,
                                # s=markersize,
                                s=size,
                                c=residuum,
                                cmap=cmapdiffname, zorder=20.,
                                vmin=min(reslevel), vmax=max(reslevel))

                        if n == 0 and showcbar:
                            owncolorbar(sc, fig=fig, ax=ax,
                                        label='Difference [log10]',
                                        ticks=ticks,
                                        side='right')

                    titlestr = 'Difference-Plot'

                else:
                    '''
                    Plotting observed data
                    '''
                    refx, refy = m(obsLon, obsLat)
                    m.scatter(refx, refy, s=markersize, c=obsData,
                            cmap=cmap, zorder=20.,  # alpha=0.1,
                            edgecolor='white', linewidth=1,
                            vmin=shmLevels.min(), vmax=shmLevels.max())
            if figtitle is None:
                titlestr = '%s' % (source.name)
                if hasattr(source, 'region'):
                    titlestr += ', %s' % (source.region)
                titlestr += '\nMag: %0.1f, Depth: %0.1f' \
                    % (source.magnitude, source.depth)

                if hasattr(source, 'form') and source.form == 'rectangular':
                    titlestr += ' Strike: %0.1f, Dip: %0.1f, Rake: %0.1f' \
                        % (source.strike, source.dip, source.rake)
                    titlestr += '\nNucX: %.1f, NucY: %.1f' \
                        % (source.nucleation_x, source.nucleation_y)

                elif source.tensor:
                    tn = source['tensor']
                    titlestr += '\nmnn: %0.1e, mee: %0.1e, mdd: %0.1e\nmne: %0.1e, mnd: %0.1e, med: %0.1e,' \
                        % (tn['mnn'], tn['mee'], tn['mdd'], tn['mne'], tn['mnd'], tn['med'])

                elif source.strike is not None and source.dip is not None \
                    and source.rake is not None:
                    titlestr += ' Strike: %0.1f, Dip: %0.1f, Rake: %0.1f' \
                        % (source.strike, source.dip, source.rake)

                else:
                    titlestr += '\nExplosion source'
            else:
                titlestr = figtitle

            if compCnt == 0:
                if obsCont:
                    # ax.set_title('%s vs %s_obs' % (comp, obsComp), fontsize=30)
                    ax.set_title('%s' % (comp))
                else:
                    ax.set_title('%s' % (comp))

            # items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
            items = [ax.title, ax.xaxis.label, ax.yaxis.label]
            for item in (items):
                item.set_fontsize(fontsize)

            fig.add_subplot(ax)
            if plotindi:
                plt.suptitle(titlestr)
                plt.tight_layout()
                if savename != [] and savename != '':
                    fig.savefig('%s_%s_%s.png' % (savename, gm, comp))

        if plotgmvise and not plotindi:
            # outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
            plt.suptitle(titlestr)
            plt.tight_layout()
            if savename != [] and savename != '':
                fig.savefig('%s_%s.png' % (savename, gm))
                # plt.close()

    if not plotgmvise and not plotindi:
        outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
        plt.suptitle(titlestr)
        if savename != [] and savename != '':
            fig.savefig('%s.png' % (savename))
        else:
            return m
    else:
        # plt.close('all')
        return m


def plot_1d(obsCont, resCont, mode='dist', distType='hypo', aziType='hypo',
            savename='gm_diagram', figtitle=None, valmode='log', 
            fontsize=10,
            plotgmvise=False, plotindi=False):
    source = resCont.refSource

    sta1 = next(iter(resCont.stations))
    comp1 = next(iter(resCont.stations[sta1].components))

    outCol = len(resCont.stations[sta1].components)
    outRow = len(resCont.stations[sta1].components[comp1].gms)
    figWidth = outCol * 7
    figHeight = outRow * 5
    # figWidth = outCol * 1
    # figHeight = outRow * 1

    if not plotgmvise and not plotindi:
        fig = plt.figure(figsize=(figWidth, figHeight))
        outer = gridspec.GridSpec(outRow, outCol, wspace=0., hspace=0.)

    obsDict = obsCont.to_dictionary()
    resDict = resCont.to_dictionary()

    if mode in ['dist', 'distance']:
        if hasattr(source, 'surface'):
            if distType == 'hypo':
                distStr = 'Hypocentral'
            elif distType == 'rrup':
                distStr = 'Minimal Rupture'
            elif distType == 'rx':
                distStr = 'Perpendicular-to-Strike (rx)'
            elif distType == 'rjb':
                distStr = 'Joyner-Boore'
            elif distType == 'ry0':
                distStr = 'Parallel-to-Strike (ry0)'
        else:
            distType = 'hypo'
            distStr = 'Hypocentral'

    if mode in ['azi', 'azimuth']:
        if hasattr(source, 'surface'):
            if aziType == 'hypo':
                aziStr = 'Hypocentral'
            elif aziType == 'rup':
                aziStr = 'Minimal Rupture'
            elif aziType == 'centre':
                aziStr = 'Ruptrue Centre'
        else:
            aziType = 'hypo'
            aziStr = 'Hypocentral'

    compCnt = -1
    for gm, gmParams in resDict.items():

        if plotgmvise and not plotindi:
            compCnt = -1
            fig = plt.figure(figsize=(figWidth, 10))
            outer = gridspec.GridSpec(1, outCol, wspace=0., hspace=0.)

        compCnt += 1
        n = -1
        valMax = max(
                    max([max(num.array(vals['vals']))
                        for comp, vals in obsDict[gm].items()]),
                    max([max(num.array(vals['vals']))
                        for comp, vals in resDict[gm].items()]))
        valMin = min(
                    min([min(num.array(vals['vals']))
                        for comp, vals in obsDict[gm].items()]),
                    min([min(num.array(vals['vals']))
                        for comp, vals in resDict[gm].items()]))

        for comp, vals in gmParams.items():
            n += 1

            if plotindi:
                # fig = plt.figure(figsize=(10, 5))
                fig = plt.figure(figsize=(8, 4))
                outer = gridspec.GridSpec(1, 1, wspace=0., hspace=0.)
                compCnt = 0
                n = 0

            inner = gridspec.GridSpecFromSubplotSpec(2, 1,
                    subplot_spec=outer[compCnt * len(gmParams) + (n)],
                    wspace=0., hspace=0.05,
                    width_ratios=[1], height_ratios=[1.5, 1])

            ax1 = plt.Subplot(fig, inner[0])
            ax2 = plt.Subplot(fig, inner[1])

            obsmarker = 'o'
            # obscolor = 'darkgreen'
            # obscolor = 'dimgrey'
            # obscolor = 'mediumseagreen'
            obscolor = 'black'
            # ax1color = 'orangered'
            # ax1color = 'black'
            ax1color = 'mediumseagreen'
            # ax1color = 'goldenrod'
            ax1marker = 'x'
            if len(comp) > 1:
                ax1color = 'goldenrod'
                ax1marker = '+'
                if gm in ['PGD', 'pgd']:
                    continue

            ax2color = 'darkgreen'
            # ax2color = 'black'
            ax2marker = '+'

            ### data
            lons = vals['lons']
            lats = vals['lats']
            data = num.array(vals['vals'])

            if comp in obsDict[list(obsDict.keys())[0]]:
                obsComp = comp
            elif 'H' in obsDict[list(obsDict.keys())[0]]:
                obsComp = 'H'
            else:
                obsComp = list(obsDict[list(obsDict.keys())[0]].keys())[0]
            obsLons = obsDict[gm][obsComp]['lons']
            obsLats = obsDict[gm][obsComp]['lats']
            obsData = num.array(obsDict[gm][obsComp]['vals'])
            
            if valmode == 'log':
                pass
            elif valmode in ['abs', 'true']:
                data = 10**data
                obsData = 10**obsData

            if mode in ['dist', 'distance']:
                '''
                Distance
                '''
                xdata = GMs.get_distances(lons, lats, source, distType=distType)
                xdataRef = GMs.get_distances(obsLons, obsLats, source, distType=distType)
                xdata[xdata <= 0.] = 0.1
                xdataRef[xdataRef <= 0.] = 0.1
                ax1.set_xscale('log')
                ax2.set_xscale('log')
                mindata = min(xdata) * 0.9
                maxdata = max(xdata) * 1.1
                ax1.set_xlim(mindata, maxdata)
                ax1.set_xticklabels([])
                ax2.set_xlim(mindata, maxdata)

                if compCnt + 1 == len(resDict) or plotgmvise or plotindi:
                    ax2.set_xlabel('%s - Distance [km]' % (distStr))

            elif mode in ['azi', 'azimuth']:
                '''
                Azimuth
                '''
                xdata = GMs.get_azimuths(lons, lats, source, aziType=aziType)
                xdataRef = GMs.get_azimuths(obsLons, obsLats, source, aziType=aziType)
                mindata = -180.
                maxdata = 180.
                ax1.set_xlim((mindata, maxdata))
                ax1.set_xticklabels([])
                ax2.set_xlim((mindata, maxdata))

                if compCnt + 1 == len(resDict) or plotgmvise or plotindi:
                    ax2.set_xlabel('%s - Azimuth [Degree]' % (aziStr))

            else:
                print('Wrong mode')
                exit()

            if n == 0:
                if valmode == 'log':
                    if gm == 'pga':
                        label = '% g in log10'
                    elif gm == 'pgv':
                        label = 'cm/s in log10'
                    elif gm == 'pgd':
                        label = 'cm in log10'
                    elif gm == 'sigdur':
                        label = 's'
                    elif gm == 'ai':
                        label = 'cm/s in log10'
                    elif ':' in gm:
                        label = 'Log(Ratio)'
                    else:
                        label = 'UKN'
                    ax1.set_ylabel('%s [%s]' % (str(gm).upper(), label))
                    ax2.set_ylabel('Difference [log10]', color=ax2color)
                else:
                    if gm == 'pga':
                        label = '% g'
                    elif gm == 'pgv':
                        label = 'cm/s'
                    elif gm == 'pgd':
                        label = 'cm'
                    elif gm == 'sigdur':
                        label = 's'
                    elif gm == 'ai':
                        label = 'cm/s'
                    elif ':' in gm:
                        label = 'Ratio'
                    else:
                        label = 'UKN'
                    ax1.set_ylabel('%s [%s]' % (str(gm).upper(), label))
                    ax2.set_ylabel('Difference', color=ax2color)

            '''
            Plotting
            '''
            ## Primary plot, e.g. Azimuth or distance
            ax1.plot(xdata, data, marker=ax1marker, color=ax1color,
                linestyle='None', label='Predicted')
            ax1.plot(xdataRef, obsData, marker=obsmarker, color=obscolor,
                linestyle='None', label='Observed', fillstyle='none')

            ax1.set_xticklabels([])
            ax1.tick_params(which='both', direction='in', bottom=True, top=True,
                            left=True, right=True)
            
            ## verbindung zwischen den correspondieren punkten
            # for nn in range(len(ydata)):
            #     color = 'gray'
            #     ax1.plot([ydata[nn], ydata[nn]],
            #             [obsData[nn], data[nn]], ':', zorder=-100, linewidth=0.5,
            #             color=color)
            if valmode == 'log':
                ax1.set_ylim((valMin - num.log10(2.), valMax + num.log10(5.)))
            else:
                pass
                # ax1.set_ylim((10**valMin, 10**valMax + num.log10(5.)))

            ## Secondary plot, residual
            residuum = data - obsData
            nn_mean = num.mean(residuum)
            nn_std = num.std(residuum)
            ax2.plot(xdata, residuum, marker=ax2marker, color=ax2color,
                linestyle='None')#, label='log10-Res')

            if valmode == 'log':
               
                gmpe_std = 0.3
                gmpep = ax2.fill_between(num.linspace(mindata, maxdata),
                    gmpe_std, -gmpe_std, color='grey',
                    alpha=0.3, label='μ=0; σ=%.1f' % gmpe_std)

            mup = ax2.plot((mindata, maxdata), (nn_mean, nn_mean),
                color='black', marker='+', linestyle='--', label='μ-PWS', zorder=-2)
            sigmap = ax2.plot((mindata, maxdata), (nn_mean + nn_std, nn_mean + nn_std),
                color='black', marker='*', linestyle=':', label='σ-PWS', zorder=-2)
            ax2.plot((mindata, maxdata), (nn_mean -nn_std, nn_mean -nn_std),
                color='black', linestyle=':', zorder=-2)
            textstr = 'μ=%0.2f; σ=%0.2f' % (nn_mean, nn_std)
            ax2.annotate(textstr, (0.98, 0.98), xycoords='axes fraction',
                        ha='right', va='top', fontsize=12, color=ax2color)

            ax2.tick_params(which='both', direction='in', bottom=True, top=True,
                            left=True, right=True)

            # else:
            #     mup = ax2.plot((mindata, maxdata), (0, 0),
            #         color='black', linestyle='--', zorder=-2)
            # maxval = num.ceil((max(ydata) / 10.)) * 10.
            # minval = num.floor((min(ydata) / 10.)) * 10.
            # # print(minval, maxval)

            # binwidth = int((maxval - minval) / 20)
            # # print(binwidth)
            # datmus = []
            # resmus = []
            # for lim1 in range(int(minval), int(maxval), binwidth):
            #     lim2 = lim1 + 1.5 * binwidth
            #     lim1 = lim1 - 1.5 * binwidth

            #     dat = ydata[(ydata < lim2) & (ydata > lim1)]
            #     res = residuum[(ydata < lim2) & (ydata > lim1)]

            #     if len(dat) == 0:
            #         continue
            #     datmus.append(num.mean(dat))
            #     resmus.append(num.mean(res))

            # ax2.plot(datmus, resmus, color=ax2color, linestyle='-',
            #         linewidth=1, zorder=-100)

            # coef = num.polyfit(ydata, residuum, 1)
            # poly1d_fn = num.poly1d(coef)
            # dummyX = num.linspace(min(ydata), max(ydata))
            # ax2.plot(dummyX, poly1d_fn(dummyX), '--', color=ax2color)

            if valmode == 'log':
                tmpmax = max(1.1, max(residuum))
                tmpmin = min(-1.1, min(residuum))
            elif valmode in ['abs', 'true']:
                tmpmax = 1.1 * max(abs(residuum))
                tmpmin = -tmpmax

            ax2.set_ylim((tmpmin, tmpmax))
            # ax2.axhline(y=0, color=ax2color, linestyle=':', alpha=0.3)
            # ax2.axhline(y=1, color=ax2color, linestyle=':', alpha=0.3)
            # ax2.axhline(y=-1, color=ax2color, linestyle=':', alpha=0.3)

            ax2.tick_params(axis='y', colors=ax2color)

            # ax1.set_title('%s\n%s vs. %s' % (gm, comp, obsComp))
            if compCnt == 0:
                # if obsCont:
                #     ax1.set_title('%s vs %s_obs' % (comp, obsComp), fontsize=30)
                # else:
                # ax1.set_title('%s' % (comp), fontsize=30)
                ax1.set_title('%s' % (comp))

            for ax in [ax1, ax2]:
                # items = [ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()
                items = [ax.title, ax.xaxis.label]
                for item in (items):
                    item.set_fontsize(fontsize)

            fig.add_subplot(ax1)
            fig.add_subplot(ax2)

            if plotindi:
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax1.legend(lines + lines2, labels + labels2, prop={'size': 8})
                plt.suptitle(figtitle)
                plt.tight_layout()
                if savename != [] and savename != '':
                    fig.savefig('%s_%s_%s.png' % (savename, gm, comp))

        if plotgmvise and not plotindi:
            lines, labels = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines + lines2, labels + labels2)
            plt.suptitle(figtitle)
            outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
            if savename != [] and savename != '':
                fig.savefig('%s_%s.png' % (savename, gm))

    if not plotgmvise and not plotindi:
        # lines, labels = ax1.get_legend_handles_labels()
        # lines2, labels2 = ax2.get_legend_handles_labels()
        # ax1.legend(lines + lines2, labels + labels2)

        ax1.legend(fontsize=10)
        # from matplotlib.legend_handler import HandlerTuple
        from matplotlib.legend_handler import HandlerBase

        class AnyObjectHandler(HandlerBase):
            def create_artists(self, legend, orig_handle,
                               x0, y0, width, height, fontsize, trans):
                l1 = plt.Line2D([x0, y0 + width], [0.75 * height, 0.75 * height], 
                    linestyle=orig_handle[1], color=orig_handle[0])
                l2 = plt.Line2D([x0, y0 + width], [0.25 * height, 0.25 * height],
                    linestyle=orig_handle[3], color=orig_handle[2])
                return [l1, l2]
        if valmode == 'log':        
            ax2.legend([(mup[0].get_color(), mup[0].get_linestyle(),
                    sigmap[0].get_color(), sigmap[0].get_linestyle()),
                    gmpep], ['PWS μ, σ', 'μ=0; σ=0.3'],
                loc='upper left',
                handler_map={tuple: AnyObjectHandler()}, fontsize=10)

        plt.suptitle(figtitle)
        outer.tight_layout(fig, rect=[0., 0., 1., 0.98])
        if savename != [] and savename != '':
            fig.savefig('%s.png' % (savename))


def synth_peak_orientation_plot(staCont, imts=['pga', 'pgv'], savepath=None):

    source = staCont.refSource

    peaks = {}
    lats = []
    lons = []
    for sta in staCont.stations:
        for gm in imts:
            vals = []
            tracesE = staCont.stations[sta].components['E'].traces
            tracesN = staCont.stations[sta].components['N'].traces

            if 'pga' == gm:
                code = 'acc'

            elif 'pgv' == gm:
                code = 'vel'

            elif 'pgd' == gm:
                code = 'disp'

            else:
                print('%s not defined for that purpose.')
                continue

            trE = tracesE[code].copy()
            trN = tracesN[code].copy()
            tmin = max(trE.tmin, trN.tmin)
            tmax = min(trE.tmax, trN.tmax)
            trE.chop(tmin, tmax)
            trN.chop(tmin, tmax)

            dataH = abs(num.sqrt(trE.ydata**2 + trN.ydata**2))
            trH = trace.Trace(station='MAN', channel='H',
                        deltat=trE.deltat, tmin=tmin, ydata=dataH)

            maxidx = num.where(trH.ydata == max(trH.ydata))[0]
            # tmax = trH.get_xdata()[maxidx]
            # valE = max(trE.ydata, key=abs)
            # valN = max(trN.ydata, key=abs)

            valE = trE.ydata[maxidx]
            valN = trN.ydata[maxidx]

            vals = [valE, valN]

            if gm not in peaks:
                peaks[gm] = []
            peaks[gm].append(vals)

        lat = staCont.stations[sta].lat
        lon = staCont.stations[sta].lon
        lats.append(lat)
        lons.append(lon)

    #############
    ### Plotting
    #############
    for gm, ps in peaks.items():
        c = []
        dx = []
        dy = []
        for p in ps:
            E = p[0]
            N = p[1]
            c.append(num.sqrt(E**2 + N**2))
            dx.append(E)
            dy.append(N)

        lowerLat = min(lats) - 0.05
        upperLat = max(lats) + 0.05
        lowerLon = min(lons) - 0.05
        upperLon = max(lons) + 0.05

        plt.figure(figsize=(10, 10))
        m = basemap.Basemap(projection='gall',
                            llcrnrlat=lowerLat, urcrnrlat=upperLat,
                            llcrnrlon=lowerLon, urcrnrlon=upperLon,
                            resolution='l', epsg=3857)  # 1 - 4326, 2 - 3857
        try:
            m.drawcoastlines()
        except ValueError as e:
            print(e)

        m.drawcountries(linewidth=2.0)

        m.drawmeridians([lowerLon, source.lon, upperLon],
                            labels=[0, 0, 0, 1])  # , rotation=90.)
        m.drawparallels([lowerLat, source.lat, upperLat],
                            labels=[1, 0, 0, 0], rotation=90.)

        x, y = m(lons, lats)
        plt.quiver(x, y, dx, dy, c)

        if source.form == 'rectangular':

            x1, y1 = m(source.rupture['UR'][0], source.rupture['UR'][1])
            x2, y2 = m(source.rupture['UL'][0], source.rupture['UL'][1])
            x3, y3 = m(source.rupture['LL'][0], source.rupture['LL'][1])
            x4, y4 = m(source.rupture['LR'][0], source.rupture['LR'][1])

            m.plot([x1, x2], [y1, y2], 'k-', linewidth=2., zorder=10.)
            m.plot([x1, x4], [y1, y4], 'k:', zorder=10.)
            m.plot([x2, x3], [y2, y3], 'k:', zorder=10.)
            m.plot([x3, x4], [y3, y4], 'k:', zorder=10.)

        plt.title('Orientation of the maximum horizontal peak value; %s' % (gm))
        plt.tight_layout()
        plt.savefig('%smap_orientation_%s.png' % (savepath, gm))
        plt.close()


def waveform_comparison(outputdir, source, syntheticCont, realCont, channels,
                        appendix='', mode='acc', spectrum=False):
    import matplotlib.backends.backend_pdf
    import warnings
    
    plt.close('all')

    warnings.filterwarnings("ignore")

    if mode == 'acc':
        trtype = 'Acceleration'
        unit = '[m/s^2]'
    elif mode == 'vel':
        trtype = 'Velocity'
        unit = '[m/s]'
    elif mode == 'disp':
        trtype = 'Displacement'
        unit = '[m]'
    else:
        print('WRONG mode set')
        exit()
    if spectrum:
        print('Starting to plot %s spectra' % (trtype))
    else:
        print('Starting to plot %s' % (trtype))

    syntheticCont.calc_distances()
    realCont.calc_distances()
    synStas = syntheticCont.stations
    realStas = realCont.stations
    for sySta in synStas:
        for realSta in realStas:
            if synStas[sySta].lon == realStas[realSta].lon\
                    and synStas[sySta].lat == realStas[realSta].lat:

                plt.figure(figsize=(16, 9))
                ampmax = 0.
                for ii, comp in enumerate(channels):
                    synTr = synStas[sySta].components[comp].traces[mode].copy()
                    realTr = realStas[realSta].components[comp].traces[mode].copy()

                    ax = plt.subplot(len(channels), 1, ii + 1)

                    if spectrum:
                        tfade = 3
                        realVals, realFreqs = GMs.get_spectra(realTr.ydata, realTr.deltat, tfade)
                        synVals, synFreqs = GMs.get_spectra(synTr.ydata, synTr.deltat, tfade)

                        minT = source.time
                        ampmax = max(max(abs(realVals)), max(abs(synVals)),
                                    ampmax)
                        ax.loglog(realFreqs, abs(realVals),
                                    'k', label='Real')
                        ax.loglog(synFreqs, abs(synVals),
                                    'r', label='Synthetic')
                        ax.set_ylabel('%s - %s\nAmplitude %s'
                                % (realTr.channel, synTr.channel, unit))
                        ax.set_xlim((0.01, 1 / synTr.deltat))
                    else:
                        minT = source.time
                        ampmax = max(max(abs(synTr.ydata)), max(abs(realTr.ydata)),
                                    ampmax)
                        ax.plot(realTr.get_xdata() - minT, realTr.ydata,
                                    'k', label='Real')
                        ax.plot(synTr.get_xdata() - minT, synTr.ydata,
                                    'r', label='Synthetic')
                        ax.set_ylabel('%s - %s\nAmplitude %s'
                                % (realTr.channel, synTr.channel, unit))

                for nn in range(ii + 1):
                    ax = plt.subplot(len(channels), 1, nn + 1)
                    ax.set_ylim((-ampmax * 1.1, ampmax * 1.1))

                if spectrum:
                    plt.xlabel('Frequency [Hz]')
                else:
                    plt.xlabel('Time after Origin [s]')
                plt.suptitle('%s\n%s.%s.%s - %s.%s.%s\nHypodist - %0.1f'
                            % (trtype, realTr.network, realTr.location, realTr.station,
                            synTr.network, synTr.location, synTr.station,
                            realStas[realSta].rhypo))
                plt.legend()
                plt.tight_layout()
                plt.subplots_adjust(hspace=0)

    if spectrum:
        wvplotfile = outputdir + '/spectrum_comparison_%s_%s.pdf'\
                    % (trtype, appendix)
    else:
        wvplotfile = outputdir + '/waveform_comparison_%s_%s.pdf'\
                    % (trtype, appendix)
    pdf = matplotlib.backends.backend_pdf.PdfPages("%s" % (wvplotfile))
    for fig in range(1, plt.gcf().number + 1):
        pdf.savefig(fig)
    pdf.close()
    plt.close('all')

    print('Finished')
