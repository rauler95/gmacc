from bokeh.plotting import figure
from bokeh.models import ColorBar, ColumnDataSource, HoverTool,\
							TapTool, Select, LinearColorMapper, Patches
from bokeh.io import curdoc
from bokeh.layouts import column, row
from bokeh.tile_providers import get_provider, Vendors
from bokeh.colors import RGB

from geojson import load
import numpy as num
import pandas as pd
import geopandas as gpd
from matplotlib import cm
import argparse
import matplotlib.colors


def wgs84_to_web_mercator(lons, lats):
	k = 6378137
	x = lons * (k * num.pi / 180.0)
	y = num.log(num.tan((90 + lats) * num.pi / 360.0)) * k

	return x, y


def get_dataset_gdf(src, gm, comp, target='mean'):
	rawlons = num.array(src.geometry.x)
	rawlats = num.array(src.geometry.y)

	lons, lats = wgs84_to_web_mercator(rawlons, rawlats)

	lonsSet = num.sort(list(set(lons)))
	latsSet = num.sort(list(set(lats)))

	dlon = abs(lonsSet[1] - lonsSet[0]) / 2
	dlat = abs(latsSet[1] - latsSet[0]) / 2

	xlons = list(num.array([lons + dlon, lons + dlon,
							lons - dlon, lons - dlon]).T)
	ylats = list(num.array([lats + dlat, lats - dlat,
							lats - dlat, lats + dlat]).T)

	select_cols = [col for col in src.columns if gm in col]
	selectSide = [col for col in select_cols if comp in col]

	appendix = ['mean', 'std']
	selectMain = [col for col in selectSide for app in appendix if app in col]

	dfMain = src[selectMain].copy(deep=True)
	dfMain = dfMain.rename(columns={
				'%s_%s_mean' % (comp, gm): 'mean',
				'%s_%s_std' % (comp, gm): 'std',
				'%s_%s_wmean' % (comp, gm): 'wmean',
				'%s_%s_wstd' % (comp, gm): 'wstd'})

	dfMain['show'] = dfMain[target]
	dfMain['lon'] = xlons
	dfMain['lat'] = ylats
	dfMain['rawlon'] = rawlons
	dfMain['rawlat'] = rawlats

	dfSide = src[selectSide].copy(deep=True)

	return ColumnDataSource(data=dfMain), ColumnDataSource(data=dfSide)


def get_dataset(data, gm, comp, thresholds, target='mean'):

	rawlons = []
	rawlats = []
	dfMain = {}
	dfSide = {}
	for pos in data.keys():
		rlon = data[pos]['lon']
		rlat = data[pos]['lat']
		rawlons.append(rlon)
		rawlats.append(rlat)
		for var in ['mean', 'wmean', 'std', 'wstd']:
			val = data[pos]['%s_%s_%s' % (comp, gm, var)]
			if var not in dfMain:
				dfMain[var] = []
			dfMain[var].append(val)

			if '%s_%s_%s' % (comp, gm, var) not in dfSide:
				dfSide['%s_%s_%s' % (comp, gm, var)] = []
			dfSide['%s_%s_%s' % (comp, gm, var)].append(val)
		if '%s_%s' % (comp, gm) not in dfSide:
			dfSide['%s_%s' % (comp, gm)] = []
		dfSide['%s_%s' % (comp, gm)].append(num.array(data[pos]['%s_%s' % (comp, gm)]))

	rawlons = num.array(rawlons)
	rawlats = num.array(rawlats)

	lons, lats = wgs84_to_web_mercator(rawlons, rawlats)

	lonsSet = num.sort(list(set(lons)))
	latsSet = num.sort(list(set(lats)))

	dlon = abs(lonsSet[1] - lonsSet[0]) / 2
	dlat = abs(latsSet[1] - latsSet[0]) / 2

	xlons = list(num.array([lons + dlon, lons + dlon,
							lons - dlon, lons - dlon]).T)
	ylats = list(num.array([lats + dlat, lats - dlat,
							lats - dlat, lats + dlat]).T)

	threshold = thresholds[gm]
	dfMain['prob'] = num.sum(num.array(dfSide['%s_%s' % (comp, gm)]) > threshold, axis=1) / len(dfSide['%s_%s' % (comp, gm)][0])
	dfMain = pd.DataFrame(dfMain)
	dfMain['show'] = dfMain[target]
	dfMain['lon'] = xlons
	dfMain['lat'] = ylats
	dfMain['rawlon'] = rawlons
	dfMain['rawlat'] = rawlats
	dfSide = pd.DataFrame(dfSide)

	return ColumnDataSource(data=dfMain), ColumnDataSource(data=dfSide)


def make_main_plot(source, title, color_mapper, color_bar):
	tile_provider = get_provider(Vendors.CARTODBPOSITRON)

	maxlon = max([max(sublist) for sublist in source.data['lon']])
	minlon = min([min(sublist) for sublist in source.data['lon']])
	maxlat = max([max(sublist) for sublist in source.data['lat']])
	minlat = min([min(sublist) for sublist in source.data['lat']])

	# range bounds supplied in web mercator coordinates
	plot = figure(plot_width=800, plot_height=800,
				tools="pan, wheel_zoom, box_zoom, reset",
				toolbar_location='left',
				x_range=(minlon, maxlon), y_range=(minlat, maxlat),
				x_axis_type="mercator", y_axis_type="mercator")
	plot.add_tile(tile_provider)

	plot.title.text = title

	patch = plot.patches(xs='lon', ys='lat', source=source,
		color={'field': 'show', 'transform': color_mapper},
		fill_alpha=0.7,
		hover_line_color="black",
		hover_fill_alpha=1.0,
		selection_fill_alpha=1.0,
		selection_line_alpha=1.0,
		selection_line_color='black',
		nonselection_fill_alpha=0.7,
		nonselection_line_alpha=1)

	# plot.add_layout(color_bar, 'below')
	plot.add_layout(color_bar, 'right')

	my_hover = HoverTool()
	my_hover.tooltips = [
			("index", "$index"),
			('Lon', '@rawlon'),
			('Lat', '@rawlat'),
			('mean', '@mean'),
			('std', '@std'),
			('wmean', '@wmean'),
			('wstd', '@wstd'),
			('prob', '@prob')
			]
	# my_hover.point_policy = 'snap_to_data'
	plot.add_tools(my_hover, TapTool(renderers=[patch]))

	plot.grid.grid_line_alpha = 0.3
	plot.xaxis.axis_label = 'Longitude'
	plot.yaxis.axis_label = "Latitude"
	plot.axis.axis_label_text_font_style = "bold"
	plot.axis.axis_label_text_font_size = '20pt'
	plot.title.text_font_size = '20pt'
	plot.yaxis.major_label_text_font_size = '15pt'
	plot.xaxis.major_label_text_font_size = '15pt'

	return plot


def gaussian(x, mu, sig):
	return num.exp(-num.power(x - mu, 2.) / (2 * num.power(sig, 2.)))


def get_source_side(source, idx, thresholds):
	d = pd.DataFrame(source.data).drop(columns='index')
	appendix = ['mean', 'std']
	statCols = [col for col in d.columns for app in appendix if app in col]
	statData = d[statCols]
	histData = d.drop(columns=statCols)
	histData = list(histData.iloc[idx])
	statData = statData.iloc[idx]
	hist, edges = num.histogram(histData, bins=15)#, range=(lowval, highval))
	histDf = pd.DataFrame({'column': hist,
						"left": edges[:-1],
						"right": edges[1:]})
	histDf["interval"] = ["%0.2f to %0.2f" % (left, right)
			for left, right in zip(histDf["left"], histDf["right"])]

	histDf = pd.DataFrame(histDf)

	#### Data for statistics
	statsDf = {}
	gaussX = num.linspace(min(edges), max(edges))
	lenfac = len(gaussX)
	statsDf['ys'] = num.linspace(-max(hist)*0.1, 0, len(gaussX))
	statList = list(statData.index)
	statsDf['mean'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_mean')] * lenfac)
	statsDf['wmean'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_wmean')] * lenfac)
	statsDf['std'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_std')] * lenfac)
	statsDf['wstd'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_wstd')] * lenfac)

	statsDf['mean-std'] = statsDf['mean'] - statsDf['std']
	statsDf['mean+std'] = statsDf['mean'] + statsDf['std']

	statsDf['wmean-wstd'] = statsDf['wmean'] - statsDf['wstd']
	statsDf['wmean+wstd'] = statsDf['wmean'] + statsDf['wstd']

	statsDf['gaussX'] = gaussX
	statsDf['gaussY'] = max(histDf['column']) * gaussian(gaussX,
											statsDf['mean'], statsDf['std'])
	statsDf['wgaussY'] = max(histDf['column']) * gaussian(gaussX,
											statsDf['wmean'], statsDf['wstd'])

	gm = statCols[0].rsplit('_')[1]
	statsDf['threshold'] = num.array([thresholds[gm]] * lenfac)

	return ColumnDataSource(data=histDf), ColumnDataSource(data=statsDf)


def make_hist_plot(sourceFin, sourceStat, title, index):
	# print(sourceFin)
	# print()
	# print(sourceStat)
	plotHist = figure(plot_width=800, plot_height=800, title=title,
						x_axis_label="Val", y_axis_label="Count")

	plotHist.quad(bottom=0, top='column', left="left", fill_color='black',
		line_color='black',
		right="right", source=sourceFin, fill_alpha=0.7, hover_fill_alpha=1.0,
		name='hist')

	plotHist.line(x='mean', y='ys', source=sourceStat,
					line_width=3, color='red', legend_label='Mean')
	plotHist.harea(x1='mean+std', x2='mean-std', y='ys', source=sourceStat,
					color='red', alpha=0.2, legend_label='Std')
	plotHist.line(x='gaussX', y='gaussY', source=sourceStat,
				line_width=1, color='red', line_dash='dashed',
				legend_label='Gaussian')

	plotHist.line(x='wmean', y='ys', source=sourceStat,
					line_width=3, color='green', legend_label='WMean')
	plotHist.harea(x1='wmean+wstd', x2='wmean-wstd', y='ys', source=sourceStat,
					color='green', alpha=0.2, legend_label='WStd')
	plotHist.line(x='gaussX', y='wgaussY', source=sourceStat,
					line_width=1, color='green', line_dash='dashed',
					legend_label='wGaussian')

	plotHist.line(x='threshold', y='wgaussY', source=sourceStat,
					line_width=1, color='black', line_dash='solid',
					legend_label='threshold')

	hover = HoverTool(names=['hist'],
						tooltips=[('Interval', '@interval'),
								('Count', str("@" + 'column'))])
	plotHist.add_tools(hover)

	# plotHist.legend.location = "top_left"
	plotHist.legend.location = "top_right"
	plotHist.legend.click_policy = "hide"
	plotHist.axis.axis_label_text_font_size = '20pt'
	plotHist.title.text_font_size = '20pt'
	plotHist.yaxis.major_label_text_font_size = '15pt'
	plotHist.xaxis.major_label_text_font_size = '15pt'

	return plotHist


######
def plot_interactive_map_read(filename):
	data = load(open(filename))

	plot_interactive_map(data)


def plot_interactive_map(data):

	def update_plot(attrname, old, new):
		gm = gmSelect.value
		comp = compSelect.value
		target = targetSelect.value

		srcMain, srcSide = get_dataset(data, gm, comp, thresholds, target)
		sourceMain.data.update(srcMain.data)
		plot.title.text = '%s_%s_%s' % (comp, gm, target)

		if target in ['std', 'wstd']:
			color_bar.color_mapper.palette = coolwarm_palette
			color_bar.color_mapper.low = srcMain.data[target].min()
			color_bar.color_mapper.high = srcMain.data[target].max()
			color_bar.title = 'Std'

		elif target in ['prob']:
			color_bar.color_mapper.palette = gm_palette
			color_bar.color_mapper.low = 0
			color_bar.color_mapper.high = 1
			color_bar.title = 'Probability of Exceedance'

		else:
			color_bar.color_mapper.palette = gm_palette
			color_bar.color_mapper.low = srcMain.data['mean'].min()
			color_bar.color_mapper.high = srcMain.data['mean'].max()
			color_bar.title = 'GM log'

		srcHist, srcStats = get_source_side(srcSide, gidx, thresholds)
		sourceHist.data.update(srcHist.data)
		sourceStats.data.update(srcStats.data)

	def update_hist_plot(attr, old, new):
		try:
			index = new[0]
			flag = True
		except IndexError:
			sourceHist.data.update([])
			sourceStats.data.update([])
			flag = False
			plotHist.visible = False

		if flag:
			plotHist.visible = True
			plotHist.title.text = str(index)

			gm = gmSelect.value
			comp = compSelect.value
			target = targetSelect.value

			srcMain, srcSide = get_dataset(data, gm, comp, thresholds, target)
			srcHist, srcStats = get_source_side(srcSide, index, thresholds)
			sourceHist.data.update(srcHist.data)
			sourceStats.data.update(srcStats.data)

			plotHist.xaxis.axis_label = '%s-%s (log10)' % (comp, gm)

			global gidx
			gidx = index

	m_coolwarm_rgb = (255 * cm.Reds(range(256))).astype('int')
	coolwarm_palette = [RGB(*tuple(rgb)).to_hex() for rgb in m_coolwarm_rgb]

	# gm_rgb = (255 * cm.gist_stern_r(range(256))).astype('int')
	gm_rgb = (255 * cm.afmhot_r(range(256))).astype('int')
	gm_rgb = list(gm_rgb)
	# gm_rgb.append(num.array([255, 255, 255, 1]))
	gm_rgb[0] = num.array([250, 250, 250, 255])
	# print(gm_rgb)
	gm_palette = [RGB(*tuple(rgb)).to_hex() for rgb in gm_rgb]

	for sr in data.keys():
		chagms = [x for x in data[sr].keys() if len(x.rsplit('_')) == 2]
		break

	gms = {}
	comps = {}
	for chagm in chagms:
		comp, gm = chagm.rsplit('_')
		gms[str(gm)] = str(gm)
		comps[str(comp)] = str(comp)

	target = 'mean'
	global gidx
	gidx = 0

	# gm = 'pgd'
	# comp = 'Z'

	targets = {'mean': 'mean',
			'wmean': 'wmean',
			'std': 'std',
			'wstd': 'wstd',
			'prob': 'prob'}

	thresholds = {
		'pga': num.log10(25), # log10(in %g) 
		'pgv': num.log10(10), # log10(in cm/s)
		'pgd': num.log10(1), # log10(in cm)
		}

	gmSelect = Select(value=gm, title='GM', width=70,
					options=list(gms.keys()))
	compSelect = Select(value=comp, title='Component', width=70,
						options=list(comps.keys()))
	targetSelect = Select(value=target, title='Targets', width=70,
						options=list(targets.keys()))

	sourceMain, sourceSide = get_dataset(data, gm, comp, thresholds, target)

	color_mapper = LinearColorMapper(palette=gm_palette,
		low=sourceMain.data['mean'].min(),
		high=sourceMain.data['mean'].max())
	color_bar = ColorBar(color_mapper=color_mapper,
			title='GM log', orientation='vertical',
			label_standoff=5, border_line_color=None, location='bottom_center')

	plot = make_main_plot(sourceMain, '%s_%s_%s' % (comp, gm, target),
		color_mapper, color_bar)

	sourceHist, sourceStats = get_source_side(sourceSide, gidx, thresholds)
	plotHist = make_hist_plot(sourceHist, sourceStats, str(gidx), gidx)
	plotHist.visible = False

	gmSelect.on_change('value', update_plot)
	compSelect.on_change('value', update_plot)
	targetSelect.on_change('value', update_plot)

	sourceMain.selected.on_change('indices', update_hist_plot)

	# controls = row(gmSelect, compSelect, targetSelect)
	controls = column(gmSelect, compSelect, targetSelect)

	# curdoc().add_root(column(column(plot, controls), plotHist))
	# curdoc().add_root(row(column(plot, controls), plotHist))
	curdoc().add_root(row(row(plot, controls), plotHist))
	curdoc().title = "Interactive Ground-Motion Map"


# file = '/home/lehmann/src/ewricagm/ewricagm/examples/Samos2020/MTSource/predicted_data_NN.json'
file = '/home/lehmann/src/ewricagm/ewricagm/examples/Norcia2016/MTSource/predicted_data_NN.json'
# file = '/home/lehmann/src/ewricagm/ewricagm/examples/Norcia2016/MTSource/predicted_data_Pyrocko.json'
plot_interactive_map_read(file)