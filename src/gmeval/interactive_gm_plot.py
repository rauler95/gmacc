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


def parser_func():
	parser = argparse.ArgumentParser(description='''
		Reads a geojson file and converts it to an interactive map
		''',
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument('-d', '--datadir', dest='datadir',
					nargs=1, metavar='dirPATH',
					help='''
					Path to the directory where the geojson can be found.
					''')

	args = parser.parse_args()
	print(args)

	return args


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


def get_dataset(src, gm, comp, target='mean'):

	rawlons = []
	rawlats = []
	dfMain = {}
	dfSide = {}
	for sr in src['features']:
		rlon, rlat = sr['geometry']['coordinates']
		rawlons.append(rlon)
		rawlats.append(rlat)
		for var in ['mean', 'wmean', 'std', 'wstd']:
			val = sr['properties']['%s_%s_%s' % (comp, gm, var)]
			if var not in dfMain:
				dfMain[var] = []
			dfMain[var].append(val)

			if '%s_%s_%s' % (comp, gm, var) not in dfSide:
				dfSide['%s_%s_%s' % (comp, gm, var)] = []
			dfSide['%s_%s_%s' % (comp, gm, var)].append(val)
		if '%s_%s' % (comp, gm) not in dfSide:
			dfSide['%s_%s' % (comp, gm)] = []
		dfSide['%s_%s' % (comp, gm)].append(num.array(sr['properties']['%s_%s' % (comp, gm)]))

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

	dfMain = pd.DataFrame(dfMain)
	dfMain['show'] = dfMain[target]
	dfMain['lon'] = xlons
	dfMain['lat'] = ylats
	dfMain['rawlon'] = rawlons
	dfMain['rawlat'] = rawlats
	dfSide = pd.DataFrame(dfSide)

	return ColumnDataSource(data=dfMain), ColumnDataSource(data=dfSide)


def make_main_plot(source, title):
	tile_provider = get_provider(Vendors.CARTODBPOSITRON)

	maxlon = max([max(sublist) for sublist in source.data['lon']])
	minlon = min([min(sublist) for sublist in source.data['lon']])
	maxlat = max([max(sublist) for sublist in source.data['lat']])
	minlat = min([min(sublist) for sublist in source.data['lat']])

	# range bounds supplied in web mercator coordinates
	plot = figure(plot_width=1000, plot_height=600,
				tools="pan, wheel_zoom, box_zoom, reset",
				toolbar_location='right',
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

	plot.add_layout(color_bar, 'below')

	my_hover = HoverTool()
	my_hover.tooltips = [
			("index", "$index"),
			('Lon', '@rawlon'),
			('Lat', '@rawlat'),
			('mean', '@mean'),
			('std', '@std'),
			('wmean', '@wmean'),
			('wstd', '@wstd')]
	# my_hover.point_policy = 'snap_to_data'
	plot.add_tools(my_hover, TapTool(renderers=[patch]))

	plot.xaxis.axis_label = 'Longitude'
	plot.yaxis.axis_label = "Latitude"
	plot.axis.axis_label_text_font_style = "bold"
	plot.grid.grid_line_alpha = 0.3

	return plot


def update_plot(attrname, old, new):
	gm = gmSelect.value
	comp = compSelect.value
	target = targetSelect.value

	srcMain, srcSide = get_dataset(data, gm, comp, target)
	sourceMain.data.update(srcMain.data)
	plot.title.text = '%s_%s_%s' % (comp, gm, target)

	if target in ['std', 'wstd']:
		color_bar.color_mapper.palette = coolwarm_palette
		color_bar.color_mapper.low = srcMain.data[target].min()
		color_bar.color_mapper.high = srcMain.data[target].max()
		color_bar.title = 'Std'

	else:
		color_bar.color_mapper.palette = gm_palette
		color_bar.color_mapper.low = srcMain.data['mean'].min()
		color_bar.color_mapper.high = srcMain.data['mean'].max()
		color_bar.title = 'GM [log10]'

	srcHist, srcStats = get_source_side(srcSide, gidx)
	sourceHist.data.update(srcHist.data)
	sourceStats.data.update(srcStats.data)


###

def gaussian(x, mu, sig):
	return num.exp(-num.power(x - mu, 2.) / (2 * num.power(sig, 2.)))


def get_source_side(source, idx):
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
	statsDf['ys'] = num.linspace(-2, 0, len(gaussX))
	statList = list(statData.index)
	statsDf['mean'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_mean')] * len(gaussX))
	statsDf['wmean'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_wmean')] * len(gaussX))
	statsDf['std'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_std')] * len(gaussX))
	statsDf['wstd'] = num.array([statData[stat] for stat in statList
							if stat.endswith('_wstd')] * len(gaussX))

	statsDf['mean-std'] = statsDf['mean'] - statsDf['std']
	statsDf['mean+std'] = statsDf['mean'] + statsDf['std']

	statsDf['wmean-wstd'] = statsDf['wmean'] - statsDf['wstd']
	statsDf['wmean+wstd'] = statsDf['wmean'] + statsDf['wstd']

	statsDf['gaussX'] = gaussX
	statsDf['gaussY'] = max(histDf['column']) * gaussian(gaussX,
											statsDf['mean'], statsDf['std'])
	statsDf['wgaussY'] = max(histDf['column']) * gaussian(gaussX,
											statsDf['wmean'], statsDf['wstd'])
	return ColumnDataSource(data=histDf), ColumnDataSource(data=statsDf)


def make_hist_plot(sourceFin, sourceStat, title, index):
	plotHist = figure(plot_width=1000, plot_height=400, title=title,
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
				line_width=1, color='red', legend_label='Gaussian')

	plotHist.line(x='wmean', y='ys', source=sourceStat,
					line_width=3, color='green', legend_label='WMean')
	plotHist.harea(x1='wmean+wstd', x2='wmean-wstd', y='ys', source=sourceStat,
					color='green', alpha=0.2, legend_label='WStd')
	plotHist.line(x='gaussX', y='wgaussY', source=sourceStat,
					line_width=1, color='green', legend_label='wGaussian')

	hover = HoverTool(names=['hist'],
						tooltips=[('Interval', '@interval'),
								('Count', str("@" + 'column'))])
	plotHist.add_tools(hover)

	plotHist.legend.location = "top_left"
	plotHist.legend.click_policy = "hide"

	return plotHist


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

		srcMain, srcSide = get_dataset(data, gm, comp, target)
		srcHist, srcStats = get_source_side(srcSide, index)
		sourceHist.data.update(srcHist.data)
		sourceStats.data.update(srcStats.data)

		global gidx
		gidx = index


######

args = parser_func()

filename = 'misc/predicted_data_GMPE.geojson'
file = args.datadir[0] # + '/' + filename

m_coolwarm_rgb = (255 * cm.Reds(range(256))).astype('int')
coolwarm_palette = [RGB(*tuple(rgb)).to_hex() for rgb in m_coolwarm_rgb]

# gm_rgb = (255 * cm.gist_stern_r(range(256))).astype('int')
gm_rgb = (255 * cm.afmhot_r(range(256))).astype('int')
gm_rgb = list(gm_rgb)
# gm_rgb.append(num.array([255, 255, 255, 1]))
gm_rgb[0] = num.array([250, 250, 250, 255])
# print(gm_rgb)
gm_palette = [RGB(*tuple(rgb)).to_hex() for rgb in gm_rgb]

data = load(open(file))

# print(data)
for sr in data['features']:
	print(sr['properties'].keys())
	chagms = [x for x in sr['properties'].keys() if len(x.rsplit('_')) < 3]
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

targets = {'mean': 'mean',
		'wmean': 'wmean',
		'std': 'std',
		'wstd': 'wstd'}

gmSelect = Select(value=gm, title='GM', width=100,
				options=list(gms.keys()))
compSelect = Select(value=comp, title='Component', width=200,
					options=list(comps.keys()))
targetSelect = Select(value=target, title='Targets', width=100,
					options=list(targets.keys()))

sourceMain, sourceSide = get_dataset(data, gm, comp, target)

color_mapper = LinearColorMapper(palette=gm_palette,
	low=sourceMain.data['mean'].min(),
	high=sourceMain.data['mean'].max())
color_bar = ColorBar(color_mapper=color_mapper,
		title='GM [log10]', orientation='horizontal',
		label_standoff=10, border_line_color=None, location='bottom_center')

plot = make_main_plot(sourceMain, '%s_%s_%s' % (comp, gm, target))

sourceHist, sourceStats = get_source_side(sourceSide, gidx)
plotHist = make_hist_plot(sourceHist, sourceStats, '', gidx)
plotHist.visible = False

gmSelect.on_change('value', update_plot)
compSelect.on_change('value', update_plot)
targetSelect.on_change('value', update_plot)

sourceMain.selected.on_change('indices', update_hist_plot)

controls = row(gmSelect, compSelect, targetSelect)

curdoc().add_root(column(column(plot, controls), plotHist))
curdoc().title = "Interactive Ground-Motion Map"
