# Creation of an own synthetic database and a trained Neural Network

## Before starting
- You need to make sure to have a Green's Function store available on your machine.
	- either calculate your own: https://pyrocko.org/docs/current/apps/fomosto/
	- or download from: https://greens-mill.pyrocko.org/


## Database generation

To create a own synthetic database execute:

*python3 synthetic_database_generation.py*

to modify the parameters have a look at the config file:

*synthetic_database_config.yaml*

here change the parameter `gf` to the path were your Green's Function store is, for more details look in `gmacc/src/config.py`. Most important is the sourcemode: MT or RS (RS will take much more time to calculate), srccnt the number of different sources and mappoints how many points per source should be considered for the waveform calculation.
In the declared directory a new one is produced (named after the sourcemode) which consists of one file, per default, `database.csv` is generated, containing all information, and one directory storing all calculated waveforms.

HEADER: Description
evID: string
magnitude: Mw
ev_lat: degree
ev_lon: degree
ev_depth: km
strike: degree
dip: degree
rake: degree
src_duration: s
width: km
length: km
nucleation_x/y: 0-1 position fraction of width/length (see pyrocko documentation)
ns: network.station
st_lat: degree
st_lon: degree
azimuth: degree (to north)
rhypo, rjb, ry0, rx, rrup: km
rup_azimuth, center_azimuth: degree

Available GMs (not all make sense for synthetic waveforms):
pga: log10 %g
pgv: log10 cm/s
pgd: log10 cm
ai: arias intensity, log 10 m/s
vi: integrated veloctiy, log10 m
sigdur: significant duration, s
SA: spectral acceleration in log 10

## Neural Network learning

To train a neural network, first the database needs to be preprocessed, e.g. split in train and test datasets, normalized etc. To execute this preprocessing:

*python3 synthetic_database_preprocessing.py*

this file also read content from *synthetic_database_config.yaml*, but is mainly modified within the file itself. Here, most importantly are the options, `target consists of gms and comps` which ground motion parameter should be considered by the preprocessing. 
After the execution, a subdirectory is produced (defined within the file) containing of x-(input) and y-(output)-train, test and eval (for the default settings, empty) files additionally to a scalingdict, which is important for the normalization.

To finally train the NN execute:

*python3 synthetic_database_nn.py*

to modify the parameters have a look at the config file:

*synthetic_database_config_nn.yaml*

for more details look in `gmacc/src/config.py`. Here, you will have to play around with all kinds of settings.

