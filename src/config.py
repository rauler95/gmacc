# GPLv3
#
# The Developers, 21st Century

from pyrocko.guts import Float, String, Bool, List, Int, load, StringChoice, Object

from ewrica.module import ModuleConfig


# import logging
# logger = logging.getLogger('ewrica.gm.config')


# def get_config(*args, **kwargs):
#     return SyntheticDatabase(*args, **kwargs)


class ObservationalData(ModuleConfig):
    config_path = String.T(
        default='',
        help='Path to the ... configuration file.')

    indir = String.T(
        default='',
        help='Directory where directories of individual event flat files can be found.')

    outdir = String.T(
        default='',
        help='Directory where the output is stored in.')

    subdirs = List.T(
        String.T(),
        optional=True,
        help='List of subdirectories with flatfiles/csv in it. E.g. form different sources, as ESM, CESMD or Japan.')

    comps = List.T(
        String.T(),
        optional=True,
        help='List of components targets.')

    input_gms = List.T(
        String.T(),
        optional=True,
        help='List of ground motion as input.')

    target_gms = List.T(
        String.T(),
        optional=True,
        help='List of ground motion targets.')

    magnitude = Float.T(
        optional=True,
        default=None,
        help='Magnitude for preprocessing.')

    only_japan = Bool.T(
        optional=True,
        default=False,
        help='Bool if only earthquake are taken that were in greater Japanese region.')

    include_japan = Bool.T(
        optional=True,
        default=False,
        help='Bool if a column is added, that states if earthquake was in greater Japanese region.')

    include_site = Bool.T(
        optional=True,
        default=False,
        help='Bool if site columns are added.')

    include_source = Bool.T(
        optional=True,
        default=False,
        help='Bool if source column are added.')

    include_landform = Bool.T(
        optional=True,
        default=False,
        help='Bool if landforms are added.')

    categorial_to_numeric = Bool.T(
        optional=True,
        default=False,
        help='Bool if Tect_region, Lithology_hm12 and landform (all as categorial columns) should be converted to multiple columns with numerical values.')

    categorial_most = Bool.T(
        optional=True,
        default=False,
        help='Bool if only the column with the most members is selected as numerical value (for Tect_region, Lithology_hm12 and landform (all as categorial columns)).')

    learn_on_residuals = Bool.T(
        optional=True,
        default=False,
        help='Bool if targets should be absolute values (False) or residuals (True). For residuals, the corresponding low-frequent parameter is subtracted from.')

    normalize_frequencies = Bool.T(
        optional=True,
        default=True,
        help='Bool if acceleration frequencies are normed together (True) or seperate (False)')

    def get_config(self):
        '''
        Reads config from a YAML file.
        '''
        print(self.config_path)
        config = load(filename=self.config_path)
        self.config__ = config

        return config


class NeuralNetwork(ModuleConfig):
    config_path = String.T(
        default='',
        help='Path to the synthetic database configuration file.')

    indir = String.T(
        default='',
        help='Directory where the data can be found.')

    filecore = String.T(
        default='data',
        help='Filecore in indir, with the corresponding _xEval, xTest etc.')

    outdir = String.T(
        default='',
        help='Directory where the output is stored in.')

    modeldir = String.T(
        optional=True,
        help='Directory where one single model is stored for further use. (as MODEL.h5)')

    maxepochnum = Int.T(
        default=100,
        help='The number of maximal epochs.')

    minepochnum = Int.T(
        default=10,
        help='The number of minimal epochs.')

    validepochnum = Int.T(
        default=10,
        help='The number of epochs to validate.')

    device = StringChoice.T(
        choices=['cpu', 'gpu'],
        default='cpu',
        help='Computation on either on cpu or gpu.')

    targetmode = StringChoice.T(
        choices=['multi', 'single'],
        default='multi',
        help='Whether to include all targets in the last layer (multi) or calculate on NN per target (single).')

    dropout = Float.T(
        default=0.,
        help='Fraction of nodes per layer that are dropped out.')

    activations = List.T(
        String.T(),
        default=['elu'],
        help='List of activation functions to iterate through. E.g. sigmoid, tanh, elu, relu.')

    optimizers = List.T(
        String.T(),
        default=['adam'],
        help='List of optimizer to iterate through. E.g. adam, RMS, sgd.')

    learningtypes = List.T(
        String.T(),
        default=['default'],
        help='List of learning types to iterate through. E.g. default, own.')

    batchsizes = List.T(
        Float.T(),
        default=[0.1],
        help='Fractions of batchsize relative to input length to iterate through.')

    hiddenlayers = List.T(
        List.T(),
        default=[[10, 10]],
        help='A list of list, each list consists of a number corresponding to nodes and the number of numbers to the number of layers')

    def get_config(self):
        '''
        Reads config from a YAML file.
        '''
        print(self.config_path)
        config = load(filename=self.config_path)
        self.config__ = config

        return config


class SyntheticDatabase(ModuleConfig):
    config_path = String.T(
        default='',
        help='Path to the synthetic database configuration file.')

    outdir = String.T(
        default='',
        help='Directory where the output is stored in.')

    sourcemode = StringChoice.T(
        choices=['MT', 'RS'],
        default='MT',
        help='For which source type MT or RectangularSource the processing is done.')

    comps = List.T(
        String.T(),
        default=['Z', 'E', 'N'],
        optional=True,
        help='List of components to calculate the ground motions values for; e.g. Z, N, E')

    gmpes = List.T(
        String.T(),
        default=[],
        optional=True,
        help='List of GMPEs to use for the ground motion calculation. Needs the exact name as in OpenQuake.')

    imts = List.T(
        String.T(),
        default=['pga', 'pgv', 'pgd'],
        optional=True,
        help='Intensity measures to calculate.')

    freqs = List.T(
        String.T(),
        default=[],
        optional=True,
        help='Frequencies of spectral acceleration or Fourier spectrum to calculate.')

    gf = String.T(
        default='',
        help='Path to the Green\'s function store.', optional=True)

    rotd100 = Bool.T(
        default=True,
        help='Enables the option to calculate the vector-sum of both horizontal, aka the RotD100 (BooreRef).')

    srccnt = Int.T(
        default=100,
        help='The number of created sources.')

    mappoints = Int.T(
        default=10,
        help='SQRT of Number of points/locations to calculate imts for.')

    mapextent = List.T(
        Float.T(),
        default=[1., 1.],
        help='List of two numbers which define the map size in degree around the hypocenter.')

    mapmode = StringChoice.T(
        choices=['rectangular', 'circular', 'random'],
        default='rectangular',
        help='Mode of the map to be calculated for.')

    mp = Int.T(
        default=0,
        help='Enables multiprocessing for the number of cores, if value > 1.')

    append = Bool.T(
        default=False,
        help='Enables \'append\' mode. If database already exists, continue till srccnt.')

    def get_config(self):
        '''
        Reads config from a YAML file.
        '''
        print(self.config_path)
        config = load(filename=self.config_path)
        self.config__ = config

        return config
