# GPLv3
#
# The Developers, 21st Century

from pyrocko.guts import Float, String, Bool, List, Int, load, StringChoice, Object


# def get_config(*args, **kwargs):
#     return SyntheticDatabase(*args, **kwargs)


class ModuleConfig(Object):
    '''Configuration of the module run
    '''
    run = Bool.T(
        default=True,
        help='Set to True, if the module shall be runned, else False')


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

    dropouts = List.T(
        Float.T(),
        default=[0.],
        help='List of fraction of nodes per layer that are dropped out. Floats between 0 and 1.')

    losses = List.T(
        String.T(),
        default=['mse'],
        help='List of loss functions to iterate through, E.g. mse, mae, CorrelationCoefficient, CosineSimilarity')

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
        help='''
        A list of list, each list consists of a number corresponding to nodes
        and the number of numbers to the number of layers.
        It is also possible to define the hidderlayers depending on the input-
        size. For that, use an additional 'n',
        e.g. '5n', 5 times the number of input features as node.
        ''')
    def get_config(self):
        '''
        Reads config from a YAML file.
        '''
        print(self.config_path)
        config = load(filename=self.config_path)
        self.config__ = config

        return config


# class SyntheticDatabase(ModuleConfig):
#     config_path = String.T(
#         default='',
#         help='Path to the synthetic database configuration file.')

#     outdir = String.T(
#         default='',
#         help='Directory where the output is stored in.')

#     sourcemode = StringChoice.T(
#         choices=['MT', 'RS'],
#         default='MT',
#         help='For which source type MT or RectangularSource the processing is done.')

#     comps = List.T(
#         String.T(),
#         default=['Z', 'E', 'N'],
#         optional=True,
#         help='List of components to calculate the ground motions values for; e.g. Z, N, E')

#     gmpes = List.T(
#         String.T(),
#         default=[],
#         optional=True,
#         help='List of GMPEs to use for the ground motion calculation. Needs the exact name as in OpenQuake.')

#     imts = List.T(
#         String.T(),
#         default=['pga', 'pgv', 'pgd'],
#         optional=True,
#         help='Intensity measures to calculate.')

#     freqs = List.T(
#         String.T(),
#         default=[],
#         optional=True,
#         help='Frequencies of spectral acceleration or Fourier spectrum to calculate.')

#     gf = String.T(
#         default='',
#         help='Path to the Green\'s function store.', optional=True)

#     rotd100 = Bool.T(
#         default=True,
#         help='Enables the option to calculate the vector-sum of both horizontal, aka the RotD100 (BooreRef).')

#     srccnt = Int.T(
#         default=100,
#         help='The number of created sources.')

#     mappoints = Int.T(
#         default=10,
#         help='SQRT of Number of points/locations to calculate imts for.')

#     mapextent = List.T(
#         Float.T(),
#         default=[1., 1.],
#         help='List of two numbers which define the map size in degree around the hypocenter.')

#     mapmode = StringChoice.T(
#         choices=['rectangular', 'circular', 'random'],
#         default='rectangular',
#         help='Mode of the map to be calculated for.')

#     mp = Int.T(
#         default=0,
#         help='Enables multiprocessing for the number of cores, if value > 1.')

#     append = Bool.T(
#         default=False,
#         help='Enables \'append\' mode. If database already exists, continue till srccnt.')

#     def get_config(self):
#         '''
#         Reads config from a YAML file.
#         '''
#         print(self.config_path)
#         config = load(filename=self.config_path)
#         self.config__ = config

#         return config


class GroundMotionData(ModuleConfig):
    config_path = String.T(
        default='',
        help='Path to the synthetic database configuration file.')

    outdir = String.T(
        default='',
        help='Directory where the output is stored in.')

    sourcemode = StringChoice.T(
        choices=['MT', 'RS', 'PDR'],
        default='MT',
        help='For which source type MT or RectangularSource the processing is done. (Pseudo-dynamic-rupture is currently not fully tested).')

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
        Float.T(),
        default=[],
        optional=True,
        help='Frequencies of spectral acceleration or Fourier spectrum to calculate.')

    gf = String.T(
        default='',
        help='Path to the Green\'s function store.', optional=True)

    rotd100 = Bool.T(
        default=True,
        help='Enables the option to calculate the vector-sum of both horizontal, aka the RotD100 (BooreRef).')

    mappoints = Int.T(
        default=10,
        help='SQRT of Number of points/locations to calculate imts for.')

    mapextent = List.T(
        Float.T(),
        default=[1., 1.],
        help='List of two numbers which define the map size in degree around the hypocenter.')

    mapmode = StringChoice.T(
        choices=['rectangular', 'circular', 'random', 'random_circular'],
        default='rectangular',
        help='Mode of the map to be calculated for.')

    def get_config(self):
        '''
        Reads config from a YAML file.
        '''
        print(self.config_path)
        config = load(filename=self.config_path)
        self.config__ = config

        return config


class SyntheticDatabase(GroundMotionData):
    
    srccnt = Int.T(
        default=100,
        help='The number of created sources.')

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


class ObservationalData(GroundMotionData):

    eqname = String.T(
        optional=True,
        default='Event',
        help='Name of the event.')

    filterfrequencies = List.T(
        optional=True,
        default=[None, None],
        help='Frequency for high and low-pass filtering of the waveform. e.g. [high-pass, low-pass]')

    waveformpath = String.T(
        default='',
        help='Path where the waveform is stored, either a directory or a single file.')

    eventfile = String.T(
        default='',
        help='Read-in of event information to create a source object.')

    faultfile = String.T(
        optional=True,
        default='',
        help='If sourcemode is RS or PDR; file with information of finite-fault. (Preferable .fsp) ')

    stationpath = String.T(
        default='',
        help='Station information (e.g. .xml) as file or directory.')

    stationstoremove = List.T(
        String.T(),
        default=[],
        optional=True,
        help='List of two numbers which define the map size in degree around the hypocenter.')

    ####

    datadir = String.T(
        default='',
        optional=True,
        help='Directory where all information, as wv, station etc. of the earthquake is stored.')

    synthetics = Bool.T(
        default=False,
        optional=True,
        help='Enables synthetic waveform processing for station coordinates of observation.')