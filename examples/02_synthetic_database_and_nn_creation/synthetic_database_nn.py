import os
import time

from gmacc import config
from gmacc.nnsynth import neural_networks as GMnn
from gmacc.nnsynth import preprocessing as GMpre


args = config.NeuralNetwork(config_path='synthetic_database_config_nn.yaml').get_config()
print(args)

xTrain, yTrain, xTest, yTest, xEval, yEval,\
		scalingDict, targets, inputcols = GMpre.read_subsets('%s/%s' % (args.indir, args.filecore), filetype='pkl')

absRefTime = time.time()
GMnn.nn_computation(args, xTrain, yTrain, xTest, yTest, xEval, yEval,
		scalingDict, targets, inputcols)
print('Script running time:', time.time() - absRefTime)


for ii, directory in enumerate(os.listdir(args.outdir)):
	modeldir = '%s/%s' % (args.outdir, directory)
	if not os.path.isdir(modeldir):
		continue
		
	modelfile = '%s/%s/MODEL.h5' % (args.outdir, directory)
	model = GMnn.load_model(modelfile)

	GMnn.evaluation_synthetic_database(model, xEval, yEval, scalingDict, targets, modeldir)
