import argparse
import os
from shutil import copy2 as copy

from config.config import get_configs
from model import Transformer

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main argument parser')
	parser.add_argument('run_mode',
	                    choices=('train', 'infer'),
	                    help='Main running mode of the program')
	parser.add_argument('--model_dir',
	                    type=str,
	                    required=True,
	                    help='Location of model')
	parser.add_argument('--config',
	                    type=str,
	                    nargs='+',
	                    default=None,
	                    help='Location of the config file')
	parser.add_argument('--features_file',
	                    type=str,
	                    help='Inference mode: Provide the location of features file')
	parser.add_argument('--predictions_file',
	                    type=str,
	                    help='Inference mode: Provide Location of output file which is predicted from features file')

	args = parser.parse_args()

	# create directory if not exist
	os.makedirs(args.model_dir, exist_ok=True)
	config_path = args.config
	if config_path is None:
		config_path = get_configs(args.model_dir)
		print(f'Config path not specified, load the configs in model directory which is {config_path}')
	else:
		# store false variable, mean true is default
		print('Config specified, copying all to model dir')
		for path in config_path:
			copy(path, args.model_dir)

	# load model. Specific run mode required converting
	mode = args.run_mode
	assert mode in ['train', 'infer'], f'Unknown mode: {mode}'

	model = Transformer(mode, args.model_dir, config_path)

	# run model
	match mode:
		case 'train':
			model.run_train(args.model_dir)
		case 'infer':
			model.run_infer(args.features_file, args.predictions_file)
