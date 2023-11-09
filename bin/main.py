import argparse
import os
from shutil import copy2 as copy

import model
from config.config import get_configs

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main argument parser')
	parser.add_argument('run_mode',
	                    choices=('train', 'infer'),
	                    help='Main running mode of the program')
	parser.add_argument('--model',
	                    type=str,
	                    choices=model.AvailableModels.keys(),
	                    help='The type of model to be ran')
	parser.add_argument('--model_dir',
	                    type=str,
	                    required=True,
	                    help='Location of model')
	parser.add_argument('--config',
	                    type=str,
	                    nargs='+',
	                    default=None,
	                    help='Location of the config file')
	parser.add_argument('--no_keeping_config',
	                    action='store_false',
	                    help='If set, do not copy the config file to the model directory')

	# arguments for inference
	parser.add_argument('--features_file',
	                    type=str,
	                    help='Inference mode: Provide the location of features file')
	parser.add_argument('--predictions_file',
	                    type=str,
	                    help='Inference mode: Provide Location of output file which is predicted from features file')
	parser.add_argument('--infer_batch_size',
	                    type=int,
	                    default=None,
	                    help='Specify the batch_size to run the model with. '
	                         'Default use the config value.')
	parser.add_argument('--checkpoint',
	                    type=str,
	                    default=None,
	                    help='All mode: specify to load the checkpoint into model.')
	parser.add_argument('--checkpoint_idx',
	                    type=int,
	                    default=0,
	                    help='All mode: specify the epoch of the checkpoint loaded. '
	                         'Only useful for training.')

	args = parser.parse_args()

	# create directory if not exist
	os.makedirs(args.model_dir, exist_ok=True)
	config_path = args.config
	if config_path is None:
		config_path = get_configs(args.model_dir)
		print(f'Config path not specified, load the configs in model directory which is {config_path}')
	elif args.no_keeping_config:
		# store false variable, mean true is default
		print('Config specified, copying all to model dir')
		for path in config_path:
			copy(path, args.model_dir)

	# load model. Specific run mode required converting
	mode = args.run_mode
	assert mode in ['train', 'infer'], f'Unknown mode: {mode}'

	model = model.AvailableModels[args.model](mode, args.model_dir, config_path)
	model.load_checkpoint(args.model_dir, args.checkpoint, args.checkpoint_idx)

	# run model
	match mode:
		case 'train':
			model.run_train(model_dir=args.model_dir)
		case 'infer':
			model.run_infer(args.features_file,
			                args.predictions_file,
			                src_lang=args.src_lang,
			                trg_lang=args.trg_lang,
			                batch_size=args.infer_batch_size)
