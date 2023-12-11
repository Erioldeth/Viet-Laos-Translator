import argparse

import torch

from model import Transformer
from model.save import load_model

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Main argument parser')
	parser.add_argument('run_mode',
	                    choices=('train', 'infer'),
	                    help='Main running mode of the program')
	parser.add_argument('--features_file',
	                    type=str,
	                    help='Inference mode: Provide the location of features file')
	parser.add_argument('--predictions_file',
	                    type=str,
	                    help='Inference mode: Provide Location of output file which is predicted from features file')

	args = parser.parse_args()
	mode = args.run_mode
	assert mode in ['train', 'infer'], f'Unknown mode: {mode}'

	# device = torch.device('cpu')
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = Transformer(mode, 'trained_model', 'config.yml', device).to(device)

	match mode:
		case 'train':
			model.run_train('trained_model')
		case 'infer':
			load_model(model, 'trained_model')
			model.run_infer(args.features_file, args.predictions_file)
