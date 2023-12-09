import argparse

from model import Transformer

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

	model = Transformer(mode, 'trained_model', 'config.yml')

	match mode:
		case 'train':
			model.run_train('trained_model')
		case 'infer':
			model.run_infer(args.features_file, args.predictions_file)
