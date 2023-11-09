import io
import os

import yaml


def is_supported(path):
	return os.path.splitext(path)[-1] in ['.yaml', '.yml']


def get_configs(path):
	return [os.path.join(path, f) for f in os.listdir(path) if is_supported(f)]


class MonoConfig(dict):
	def __init__(self, path, **elements):
		"""Initiate a config object, where specified elements override the default config loaded"""
		super(MonoConfig, self).__init__(self._load(path))
		self.update(**elements)

	def _load(self, path):
		assert isinstance(path, str), (
			f'Basic Config class can only support a single file path (str), '
			f'but {path:s} is ({type(path):s})'
		)
		assert os.path.isfile(path), f'Config file {path:s} does not exist'
		assert is_supported(path), f'Unsupported extension from file {path:s}'

		with io.open(path, 'r', encoding='utf-8') as stream:
			return yaml.full_load(stream)


class MultiConfig(MonoConfig):
	def _load(self, paths):
		"""Update to support multiple paths."""
		super_cls = super(MultiConfig, self)

		if isinstance(paths, list):
			print('Loaded path is a list of locations.')

			result = {}
			[self._update(result, super_cls._load(path)) for path in paths]
			return result
		else:
			return super_cls._load(paths)

	def _update(self, orig, new):
		"""Instead of overriding dicts, merge them recursively."""
		for k, v in new.items():
			if k in orig and isinstance(orig[k], dict):
				assert isinstance(v, dict), f'Mismatching config with key {k}: {orig[k]} - {v}'

				self._update(orig[k], v)
			else:
				orig[k] = v
