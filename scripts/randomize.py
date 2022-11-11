import random

from modules import script_callbacks, scripts, shared
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingTxt2Img)
from modules.shared import opts
from scripts.xy_grid import build_samplers_dict


class RandomizeScript(scripts.Script):
	def title(self):
		return 'Randomize'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def process(self, p: StableDiffusionProcessing):
		if shared.opts.randomize_enabled and isinstance(p, StableDiffusionProcessingTxt2Img):
			all_opts = list(vars(shared.opts)['data'].keys())

			# Base params
			for param in self._list_params(all_opts):
				try:
					opt = self._opt(param, p)
					if opt is not None:
						setattr(p, param, opt)
				except TypeError:
					print(f'Failed to randomize param `{param}` -- incorrect value?')

			# Other params
			for param in self._list_params(all_opts, prefix='randomize_other_'):
				if param == 'CLIP_stop_at_last_layers':
					opts.data[param] = int(self._opt(param, p, prefix='randomize_other_')) # type: ignore

			# Highres. fix params
			if random.random() < float(shared.opts.randomize_hires or 0): # type: ignore
				try:
					setattr(p, 'width', self._opt('width', p, 'randomize_hires_'))
					setattr(p, 'height', self._opt('height', p, 'randomize_hires_'))

					setattr(p, 'enable_hr', True)
					setattr(p, 'firstphase_width', 0)
					setattr(p, 'firstphase_height', 0)
					setattr(p, 'truncate_x', 0)
					setattr(p, 'truncate_y', 0)

					setattr(p, 'denoising_strength', self._opt('denoising_strength', p, prefix='randomize_hires_'))
				except TypeError:
					print(f'Failed to utilize highres. fix -- incorrect value?')
		else:
			return

	def _list_params(self, opts, prefix='randomize_param_'):
		for param in [o for o in filter(lambda x: x.startswith(prefix), opts)]:
				if len(getattr(shared.opts, param).strip()) > 0:
					yield param.split(prefix)[1]

	def _opt(self, opt, p, prefix='randomize_param_'):
		opt_name = f'{prefix}{opt}'
		opt_val: str = getattr(shared.opts, opt_name)
		opt_arr: list[str] = [x.strip() for x in opt_val.split(',')]
		if self._is_num(opt_arr[0]) and len(opt_arr) == 3:
			vals = [float(v) for v in opt_arr]
			rand = self._rand(vals[0], vals[1], vals[2])
			if rand.is_integer():
				return int(rand)
			else:
				return round(float(rand), max(0, int(opt_arr[2][::-1].find('.'))))
		else:
			if opt == 'sampler_index':
				return build_samplers_dict(p).get(random.choice(opt_arr).lower(), None)
			else:
				return random.choice(opt_arr)

	def _rand(self, start: float, stop: float, step: float) -> float:
		return random.randint(0, int((stop - start) / step)) * step + start
	
	def _is_num(self, val: str):
		if val.isdigit():
			return True
		else:
			try:
				float(val)
				return True
			except ValueError:
				return False

def on_ui_settings():
	shared.opts.add_option('randomize_enabled', shared.OptionInfo(False, 'Enable Randomize extension', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_param_sampler_index', shared.OptionInfo('euler a,euler', 'Randomize Sampler', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_param_cfg_scale', shared.OptionInfo('5,15,0.5', 'Randomize CFG Scale', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_param_steps', shared.OptionInfo('10,50,2', 'Randomize Steps', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_param_width', shared.OptionInfo('256,768,64', 'Randomize Width', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_param_height', shared.OptionInfo('256,768,64', 'Randomize Height', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_hires', shared.OptionInfo('0.25', 'Randomize Highres. percentage', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_hires_denoising_strength', shared.OptionInfo('0.5,0.8,0.05', 'Randomize Highres. Denoising Strength', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_hires_width', shared.OptionInfo('768,1920,64', 'Randomize Highres. Width', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_hires_height', shared.OptionInfo('768,1920,64', 'Randomize Highres. Height', section=('randomize', 'Randomize')))
	shared.opts.add_option('randomize_other_CLIP_stop_at_last_layers', shared.OptionInfo('1,2,1', 'Randomize CLIP skip', section=('randomize', 'Randomize')))

script_callbacks.on_ui_settings(on_ui_settings)