import random

from modules import script_callbacks, scripts, shared
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingTxt2Img)
from scripts.xy_grid import build_samplers_dict


class RandomizeScript(scripts.Script):
	def title(self):
		return 'Randomize'

	def show(self, is_img2img):
		return scripts.AlwaysVisible

	def process(self, p: StableDiffusionProcessing):
		if shared.opts.randomize_enabled and isinstance(p, StableDiffusionProcessingTxt2Img):
			all_opts = list(vars(shared.opts)['data'].keys())
			for param in [o for o in filter(lambda x: x.startswith('randomize_param_'), all_opts)]:
				if len(getattr(shared.opts, param).strip()) > 0:
					param_name = param.split('randomize_param_')[1]
					try:
						opt = self._opt(param_name, p)
						if opt is not None:
							setattr(p, param_name, opt)
					except TypeError:
						print(f'Failed to randomize param `{param_name}` -- incorrect value?')
			if random.random() < float(shared.opts.randomize_hires or None): # type: ignore
				try:
					setattr(p, 'width', self._opt('width', p, 'randomize_hires_'))
					setattr(p, 'height', self._opt('height', p, 'randomize_hires_'))

					setattr(p, 'enable_hr', True)
					setattr(p, 'firstphase_width', 0)
					setattr(p, 'firstphase_height', 0)
					setattr(p, 'truncate_x', 0)
					setattr(p, 'truncate_y', 0)

					setattr(p, 'denoising_strength', float(self._opt('denoising_strength', p, 'randomize_hires_'))) # type: ignore
				except TypeError:
					print(f'Failed to utilize highres. fix -- incorrect value?')
		else:
			return

	def _opt(self, opt, p, prefix='randomize_param_'):
		opt_name = f'{prefix}{opt}'
		opt_val: str = getattr(shared.opts, opt_name)
		opt_arr: list[str] = opt_val.split(',')
		if opt_arr[0].isdigit():
			vals = [float(v) for v in opt_arr]
			rand = self._rand(vals[0], vals[1], vals[2])
			if rand.is_integer():
				return int(rand)
			else:
				return float(rand)
		else:
			if opt == 'sampler_index':
				return build_samplers_dict(p).get(random.choice(opt_arr).lower(), None)
			else:
				return random.choice(opt_arr)

	def _rand(self, start: float, stop: float, step: float) -> float:
		return random.randint(0, int((stop - start) / step)) * step + start

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

script_callbacks.on_ui_settings(on_ui_settings)