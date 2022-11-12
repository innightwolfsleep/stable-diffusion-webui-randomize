import random
import gradio as gr

from modules import scripts
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingTxt2Img)
from modules.shared import opts
from scripts.xy_grid import build_samplers_dict

class RandomizeScript(scripts.Script):
	def title(self):
		return 'Randomize'

	def show(self, is_img2img):
		if not is_img2img:
			return scripts.AlwaysVisible

	def ui(self, is_img2img):
		randomize_enabled, randomize_param_sampler_index, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers = self._create_ui()

		return [randomize_enabled, randomize_param_sampler_index, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers]

	def process_batch(
		self,
		p: StableDiffusionProcessing,
		randomize_enabled: bool,
		randomize_param_sampler_index: str,
		randomize_param_cfg_scale: str,
		randomize_param_steps: str,
		randomize_param_width: str,
		randomize_param_height: str,
		randomize_hires: str,
		randomize_hires_denoising_strength: str,
		randomize_hires_width: str,
		randomize_hires_height: str,
		randomize_other_CLIP_stop_at_last_layers: str,
		**kwargs
	):
		if randomize_enabled and isinstance(p, StableDiffusionProcessingTxt2Img):
			# TODO (mmaker): Fix this jank. Don't do this.
			all_opts = {k: v for k, v in locals().items() if k not in ['self', 'p', 'randomize_enabled', 'batch_number', 'prompts', 'seeds', 'subseeds']}

			# Base params
			for param, val in self._list_params(all_opts):
				try:
					opt = self._opt({param: val}, p)
					if opt is not None:
						if param in ['seed']:
							setattr(p, 'all_seeds', [self._opt({param: val}, p) for _ in range(0, len(getattr(p, 'all_seeds')))]) # NOTE (mmaker): Is this correct?
						else:
							setattr(p, param, opt)
					else:
						print(f'Skipping randomizing param `{param}` -- incorrect value')
				except (TypeError, IndexError):
					print(f'Failed to randomize param `{param}` -- incorrect value?')

			# Other params
			for param, val in self._list_params(all_opts, prefix='randomize_other_'):
				if param == 'CLIP_stop_at_last_layers':
					opts.data[param] = int(self._opt({param: val}, p)) # type: ignore

			# Highres. fix params
			if random.random() < float(randomize_hires or 0):
				try:
					setattr(p, 'width', self._opt({'width': randomize_hires_width}, p))
					setattr(p, 'height', self._opt({'height': randomize_hires_height}, p))

					setattr(p, 'enable_hr', True)
					setattr(p, 'firstphase_width', 0)
					setattr(p, 'firstphase_height', 0)
					setattr(p, 'truncate_x', 0)
					setattr(p, 'truncate_y', 0)

					setattr(p, 'denoising_strength', self._opt({'denoising_strength': randomize_hires_denoising_strength}, p))
				except (TypeError, IndexError):
					print(f'Failed to utilize highres. fix -- incorrect value?')
		else:
			return

	def _list_params(self, opts, prefix='randomize_param_'):
		for k, v in opts.items():
			if k.startswith(prefix) and v is not None and len(v) > 0:
				yield k.replace(prefix,''), v

	def _opt(self, opt, p):
		opt_name = list(opt.keys())[0]
		opt_val = list(opt.values())[0]

		opt_arr: list[str] = [x.strip() for x in opt_val.split(',')]

		if self._is_num(opt_arr[0]) and len(opt_arr) == 3 and opt_name not in ['seed']:
			vals = [float(v) for v in opt_arr]
			rand = self._rand(vals[0], vals[1], vals[2])
			if rand.is_integer():
				return int(rand)
			else:
				return round(float(rand), max(0, int(opt_arr[2][::-1].find('.'))))
		else:
			if opt_name == 'sampler_index':
				return build_samplers_dict(p).get(random.choice(opt_arr).lower(), None)
			elif opt_name == 'seed':
				return int(random.choice(opt_arr))
			else:
				return None

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

	def _create_ui(self):
		hint_minmax = 'Range of stepped values (min, max, step)'
		hint_list = 'Comma separated list'

		with gr.Group():
			with gr.Accordion('Randomize', open=False):
				randomize_enabled = gr.Checkbox(label='Enable', value=False)
				# randomize_param_seed = gr.Textbox(label='Seed', value='', placeholder=hint_list)
				randomize_param_sampler_index = gr.Textbox(label='Sampler', value='', placeholder=hint_list)
				randomize_param_cfg_scale = gr.Textbox(label='CFG Scale', value='', placeholder=hint_minmax)
				randomize_param_steps = gr.Textbox(label='Steps', value='', placeholder=hint_minmax)
				randomize_param_width = gr.Textbox(label='Width', value='', placeholder=hint_minmax)
				randomize_param_height = gr.Textbox(label='Height', value='', placeholder=hint_minmax)
				randomize_hires = gr.Textbox(label='Highres. percentage chance', value='0', placeholder='Float value from 0 to 1')
				randomize_hires_denoising_strength = gr.Textbox(label='Highres. Denoising Strength', value='', placeholder=hint_minmax)
				randomize_hires_width = gr.Textbox(label='Highres. Width', value='', placeholder=hint_minmax)
				randomize_hires_height = gr.Textbox(label='Highres. Height', value='', placeholder=hint_minmax)
				randomize_other_CLIP_stop_at_last_layers = gr.Textbox(label='Stop at CLIP layers', value='', placeholder=hint_minmax)
		
		return randomize_enabled, randomize_param_sampler_index, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers
