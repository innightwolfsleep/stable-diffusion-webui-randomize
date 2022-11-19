import random
import gradio as gr

from modules import scripts, sd_models, shared
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingTxt2Img)
from modules.shared import opts
from modules.sd_samplers import all_samplers_map

class RandomizeScript(scripts.Script):
	def __init__(self) -> None:
		super().__init__()

		self.randomize_prompt_word = ''

	def title(self):
		return 'Randomize'

	def show(self, is_img2img):
		if not is_img2img:
			return scripts.AlwaysVisible

	def ui(self, is_img2img):
		randomize_enabled, randomize_param_sampler_name, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers, randomize_other_sd_model_checkpoint = self._create_ui()

		return [randomize_enabled, randomize_param_sampler_name, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers, randomize_other_sd_model_checkpoint]

	def process(
		self,
		p: StableDiffusionProcessing,
		randomize_enabled: bool,
		# randomize_param_seed: str,
		randomize_param_sampler_name: str,
		randomize_param_cfg_scale: str,
		randomize_param_steps: str,
		randomize_param_width: str,
		randomize_param_height: str,
		randomize_hires: str,
		randomize_hires_denoising_strength: str,
		randomize_hires_width: str,
		randomize_hires_height: str,
		randomize_other_CLIP_stop_at_last_layers: str,
		randomize_other_sd_model_checkpoint: str,
		**kwargs
	):
		if randomize_enabled and isinstance(p, StableDiffusionProcessingTxt2Img):
			all_opts = {k: v for k, v in locals().items() if k not in ['self', 'p', 'randomize_enabled', 'batch_number', 'prompts', 'seeds', 'subseeds']}

			for param, val in self._list_params(all_opts, prefix='randomize_other_'):
				if param == 'sd_model_checkpoint':
					# TODO (mmaker): Trigger changing the ckpt dropdown value in the UI
					sd_model_checkpoint = self._opt({param: val}, p)
					if sd_model_checkpoint:
						sd_models.reload_model_weights(shared.sd_model, sd_model_checkpoint)
						p.sd_model = shared.sd_model

			# Checkpoint specific
			# TODO (mmaker): Allow for some way to format how this is inserted into the prompt
			if len(self.randomize_prompt_word) > 0:
				p.prompt = self.randomize_prompt_word + ', ' + p.prompt
				if p.all_prompts and len(p.all_prompts) > 0:
					p.all_prompts = [self.randomize_prompt_word + ', ' + prompt for prompt in p.all_prompts] # type: ignore


	def process_batch(
		self,
		p: StableDiffusionProcessing,
		randomize_enabled: bool,
		# randomize_param_seed: str,
		randomize_param_sampler_name: str,
		randomize_param_cfg_scale: str,
		randomize_param_steps: str,
		randomize_param_width: str,
		randomize_param_height: str,
		randomize_hires: str,
		randomize_hires_denoising_strength: str,
		randomize_hires_width: str,
		randomize_hires_height: str,
		randomize_other_CLIP_stop_at_last_layers: str,
		randomize_other_sd_model_checkpoint: str,
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
				except (TypeError, IndexError) as exception:
					print(f'Failed to randomize param `{param}` -- incorrect value?', exception)

			# Other params
			for param, val in self._list_params(all_opts, prefix='randomize_other_'):
				if param == 'CLIP_stop_at_last_layers':
					opts.data[param] = int(self._opt({param: val}, p)) # type: ignore

			# Highres. fix params
			if random.random() < float(randomize_hires or 0):
				try:
					setattr(p, 'enable_hr', True)
					setattr(p, 'firstphase_width', 0)
					setattr(p, 'firstphase_height', 0)
					setattr(p, 'denoising_strength', self._opt({'denoising_strength': randomize_hires_denoising_strength}, p))
					setattr(p, 'width', self._opt({'width': randomize_hires_width}, p))
					setattr(p, 'height', self._opt({'height': randomize_hires_height}, p))

					# Set up highres. fix related stuff by re-running init function
					p.init(p.all_prompts, p.all_seeds, p.all_subseeds)
				except (TypeError, IndexError) as exception:
					print(f'Failed to utilize highres. fix -- incorrect value?', exception)
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
			if opt_name == 'sampler_name':
				random_sampler = random.choice(opt_arr)
				if random_sampler in all_samplers_map:
					return random_sampler
			elif opt_name == 'seed':
				return int(random.choice(opt_arr))
			elif opt_name == 'sd_model_checkpoint':
				choice = random.choice(opt_arr)
				if ':' in choice:
					ckpt_name = choice.split(':')[0].strip()
					self.randomize_prompt_word = choice.split(':')[1].strip()
				else:
					ckpt_name = choice
					self.randomize_prompt_word = ''
				return sd_models.get_closet_checkpoint_match(ckpt_name)
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
				randomize_param_sampler_name = gr.Textbox(label='Sampler', value='', placeholder=hint_list)
				randomize_param_cfg_scale = gr.Textbox(label='CFG Scale', value='', placeholder=hint_minmax)
				randomize_param_steps = gr.Textbox(label='Steps', value='', placeholder=hint_minmax)
				randomize_param_width = gr.Textbox(label='Width', value='', placeholder=hint_minmax)
				randomize_param_height = gr.Textbox(label='Height', value='', placeholder=hint_minmax)
				randomize_hires = gr.Textbox(label='Highres. percentage chance', value='0', placeholder='Float value from 0 to 1')
				randomize_hires_denoising_strength = gr.Textbox(label='Highres. Denoising Strength', value='', placeholder=hint_minmax)
				randomize_hires_width = gr.Textbox(label='Highres. Width', value='', placeholder=hint_minmax)
				randomize_hires_height = gr.Textbox(label='Highres. Height', value='', placeholder=hint_minmax)
				randomize_other_CLIP_stop_at_last_layers = gr.Textbox(label='Stop at CLIP layers', value='', placeholder=hint_minmax)
				randomize_other_sd_model_checkpoint = gr.Textbox(label='Checkpoint name', value='', placeholder='Comma separated list. Specify ckpt OR ckpt:word')
		
		return randomize_enabled, randomize_param_sampler_name, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers, randomize_other_sd_model_checkpoint
