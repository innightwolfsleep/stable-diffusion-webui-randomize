import random
import gradio as gr

from modules import scripts, sd_models, shared
from modules.processing import (StableDiffusionProcessing,
                                StableDiffusionProcessingTxt2Img)
from modules.shared import opts, cmd_opts
from modules.sd_models import checkpoints_list
from modules.sd_samplers import all_samplers_map
from modules.hypernetworks import hypernetwork

try:
	from scripts.xy_grid import build_samplers_dict # type: ignore
except ImportError:
	def build_samplers_dict():
		return {}


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
		randomize_enabled, randomize_param_sampler_name, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers, randomize_other_sd_model_checkpoint, randomize_other_sd_hypernetwork, randomize_other_sd_hypernetwork_strength, randomize_other_eta_noise_seed_delta = self._create_ui()

		return [randomize_enabled, randomize_param_sampler_name, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers, randomize_other_sd_model_checkpoint, randomize_other_sd_hypernetwork, randomize_other_sd_hypernetwork_strength, randomize_other_eta_noise_seed_delta]

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
		randomize_other_sd_hypernetwork: str,
		randomize_other_sd_hypernetwork_strength: str,
		randomize_other_eta_noise_seed_delta: str,
		**kwargs
	):
		if randomize_enabled and isinstance(p, StableDiffusionProcessingTxt2Img):
			all_opts = {k: v for k, v in locals().items() if k not in ['self', 'p', 'randomize_enabled', 'batch_number', 'prompts', 'seeds', 'subseeds']}

			# NOTE (mmaker): Can we update these in the UI?
			for param, val in self._list_params(all_opts, prefix='randomize_other_'):
				if param == 'sd_model_checkpoint':
					sd_model_checkpoint = self._opt({param: val}, p)
					if sd_model_checkpoint:
						sd_models.reload_model_weights(shared.sd_model, sd_model_checkpoint)
						p.sd_model = shared.sd_model
				if param == 'sd_hypernetwork':
					hypernetwork.load_hypernetwork(self._opt({param: val}, p))
				if param == 'sd_hypernetwork_strength':
					hypernetwork.apply_strength(self._opt({param: val}, p))

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
		randomize_other_sd_hypernetwork: str,
		randomize_other_sd_hypernetwork_strength: str,
		randomize_other_eta_noise_seed_delta: str,
		**kwargs
	):
		if randomize_enabled and isinstance(p, StableDiffusionProcessingTxt2Img):
			# TODO (mmaker): Fix this jank. Don't do this.
			all_opts = {k: v for k, v in locals().items() if k not in ['self', 'p', 'randomize_enabled', 'batch_number', 'prompts', 'seeds', 'subseeds']}

			# Base params
			for param, val in self._list_params(all_opts):
				try:
					# Backwards compat
					if param == 'sampler_name' and hasattr(p, 'sampler_index') and not hasattr(p, 'sampler_name'):
						param = 'sampler_index'

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
				if param in ['CLIP_stop_at_last_layers', 'eta_noise_seed_delta']:
					opts.data[param] = self._opt({param: val}, p) # type: ignore

			# Highres. fix params
			if random.random() < float(randomize_hires or 0):
				try:
					setattr(p, 'enable_hr', True)
					setattr(p, 'firstphase_width', 0)
					setattr(p, 'firstphase_height', 0)

					denoising_strength = self._opt({'denoising_strength': randomize_hires_denoising_strength}, p)
					if denoising_strength:
						setattr(p, 'denoising_strength', denoising_strength)
					else:
						# Default value used by WebUI
						setattr(p, 'denoising_strength', 0.7)

					hires_width = self._opt({'width': randomize_hires_width}, p)
					hires_height = self._opt({'height': randomize_hires_height}, p)
					if hires_width:
						setattr(p, 'width', hires_width)
					if hires_height:
						setattr(p, 'height', hires_height)

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
		opt_val = list(opt.values())[0].strip()

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
				if opt_val == '*':
					return random.choice(list(all_samplers_map.keys()))
				random_sampler = random.choice(opt_arr)
				if random_sampler in all_samplers_map:
					return random_sampler
			if opt_name == 'sampler_index':
				samplers_dict = build_samplers_dict()
				if len(samplers_dict) > 0:
					if opt_val == '*':
						return random.choice(list(samplers_dict.values()))
					return samplers_dict.get(random.choice(opt_arr).lower(), None)
				else:
					return None
			if opt_name == 'seed':
				return int(random.choice(opt_arr))
			if opt_name == 'sd_model_checkpoint':
				if opt_val == '*':
					return random.choice(list(checkpoints_list.values()))
				choice = random.choice(opt_arr)
				if ':' in choice:
					ckpt_name = choice.split(':')[0].strip()
					self.randomize_prompt_word = choice.split(':')[1].strip()
				else:
					ckpt_name = choice
					self.randomize_prompt_word = ''
				return sd_models.get_closet_checkpoint_match(ckpt_name)
			if opt_name == 'sd_hypernetwork':
				if opt_val.lower() == 'none':
					return None
				if opt_val == '*':
					return random.choice(list(hypernetwork.list_hypernetworks(cmd_opts.hypernetwork_dir).keys()))
				return hypernetwork.find_closest_hypernetwork_name(random.choice(opt_arr))
			
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
		hint_list = 'Comma separated list OR * for all'
		hint_float = 'Float value from 0 to 1'

		with gr.Group():
			with gr.Accordion('Randomize', open=False):
				randomize_enabled = gr.Checkbox(label='Enable', value=False)
				# randomize_param_seed = gr.Textbox(label='Seed', value='', placeholder=hint_list)
				randomize_param_sampler_name = gr.Textbox(label='Sampler', value='', placeholder=hint_list)
				randomize_param_cfg_scale = gr.Textbox(label='CFG Scale', value='', placeholder=hint_minmax)
				randomize_param_steps = gr.Textbox(label='Steps', value='', placeholder=hint_minmax)
				randomize_param_width = gr.Textbox(label='Width', value='', placeholder=hint_minmax)
				randomize_param_height = gr.Textbox(label='Height', value='', placeholder=hint_minmax)
				randomize_hires = gr.Textbox(label='Highres. percentage chance', value='', placeholder=hint_float)
				randomize_hires_denoising_strength = gr.Textbox(label='Highres. Denoising Strength', value='', placeholder=hint_minmax)
				randomize_hires_width = gr.Textbox(label='Highres. Width', value='', placeholder=hint_minmax)
				randomize_hires_height = gr.Textbox(label='Highres. Height', value='', placeholder=hint_minmax)
				randomize_other_CLIP_stop_at_last_layers = gr.Textbox(label='Stop at CLIP layers', value='', placeholder=hint_minmax)
				randomize_other_sd_model_checkpoint = gr.Textbox(label='Checkpoint name', value='', placeholder='Comma separated list. Specify ckpt OR ckpt:word OR *')
				randomize_other_sd_hypernetwork = gr.Textbox(label='Hypernetwork', value='', placeholder=hint_list)
				randomize_other_sd_hypernetwork_strength = gr.Textbox(label='Hypernetwork strength', value='', placeholder=hint_minmax)
				randomize_other_eta_noise_seed_delta = gr.Textbox(label='Eta noise seed delta', value='', placeholder=hint_minmax)
		
		return randomize_enabled, randomize_param_sampler_name, randomize_param_cfg_scale, randomize_param_steps, randomize_param_width, randomize_param_height, randomize_hires, randomize_hires_denoising_strength, randomize_hires_width, randomize_hires_height, randomize_other_CLIP_stop_at_last_layers, randomize_other_sd_model_checkpoint, randomize_other_sd_hypernetwork, randomize_other_sd_hypernetwork_strength, randomize_other_eta_noise_seed_delta
