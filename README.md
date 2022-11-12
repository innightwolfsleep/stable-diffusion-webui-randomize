# Stable Diffusion Randomize extension

An extension for [AUTOMATIC1111/stable-diffusion-webui](https://github.com/AUTOMATIC1111/stable-diffusion-webui) that allows for random parameters during txt2img generation.

Syntax for randomization for parameters is `min, max, step`, with the exception of the randomize highres. param (a float value between `0` and `1`), and the seed and sampler list (comma separated list). Empty values are ignored.

If enabled, this script is processed for *all* generations, regardless of the script selected, meaning this script will function with others as well, such as [AUTOMATIC1111/stable-diffusion-webui-wildcards](https://github.com/AUTOMATIC1111/stable-diffusion-webui-wildcards).

Please note this is only for txt2img, I do not intend to add the same functionality for img2img.