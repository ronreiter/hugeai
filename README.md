# FLUX CLI Image Generation Tool

This repository contains a command-line interface (CLI) tool for generating images using the FLUX model, with optional super-resolution capabilities. The tool leverages the power of the FLUX model for image generation and RealESRGAN for super-resolution upscaling.

## Features

- Generate images using the FLUX model
- Customize image dimensions, guidance scale, and number of inference steps
- Generate multiple images with a single command
- Apply super-resolution upscaling (2x, 4x, or 8x) using RealESRGAN
- Reproducible results with seed control

## Requirements

- Python 3.7+
- PyTorch
- Hugging Face `diffusers` library
- RealESRGAN
- Other dependencies (see `requirements.txt`)

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/flux-cli-tool.git
   cd flux-cli-tool
   ```

2. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

3. Install RealESRGAN:
   ```
   pip install git+https://github.com/ai-forever/Real-ESRGAN.git
   ```

4. Set up your Hugging Face token:
   - Create a `.env` file in the project root
   - Add your Hugging Face token: `HF_TOKEN=your_token_here`

## Usage

Run the script with the following command:

```
python flux_cli_tool.py "your prompt here" [options]
```

### Options

- `--width`: Image width (default: 1536)
- `--height`: Image height (default: 1024)
- `--iterations`: Number of inference steps (default: 30)
- `--guidance`: Guidance scale (default: 2.5)
- `--seed`: Starting seed for image generation (default: 0)
- `--num`: Number of images to generate (default: 1)
- `--sr`: Super resolution scale (2, 4, or 8)

### Example

Generate 10 images with a specific prompt and default settings (1536x1024, 2.5 guidance scale, 30 iterations, 8x super-resolution, seeds 0-10):

```
poetry run python hugeai.py "futuristic city, midnight, huge stars and planets in sky, galaxy, lake, reflection, photo-realistic, vivid colorful lights" --num 10
```

## Output

Generated images are saved in the `outputs` directory.

## Notes

- The tool uses MPS (Metal Performance Shaders) for GPU acceleration on compatible Mac devices.
- Super-resolution is applied after the initial image generation, so the specified dimensions are for the pre-upscaled image.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]
