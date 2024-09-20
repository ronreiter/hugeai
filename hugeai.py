import os
import argparse
from slugify import slugify

import torch
import diffusers
from diffusers import FluxPipeline
from huggingface_hub import login
import dotenv
import RealESRGAN

GENERATION_MODEL = "black-forest-labs/FLUX.1-dev"
# GENERATION_MODEL = "black-forest-labs/FLUX.1-dev"


UPSCALE_MODEL = "FacehugmanIII/4x_foolhardy_Remacri/4x_foolhardy_Remacri.pth"
# UPSCALE_MODEL = "weights/RealESRGAN_x8.pth"

torch.set_default_device("mps")

dotenv.load_dotenv()

login(token=os.environ.get("HF_TOKEN"))

_flux_rope = diffusers.models.transformers.transformer_flux.rope

def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
    assert dim % 2 == 0, "The dimension must be even."
    if pos.device.type == "mps":
        return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
    else:
        return _flux_rope(pos, dim, theta)


diffusers.models.transformers.transformer_flux.rope = new_flux_rope


def apply_super_resolution(image, scale):
    model = RealESRGAN.RealESRGAN(torch.device("mps"), scale=scale)
    model.load_weights(UPSCALE_MODEL, download=True)
    #model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    return model.predict(image)


def generate_images(prompt, width, height, guidance, iterations, seed, num_images, sr_scale):
    # Load the pipeline
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", revision='refs/pr/1', torch_dtype=torch.bfloat16).to("mps")
    pipe = FluxPipeline.from_pretrained(GENERATION_MODEL, torch_dtype=torch.bfloat16).to("mps")

    # Create output directory if it doesn't exist
    os.makedirs("outputs", exist_ok=True)

    # Generate slug from prompt
    slug = slugify(prompt)

    for i in range(num_images):
        # Generate image
        out = pipe(
            prompt=prompt,
            guidance_scale=guidance,
            width=width,
            height=height,
            num_inference_steps=iterations,
            max_sequence_length=512,
            generator=torch.Generator("mps").manual_seed(seed+i)
        ).images[0]

        out.save("temp.png")
        # Apply super resolution if requested
        if sr_scale > 1:
            print(f"Upscaling {sr_scale}x...")
            out = apply_super_resolution(out, sr_scale)

        # Save image
        filename = f"outputs/{slug}_it_{iterations}_{'sr_' + str(sr_scale) if sr_scale else 'original'}_num_{i+1}.png"
        out.save(filename)
        print(f"Saved image: {filename}")


def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX model with optional super resolution")
    parser.add_argument("prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--width", type=int, default=1536, help="Image width (default: 1536)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--iterations", type=int, default=30, help="Iterations (default: 25)")
    parser.add_argument("--guidance", type=float, default=2.5, help="Guidance (default: 2.5)")
    parser.add_argument("--seed", type=int, default=0, help="Seed start (default: 0)")
    parser.add_argument("--num", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--sr", type=int, choices=[1, 2, 4, 8], default=8, help="Super resolution scale (1, 2, 4, or 8) (default: 8)")

    args = parser.parse_args()

    generate_images(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        guidance=args.guidance,
        iterations=args.iterations,
        seed=args.seed,
        num_images=args.num,
        sr_scale=args.sr
    )


if __name__ == "__main__":
    main()
