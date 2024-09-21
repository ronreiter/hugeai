import os
import argparse
import random

from PIL.Image import Image
from PIL.PngImagePlugin import PngInfo

from aura_sr import AuraSR

import torch
import diffusers
from diffusers import FluxPipeline
from huggingface_hub import login
import dotenv
import RealESRGAN
import hashlib


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
    model.load_weights("FacehugmanIII/4x_foolhardy_Remacri/4x_foolhardy_Remacri.pth", download=True)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    return model.predict(image)


# https://huggingface.co/fal/AuraSR
def apply_super_resolution_aura(image):
    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2", device="mps")
    upscaled_image = aura_sr.upscale_4x_overlapped(image)
    return upscaled_image


def generate_images(prompt, width, height, guidance, iterations, seed, num_images, sr_scale, sr_aura):
    # Load the pipeline
    # pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", revision='refs/pr/1', torch_dtype=torch.bfloat16).to("mps")
    pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16).to("mps")

    dir_name = hashlib.md5(prompt.encode()).hexdigest()
    os.makedirs(f"outputs/{dir_name}", exist_ok=True)

    for i in range(num_images):
        image_num = seed+i+1
        base_filename = f"outputs/{dir_name}/num_{image_num}_it_{iterations}_guidance_{guidance}"
        filename = f"{base_filename}_sr_1.png"

        metadata = PngInfo()
        metadata.add_text("Prompt", prompt)
        metadata.add_text("Guidance", str(guidance))
        metadata.add_text("Iterations", str(iterations))
        metadata.add_text("Seed", str(seed))
        metadata.add_text("Created With", "Huge AI")

        if os.path.exists(filename):
            out = Image.open(filename)
        else:
            # Generate image
            out = pipe(
                prompt=prompt,
                guidance_scale=guidance,
                width=width,
                height=height,
                num_inference_steps=iterations,
                max_sequence_length=512,
                generator=torch.Generator("mps").manual_seed(image_num)
            ).images[0]

            out.save(filename, pnginfo=metadata)
            print(f"Saved image: {filename}")
            out.show()

        # Apply super resolution if requested
        if sr_scale > 1:
            print(f"Upscaling {sr_scale}x...")
            out = apply_super_resolution(out, sr_scale)

            # Save upscaled image
            filename = f"{base_filename}_{'sr_' + str(sr_scale) if sr_scale else 'original'}.png"
            out.save(filename, pnginfo=metadata)
            print(f"Saved image: {filename}")
        elif sr_aura:
            print(f"Upscaling with aura...")
            out = apply_super_resolution_aura(out)

            # Save upscaled image
            filename = f"{base_filename}_sr_aura.png"
            out.save(filename, pnginfo=metadata)


def main():
    parser = argparse.ArgumentParser(description="Generate images using FLUX model with optional super resolution")
    parser.add_argument("prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--width", type=int, default=1536, help="Image width (default: 1536)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations (default: 50)")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance (default: 3.5)")
    parser.add_argument("--seed", type=int, default=0, help="Seed start (default: random)")
    parser.add_argument("--num", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--sr", type=int, choices=[1, 2, 4, 8], default=8, help="Super resolution scale (1, 2, 4, or 8) (default: 8)")
    parser.add_argument("--sr-aura", action="store_true", default=False, help="Super resolution with Aura")

    args = parser.parse_args()

    generate_images(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        guidance=args.guidance,
        iterations=args.iterations,
        seed=args.seed if args.seed else random.randint(0, 2 ** 32 - 1),
        num_images=args.num,
        sr_scale=args.sr,
        sr_aura=args.sr_aura,
    )


if __name__ == "__main__":
    main()
