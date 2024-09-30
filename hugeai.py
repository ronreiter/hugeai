import os
import argparse
import random

import piexif
from PIL.Image import Image
from PIL import Image, PngImagePlugin

from aura_sr import AuraSR

import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline
from huggingface_hub import login
import dotenv
import RealESRGAN
import hashlib

torch.set_default_device("mps")

dotenv.load_dotenv()

login(token=os.environ.get("HF_TOKEN"))

# _flux_rope = diffusers.models.transformers.transformer_flux.rope
#
#
# def new_flux_rope(pos: torch.Tensor, dim: int, theta: int) -> torch.Tensor:
#     assert dim % 2 == 0, "The dimension must be even."
#     if pos.device.type == "mps":
#         return _flux_rope(pos.to("cpu"), dim, theta).to(device=pos.device)
#     else:
#         return _flux_rope(pos, dim, theta)
#
#
# diffusers.models.transformers.transformer_flux.rope = new_flux_rope


def apply_super_resolution(image, scale):
    model = RealESRGAN.RealESRGAN(torch.device("mps"), scale=scale)
    model.load_weights("FacehugmanIII/4x_foolhardy_Remacri/4x_foolhardy_Remacri.pth", download=True)
    model.load_weights(f'weights/RealESRGAN_x{scale}.pth', download=True)
    return model.predict(image)


def apply_super_resolution_aura(image):
    aura_sr = AuraSR.from_pretrained("fal/AuraSR-v2", device="mps")
    upscaled_image = aura_sr.upscale_4x_overlapped(image)
    return upscaled_image


def create_png_metadata(title, description):
    png_metadata = PngImagePlugin.PngInfo()
    png_metadata.add_text("Title", title)
    png_metadata.add_text("Description", description)
    png_metadata.add_text("Author", "Huge AI")
    return png_metadata


def create_jpeg_exif_metadata(title, description):
    exif_dict = {"0th": {}, "Exif": {}, "GPS": {}, "1st": {}, "thumbnail": None}
    exif_dict["0th"][piexif.ImageIFD.DocumentName] = title
    exif_dict["0th"][piexif.ImageIFD.ImageDescription] = description
    exif_dict["0th"][piexif.ImageIFD.Artist] = "Huge AI"
    exif_dict["0th"][piexif.ImageIFD.Copyright] = "Generated by Huge AI"
    return piexif.dump(exif_dict)


def generate_images(prompt, width, height, guidance, iterations, seed, num_images, sr_scale, sr_aura, format, model,
                    input_image=None):
    if format not in ['png', 'jpg', 'jpeg']:
        raise ValueError(f"Unsupported image format: {format}")

    if model not in ['dev', 'schnell']:
        raise ValueError(f"Unsupported model: {model}")

    FluxClass = FluxImg2ImgPipeline if input_image is not None else FluxPipeline
    flux_model = "black-forest-labs/FLUX.1-dev" if model == "dev" else "black-forest-labs/FLUX.1-schnell"

    # Load the appropriate pipeline
    pipe = FluxClass.from_pretrained(flux_model, torch_dtype=torch.bfloat16).to("mps")

    dir_name = hashlib.md5(prompt.encode()).hexdigest()
    os.makedirs(f"outputs/{dir_name}", exist_ok=True)

    for i in range(num_images):
        image_num = seed + i + 1
        base_filename = f"outputs/{dir_name}/w_{width}_h_{height}_num_{image_num}_it_{iterations}_guidance_{guidance}"
        filename = f"{base_filename}_sr_1.{format}"

        title = f"{prompt} --guidance {guidance} --iterations {iterations} --seed {seed} --model {model}"
        description = f"Prompt: {prompt}\nGuidance: {guidance}\nIterations: {iterations}\nSeed: {seed}\nModel: {model}"

        # Create metadata objects
        png_metadata = create_png_metadata(title, description)
        jpeg_exif_metadata = create_jpeg_exif_metadata(title, description)

        if os.path.exists(filename):
            out = Image.open(filename)
        else:
            # Generate image
            generator = torch.Generator("mps").manual_seed(image_num)

            if input_image:
                # Image-to-image generation
                out = pipe(
                    prompt=prompt,
                    image=input_image,
                    guidance_scale=guidance,
                    num_inference_steps=iterations,
                    generator=generator
                ).images[0]
            else:
                # Text-to-image generation
                out = pipe(
                    prompt=prompt,
                    guidance_scale=guidance,
                    width=width,
                    height=height,
                    num_inference_steps=iterations,
                    max_sequence_length=512,
                    generator=generator
                ).images[0]

            # Save with appropriate metadata format
            if format == 'png':
                out.save(filename, pnginfo=png_metadata)
            else:  # Handle jpeg
                out.save(filename, "JPEG", exif=jpeg_exif_metadata)
            print(f"Saved image: {filename}")
            out.show()

        # Apply super resolution if requested
        if sr_scale > 1:
            print(f"Upscaling {sr_scale}x...")
            out = apply_super_resolution(out, sr_scale)

            # Save upscaled image with metadata
            filename = f"{base_filename}_{'sr_' + str(sr_scale) if sr_scale else 'original'}.{format}"
            if format == 'png':
                out.save(filename, pnginfo=png_metadata)
            else:
                out.save(filename, "JPEG", exif=jpeg_exif_metadata)
            print(f"Saved image: {filename}")
        elif sr_aura:
            print(f"Upscaling with aura...")
            out = apply_super_resolution_aura(out)

            # Save upscaled image with metadata
            filename = f"{base_filename}_sr_aura.{format}"
            if format == 'png':
                out.save(filename, pnginfo=png_metadata)
            else:
                out.save(filename, "JPEG", exif=jpeg_exif_metadata)
            print(f"Saved image: {filename}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate images using FLUX or Schnell model with optional super resolution and image-to-image support")
    parser.add_argument("prompt", type=str, help="Prompt for image generation")
    parser.add_argument("--width", type=int, default=1536, help="Image width (default: 1536)")
    parser.add_argument("--height", type=int, default=1024, help="Image height (default: 1024)")
    parser.add_argument("--iterations", type=int, default=50, help="Iterations (default: 50)")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance (default: 3.5)")
    parser.add_argument("--seed", type=int, default=0, help="Seed start (default: random)")
    parser.add_argument("--num", type=int, default=1, help="Number of images to generate (default: 1)")
    parser.add_argument("--sr", type=int, choices=[1, 2, 4, 8], default=8,
                        help="Super resolution scale (1, 2, 4, or 8) (default: 8)")
    parser.add_argument("--sr-aura", action="store_true", default=False, help="Super resolution with Aura")
    parser.add_argument("--format", type=str, choices=["png", "jpg"], default="png", help="Image format (default: png)")
    parser.add_argument("--model", type=str, choices=["dev", "schnell"], default="dev",
                        help="Model to use (default: flux)")
    parser.add_argument("--input-image", type=str, help="Path to input image for image-to-image generation")

    args = parser.parse_args()

    input_image = Image.open(args.input_image) if args.input_image else None

    generate_images(
        prompt=args.prompt,
        width=args.width,
        height=args.height,
        guidance=args.guidance,
        iterations=args.iterations,
        seed=(args.seed - 1) if args.seed else random.randint(0, 2 ** 32 - 1),
        num_images=args.num,
        sr_scale=args.sr,
        sr_aura=args.sr_aura,
        format=args.format,
        model=args.model,
        input_image=input_image
    )


if __name__ == "__main__":
    main()
