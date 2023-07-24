import torch
model_path = "./cyberrealistic_v32.safetensors"#@param {type:"string"}
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInpaintPipeline
device = "cuda" #@param ["cuda", "cpu"]
if model_path.endswith(".safetensors"):
    safetensors = True
#@title Gradio Interface
import IPython.display as ipd, gradio as gr, requests, PIL, numpy as np, os, uuid, cv2
from io import BytesIO
from PIL import Image
from torch import autocast
from matplotlib import pyplot as plt
from torchvision import transforms
import base64
import subprocess

latents = None
if not "loaded" in locals():
    txt2img = StableDiffusionPipeline.from_single_file(model_path, safety_checker=None, requires_safety_checker=False)
    img2img = StableDiffusionImg2ImgPipeline(**txt2img.components)
    inpainting = StableDiffusionInpaintPipeline(**txt2img.components)
    #txt2img.load_lora_weights(".", weight_name="InstantPhotoX3.safetensors", lora_weight=0.7)
    txt2img.to(device)
    img2img.to(device)
    inpainting.to(device)
    loaded = True

def empty(image):
    first_pixel = image.getpixel((0, 0))
    width, height = image.size
    for x in range(0, width, 4):
        for y in range(0, height, 4):
            pixel = image.getpixel((x, y))
            if pixel != first_pixel:
                return False
    return True

def predict(dict, prompt, negative_prompt,height,width, num_inference_steps, guidance_scale, reuse, strength):
    global latents
    if dict is None:
        choice = "Text To Image"
    else:
        img = dict["image"]
        mask = dict["mask"]
        if empty(mask):
            choice = "Image To Image"
        else:
            choice = "Inpainting"
    if choice == "Text To Image":
        if reuse =="New":
            generator = torch.Generator(device=device)
            latents = None
            seeds = []
            seed = generator.seed()
            seeds.append(seed)
            generator = generator.manual_seed(seed)
            image_latents = torch.randn(
                (1, txt2img.unet.in_channels, height // 8, width // 8),
                generator = generator,
                device = device
            )
            latents = image_latents if latents is None else torch.cat((latents, image_latents))
        elif reuse == "Reuse":
            if latents is None:
                generator = torch.Generator(device=device)
                latents = None
                seeds = []
                seed = generator.seed()
                seeds.append(seed)
                generator = generator.manual_seed(seed)
                image_latents = torch.randn(
                    (1, txt2img.unet.in_channels, height // 8, width // 8),
                    generator = generator,
                    device = device
                )
                latents = image_latents if latents is None else torch.cat((latents, image_latents))
        output = txt2img(prompt=prompt,
                          negative_prompt = negative_prompt,
                          height=height,
                          width=width,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          latents=latents
                          ).images[0]
    elif choice == "Image To Image":
        init_image = img.convert("RGB").resize((width,height))
        output = img2img(prompt=prompt,
                          image=init_image,
                          negative_prompt = negative_prompt,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          strength=strength,
                          ).images[0]
    elif choice == "Inpainting":
        img = img.convert("RGB").resize((width, height))
        mask = mask.convert("RGB").resize((width, height))
        output = inpainting(prompt=prompt,
                          image=img,
                          mask_image=mask,
                          negative_prompt = negative_prompt,
                          height=height,
                          width=width,
                          num_inference_steps=num_inference_steps,
                          guidance_scale=guidance_scale,
                          strength=strength,
                          ).images[0]
    return output

def feedback(img):
    return img

def save(img):
    img.save("temp.jpg")
    return "./temp.jpg"

codeforming = [
            "python",
            "CodeFormer/inference_codeformer.py",
            "-w",
            "$strength",
            "--input_path",
            "CodeFormer/inputs/temp.jpg",
            "--face_upsample",
            "--bg_upsampler",
            "realesrgan"
        ]

def upscale(dict, number, output):
    strength = float(number)
    str_strength = str(number)
    if output is not None:
        img = output
        img.save("./CodeFormer/inputs/temp.jpg")
        subprocess.run(codeforming, check=True)
        if str_strength == '1':
            out=Image.open(f"/content/results/test_img_1.0/final_results/temp.png")
            return out, f"/content/results/test_img_1.0/final_results/temp.png"
        else:
            out=Image.open(f"/content/results/test_img_{str_strength}/final_results/temp.png")
            return out, f"/content/results/test_img_{str_strength}/final_results/temp.png"
    else:
        if dict is None:
            return
        else:
            img = dict["image"]
            img.save("./CodeFormer/inputs/temp.jpg")
            subprocess.run(codeforming, check=True)
            if str_strength == '1':
                out=Image.open(f"/content/results/test_img_1.0/final_results/temp.png")
                return out, f"/content/results/test_img_1.0/final_results/temp.png"
            else:
                out=Image.open(f"/content/results/test_img_{str_strength}/final_results/temp.png")
                return out, f"/content/results/test_img_{str_strength}/final_results/temp.png"

image_blocks = gr.Blocks()
with image_blocks as demo:
    with gr.Group():
        with gr.Box():
            with gr.Row():
                image = gr.Image(source='upload', tool='sketch', type="pil", label="Upload")
                output_image = gr.Image(type="pil", label="Result")
            with gr.Row():
                prompt = gr.Textbox(placeholder = 'Your prompt (what you want in place of what is erased)', show_label=False, value=positive_template)
            with gr.Row():
                negative_prompt = gr.Textbox(placeholder = 'Negative prompt (what you do NOT want)', show_label=False, value=negative_template)
            with gr.Row():
                calculate_btn = gr.Button("Calculate")
                codeform_btn = gr.Button("CodeForm")
                feedback_btn = gr.Button("feedback")
            with gr.Row():
                height = gr.Slider(label="Height:",minimum=128, maximum=1024, step=8,value=512, interactive=True)
                width = gr.Slider(label="Width:",minimum=128, maximum=1024, step=8,value=512, interactive=True)
                calculate_btn.click(fn=lambda dict: (dict["image"].size[0] // 8 * 8, dict["image"].size[1] // 8 * 8), inputs=[image], outputs=[width, height])
                num_inference_steps = gr.Slider(label="Steps:",minimum=1,maximum=100,value=25, step=1, interactive=True)
                guidance_scale = gr.Slider(label="CFG:",minimum=1,maximum=20,value=7,step=0.1,interactive=True)
            with gr.Row():
                btn = gr.Button("Generate!").style(
                    margin=False,
                    rounded=(False, True, True, False),
                    full_width=False,
                )
                reuse = gr.Radio(label="Latent:",value="Reuse",choices=["New","Reuse"])
                strength = gr.Slider(label="Denoising:", value=0.75, minimum=0, maximum=1, step=0.01)
                btn.click(fn=predict, inputs=[image, prompt, negative_prompt,height,width,num_inference_steps,guidance_scale, reuse, strength], outputs=[output_image])
                save_button = gr.Button("Save")
            with gr.Row():
                download_menu = gr.File(label="Files:", file_types=['image'])
                codeform_btn.click(fn=upscale, inputs=[image, strength, output_image], outputs=[output_image, download_menu])
                save_button.click(fn=save, inputs=[output_image], outputs=[download_menu])
                feedback_btn.click(fn=feedback, inputs=[output_image], outputs=[image])

ipd.clear_output()
image_blocks.launch(share=False, debug=True)
