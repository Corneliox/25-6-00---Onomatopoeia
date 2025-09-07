import os
import torch
import gradio as gr
import requests
import zipfile
import random
from datetime import datetime
from diffusers import AutoPipelineForText2Image

# --- Setup ---
os.makedirs("outputs", exist_ok=True)
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("loras", exist_ok=True)

checkpoint_map = {
    "SDXL 1.0 Base": "stabilityai/stable-diffusion-xl-base-1.0",
    "Juggernaut XL v9": "https://civitai.com/api/download/models/257749"
}

# Add LoRA 1–10 + Final
lora_map = {"None": ""}
lora_base_url = "https://huggingface.co/Corneliox/LoraOnomatopoeia/resolve/main/SDXL/3050_v4_sdxl"
lora_map["Onomatopoeia SDXL v4 (Final)"] = f"{lora_base_url}.safetensors"
for i in range(1, 11):
    lora_map[f"Epoch {i}"] = f"{lora_base_url}-00000{i}.safetensors"

# --- Utils ---
def download_file(url, save_dir, file_name):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_path = os.path.join(save_dir, file_name)
    if not os.path.exists(file_path):
        print(f"Downloading {file_name}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Downloaded {file_name} successfully.")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading {file_name}: {e}")
            return None
    else:
        print(f"{file_name} already exists.")
    return file_path

# --- Global Vars ---
pipe = None
pipe_refiner = None
current_model_identifier = ""

# --- Core ---
def generate_image(
    checkpoint_name, custom_checkpoint_url,
    lora_name, lora_weight, run_all_loras,
    positive_prompt, negative_prompt,
    width, height, batch_size,
    seed_value, control_after_generate,
    steps, cfg, device_choice,
    use_refiner
):
    global pipe, pipe_refiner, current_model_identifier
    device = "cuda" if device_choice == "GPU" else "cpu"
    torch_dtype = torch.float16 if device_choice == "GPU" else torch.float32

    # --- Checkpoint ---
    yield None, seed_value, gr.File(visible=False), "1/6: Finding model..."
    is_single_file = bool(custom_checkpoint_url)
    if custom_checkpoint_url:
        model_identifier = download_file(custom_checkpoint_url, "./checkpoints", custom_checkpoint_url.split('/')[-1].split('?')[0])
    else:
        model_identifier = checkpoint_map[checkpoint_name]
        if model_identifier.startswith("http"):
            is_single_file = True
            model_identifier = download_file(model_identifier, "./checkpoints", f"{checkpoint_name.replace(' ', '_')}.safetensors")

    if not model_identifier:
        yield None, seed_value, gr.File(visible=False), "Gagal menemukan checkpoint."
        return

    # --- Load pipeline ---
    yield None, seed_value, gr.File(visible=False), "2/6: Loading pipeline..."
    if model_identifier != current_model_identifier:
        try:
            if pipe is not None: del pipe; torch.cuda.empty_cache()
            if is_single_file:
                pipe = AutoPipelineForText2Image.from_single_file(model_identifier, torch_dtype=torch_dtype, variant="fp16", use_safetensors=True).to(device)
            else:
                pipe = AutoPipelineForText2Image.from_pretrained(model_identifier, torch_dtype=torch_dtype, variant="fp16", use_safetensors=True).to(device)
            pipe.enable_model_cpu_offload()
            current_model_identifier = model_identifier
        except Exception as e:
            yield None, seed_value, gr.File(visible=False), f"Error loading pipeline: {e}"
            return

        if use_refiner:
            try:
                pipe_refiner = AutoPipelineForText2Image.from_pretrained(
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                ).to(device)
                pipe_refiner.enable_model_cpu_offload()
            except Exception as e:
                yield None, seed_value, gr.File(visible=False), f"Error loading refiner: {e}"
                return

    # --- Seed ---
    if int(seed_value) == -1:
        seed_value = random.randint(0, 2**32 - 1)
    generator = torch.Generator(device).manual_seed(int(seed_value))

    all_outputs = []
    all_status = []

    # --- Function for one LoRA run ---
    def run_one_lora(lora_label, lora_url):
        pipe.unload_lora_weights()
        cross_attention_kwargs = None
        if lora_url:
            lora_filename = lora_url.split('/')[-1]
            lora_path = download_file(lora_url, "./loras", lora_filename)
            if lora_path:
                pipe.load_lora_weights(lora_path, adapter_name=lora_label)
                cross_attention_kwargs = {"scale": lora_weight}

        images = pipe(
            prompt=positive_prompt,
            negative_prompt=negative_prompt,
            width=width, height=height,
            num_inference_steps=steps,
            guidance_scale=cfg,
            num_images_per_prompt=batch_size,
            generator=generator,
            cross_attention_kwargs=cross_attention_kwargs
        ).images

        if use_refiner and pipe_refiner is not None:
            images = pipe_refiner(
                prompt=positive_prompt,
                negative_prompt=negative_prompt,
                image=images,
                num_inference_steps=20
            ).images

        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        lora_dir = os.path.join("outputs", f"{timestamp}_{lora_label.replace(' ', '_')}")
        os.makedirs(lora_dir, exist_ok=True)
        image_paths = []
        for i, img in enumerate(images):
            file_path = os.path.join(lora_dir, f"{lora_label}_image_{i+1}_seed{seed_value}.png")
            img.save(file_path)
            image_paths.append(file_path)

        return [f"===== {lora_label} ====="] + image_paths, f"Saved: {lora_dir}"

    # --- Run all LoRAs or single ---
    if run_all_loras:
        for name, url in lora_map.items():
            if name != "None":
                gallery, status = run_one_lora(name, url)
                all_outputs.extend(gallery)
                all_status.append(status)
                yield all_outputs, seed_value, gr.File(visible=False), "\n".join(all_status)
    else:
        gallery, status = run_one_lora(lora_name, lora_map[lora_name])
        all_outputs.extend(gallery)
        all_status.append(status)

    # Seed control
    if control_after_generate == 'increment': seed_value = int(seed_value) + 1
    elif control_after_generate == 'decrement': seed_value = int(seed_value) - 1
    elif control_after_generate == 'randomize': seed_value = -1

    yield all_outputs, seed_value, gr.File(visible=False), "\n".join(all_status)

# --- UI ---
manual_content = """
### **Panduan SDXL Generator**
- Aktifkan **Run All LoRAs** untuk menghasilkan semua versi sekaligus.
- Centang **Use Refiner** agar hasil lebih halus.
- Hasil tersimpan di folder `outputs/`.
"""

with gr.Blocks(css="#col-container {margin: 0 auto; max-width: 800px;}") as demo:
    gr.Markdown("# 🎨 Onomatopoeia Image Generator (SDXL + Refiner)")

    with gr.Row():
        with gr.Column(scale=2):
            with gr.Accordion("1. Model & LoRA", open=True):
                checkpoint_name = gr.Dropdown(list(checkpoint_map.keys()), value="SDXL 1.0 Base", label="Checkpoint")
                custom_checkpoint_url = gr.Textbox(label="Custom Checkpoint URL (opsional)")
                lora_name = gr.Dropdown(list(lora_map.keys()), value="Onomatopoeia SDXL v4 (Final)", label="Pilih LoRA")
                lora_weight = gr.Slider(0, 1.5, 0.05, value=0.8, label="LoRA Weight")
                run_all_loras = gr.Checkbox(label="Run All LoRAs", value=False)
                device_choice = gr.Dropdown(["GPU", "CPU"], value="GPU", label="Device")
                use_refiner = gr.Checkbox(label="Use SDXL Refiner", value=True)

            with gr.Accordion("2. Prompts", open=True):
                positive_prompt = gr.Textbox(value='Onomatopoeia_style, Comic art onomatopoeia "duuttt"', label="Positive Prompt", lines=3)
                negative_prompt = gr.Textbox(value='(worst quality, low quality:1.4), blurry, watermark, signature', label="Negative Prompt", lines=3)

            with gr.Accordion("3. Image Settings", open=True):
                with gr.Row():
                    width = gr.Slider(768, 1536, 64, value=1024, label="Width")
                    height = gr.Slider(768, 1536, 64, value=1024, label="Height")
                batch_size = gr.Slider(1, 4, 1, value=2, label="Batch Size")

            with gr.Accordion("4. Sampler", open=True):
                steps = gr.Slider(10, 100, 1, value=60, label="Steps")
                cfg = gr.Slider(1.0, 20.0, 0.5, value=9, label="CFG Scale")
                with gr.Row():
                    seed_value = gr.Number(value=-1, label="Seed (-1 random)", precision=0)
                    control_after_generate = gr.Radio(['fixed','increment','decrement','randomize'], value='increment', label="Seed Control")

            run_button = gr.Button("Generate", variant="primary")

        with gr.Column(scale=3):
            with gr.Tabs():
                with gr.TabItem("Results"):
                    output_gallery = gr.Gallery(label="Generated Images", show_label=False, columns=2, object_fit="contain")
                    zip_output_file = gr.File(label="Download Zips", visible=False)
                    status_text = gr.Textbox(label="Status", lines=4, interactive=False)
                with gr.TabItem("Guide"):
                    gr.Markdown(manual_content)

    run_button.click(
        fn=generate_image,
        inputs=[checkpoint_name, custom_checkpoint_url, lora_name, lora_weight, run_all_loras,
                positive_prompt, negative_prompt, width, height, batch_size,
                seed_value, control_after_generate, steps, cfg, device_choice, use_refiner],
        outputs=[output_gallery, seed_value, zip_output_file, status_text]
    )

demo.queue().launch(debug=True, share=False)
