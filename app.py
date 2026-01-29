# =========================================================
# AUTO INSTALL REQUIRED LIBRARIES
# =========================================================
import subprocess
import sys

def install(pkg):
    subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

required_libs = [
    "streamlit",
    "torch",
    "diffusers",
    "transformers",
    "accelerate",
    "safetensors",
    "pillow"
]

for lib in required_libs:
    try:
        __import__(lib)
    except ImportError:
        install(lib)

# =========================================================
# IMPORTS
# =========================================================
import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import time

# =========================================================
# PAGE CONFIG
# =========================================================
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI Image Generator")
st.write("Fast, Professional & Interactive Diffusion Image Generator")

# =========================================================
# SESSION STATE (HISTORY & GALLERY)
# =========================================================
if "history" not in st.session_state:
    st.session_state.history = []

if "gallery" not in st.session_state:
    st.session_state.gallery = []

# =========================================================
# DEVICE
# =========================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Device: {device.upper()}")

# =========================================================
# LOAD MODEL
# =========================================================
@st.cache_resource
def load_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32
    )
    pipe.to(device)
    return pipe

pipe = load_model()

# =========================================================
# SIDEBAR SETTINGS
# =========================================================
st.sidebar.header("⚙️ Generation Settings")

steps = st.sidebar.slider("Inference Steps", 20, 50, 30)
guidance = st.sidebar.slider("Guidance Scale", 5.0, 12.0, 7.5)
seed = st.sidebar.number_input("Seed", value=42, step=1)

style = st.sidebar.selectbox(
    "🎭 Style",
    ["Photorealistic", "Cinematic", "Anime", "Fantasy Art", "Digital Painting"]
)

STYLE_MAP = {
    "Photorealistic": "ultra realistic, DSLR photography, sharp focus, natural lighting",
    "Cinematic": "cinematic lighting, dramatic shadows, movie still, volumetric light",
    "Anime": "anime style, vibrant colors, clean lines, studio ghibli inspired",
    "Fantasy Art": "epic fantasy artwork, magical atmosphere, concept art, artstation",
    "Digital Painting": "digital painting, smooth brush strokes, high detail, artistic"
}

# =========================================================
# MAIN INPUT AREA
# =========================================================
prompt = st.text_area(
    "📝 Enter your prompt",
    placeholder="A futuristic city at sunset"
)

negative_prompt = st.text_input(
    "🚫 Negative Prompt",
    value="blurry, low quality, distorted, extra fingers, bad anatomy"
)

enhance = st.checkbox("✨ Professional Prompt Enhancement", value=True)

# =========================================================
# GENERATE IMAGE
# =========================================================
if st.button("🚀 Generate Image"):

    if prompt.strip() == "":
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating image... ⏳"):
            start = time.time()

            style_prompt = STYLE_MAP[style]

            if enhance:
                final_prompt = f"""
                {prompt},
                {style_prompt},
                8k resolution, highly detailed,
                professional quality, realistic textures
                """
            else:
                final_prompt = prompt

            generator = torch.manual_seed(seed)

            image = pipe(
                final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]

            end = time.time()

        # SAVE HISTORY
        st.session_state.history.append({
            "prompt": prompt,
            "style": style,
            "seed": seed
        })

        # SAVE GALLERY
        st.session_state.gallery.append(image)

        st.image(image, caption="Generated Image", use_column_width=True)
        st.success(f"Generated in {round(end - start, 2)} seconds")

        # DOWNLOAD
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            "📥 Download Image",
            buffer,
            file_name="generated_image.png",
            mime="image/png"
        )

# =========================================================
# PROMPT HISTORY
# =========================================================
st.markdown("## 🧠 Prompt History")

if len(st.session_state.history) == 0:
    st.info("No prompts yet")
else:
    for i, item in enumerate(reversed(st.session_state.history), 1):
        st.markdown(
            f"**{i}.** `{item['prompt']}`  \n"
            f"Style: *{item['style']}* | Seed: `{item['seed']}`"
        )

# =========================================================
# IMAGE GALLERY
# =========================================================
st.markdown("## 🖼️ Image Gallery")

if len(st.session_state.gallery) == 0:
    st.info("No images generated yet")
else:
    cols = st.columns(3)
    for idx, img in enumerate(reversed(st.session_state.gallery)):
        with cols[idx % 3]:
            st.image(img, use_column_width=True)

# =========================================================
# FOOTER
# =========================================================
st.markdown("---")
st.caption(
    "Built with ❤️ using Stable Diffusion • Streamlit • Hugging Face Diffusers"
)
