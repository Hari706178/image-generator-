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
# SESSION STATE
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
# SIDEBAR
# =========================================================
st.sidebar.header("⚙️ Settings")

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
    "Anime": "anime style, vibrant colors, clean lines",
    "Fantasy Art": "epic fantasy artwork, magical atmosphere, concept art",
    "Digital Painting": "digital painting, smooth brush strokes, artistic"
}

# =========================================================
# INPUTS
# =========================================================
prompt = st.text_area("📝 Enter prompt")
negative_prompt = st.text_input(
    "🚫 Negative Prompt",
    "blurry, low quality, distorted, bad anatomy"
)
enhance = st.checkbox("✨ Professional Enhancement", value=True)

# =========================================================
# GENERATE
# =========================================================
if st.button("🚀 Generate Image"):

    if not prompt.strip():
        st.warning("Please enter a prompt")
    else:
        with st.spinner("Generating image..."):
            start = time.time()

            final_prompt = (
                f"{prompt}, {STYLE_MAP[style]}, highly detailed, professional quality"
                if enhance else prompt
            )

            generator = torch.manual_seed(seed)

            image = pipe(
                final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=guidance,
                generator=generator
            ).images[0]

        st.session_state.history.append({
            "prompt": prompt,
            "style": style,
            "seed": seed
        })
        st.session_state.gallery.append(image)

        st.image(image, use_column_width=True)
        st.success(f"Done in {round(time.time()-start, 2)}s")

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        buf.seek(0)

        st.download_button("📥 Download", buf, "image.png", "image/png")

# =========================================================
# HISTORY
# =========================================================
st.markdown("## 🧠 Prompt History")
for h in reversed(st.session_state.history):
    st.markdown(f"- **{h['prompt']}** | {h['style']} | Seed {h['seed']}")

# =========================================================
# GALLERY
# =========================================================
st.markdown("## 🖼️ Gallery")
cols = st.columns(3)
for i, img in enumerate(reversed(st.session_state.gallery)):
    cols[i % 3].image(img, use_column_width=True)

