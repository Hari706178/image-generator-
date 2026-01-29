import streamlit as st
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import io
import time

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="AI Image Generator",
    page_icon="🎨",
    layout="wide"
)

st.title("🎨 AI Image Generator")
st.caption("Stable Diffusion based Image Generation with Prompt History & Gallery")

# -------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------
if "history" not in st.session_state:
    st.session_state["history"] = []

if "gallery" not in st.session_state:
    st.session_state["gallery"] = []

# -------------------------------------------------
# DEVICE SETUP
# -------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
st.sidebar.success(f"Device: {DEVICE.upper()}")

# -------------------------------------------------
# LOAD MODEL (SAFE CACHING)
# -------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_sd_model():
    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
    )
    pipe = pipe.to(DEVICE)
    pipe.enable_attention_slicing()
    return pipe

pipe = load_sd_model()

# -------------------------------------------------
# SIDEBAR SETTINGS
# -------------------------------------------------
st.sidebar.header("⚙️ Generation Settings")

steps = st.sidebar.slider("Inference Steps", 20, 50, 30)
guidance = st.sidebar.slider("Guidance Scale", 5.0, 12.0, 7.5)
seed = st.sidebar.number_input("Seed", value=42, step=1)

style = st.sidebar.selectbox(
    "Image Style",
    ["Photorealistic", "Cinematic", "Anime", "Fantasy Art", "Digital Painting"]
)

STYLE_PROMPTS = {
    "Photorealistic": "ultra realistic, DSLR photography, natural lighting, sharp focus",
    "Cinematic": "cinematic lighting, dramatic shadows, movie still, volumetric light",
    "Anime": "anime style, clean lines, vibrant colors, high detail",
    "Fantasy Art": "epic fantasy artwork, magical atmosphere, concept art",
    "Digital Painting": "digital painting, smooth brush strokes, artistic style"
}

# -------------------------------------------------
# PROMPT INPUTS
# -------------------------------------------------
prompt = st.text_area(
    "📝 Prompt",
    placeholder="A futuristic city at sunset"
)

negative_prompt = st.text_input(
    "🚫 Negative Prompt",
    value="blurry, low quality, distorted, bad anatomy, extra fingers"
)

enhance_prompt = st.checkbox("✨ Enhance prompt (Professional)", value=True)

# -------------------------------------------------
# IMAGE GENERATION
# -------------------------------------------------
if st.button("🚀 Generate Image", use_container_width=True):

    if prompt.strip() == "":
        st.warning("Please enter a prompt.")
    else:
        with st.spinner("Generating image..."):
            start_time = time.time()

            if enhance_prompt:
                final_prompt = (
                    f"{prompt}, {STYLE_PROMPTS[style]}, "
                    "highly detailed, professional quality, 8k"
                )
            else:
                final_prompt = prompt

            generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

            image = pipe(
                prompt=final_prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator
            ).images[0]

            duration = time.time() - start_time

        # SAVE HISTORY
        st.session_state.history.append({
            "prompt": prompt,
            "style": style,
            "seed": seed
        })

        # SAVE IMAGE TO GALLERY
        st.session_state.gallery.append(image)

        # DISPLAY IMAGE
        st.image(image, caption=f"Generated in {duration:.2f} seconds", use_column_width=True)

        # DOWNLOAD BUTTON
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)

        st.download_button(
            "📥 Download Image",
            data=buffer,
            file_name="generated_image.png",
            mime="image/png"
        )

# -------------------------------------------------
# PROMPT HISTORY
# -------------------------------------------------
st.markdown("## 🧠 Prompt History")

if len(st.session_state.history) == 0:
    st.info("No prompts generated yet.")
else:
    for i, h in enumerate(reversed(st.session_state.history), 1):
        st.markdown(
            f"**{i}.** `{h['prompt']}`  \n"
            f"Style: *{h['style']}* | Seed: `{h['seed']}`"
        )

# -------------------------------------------------
# IMAGE GALLERY
# -------------------------------------------------
st.markdown("## 🖼️ Image Gallery")

if len(st.session_state.gallery) == 0:
    st.info("No images generated yet.")
else:
    cols = st.columns(3)
    for idx, img in enumerate(reversed(st.session_state.gallery)):
        with cols[idx % 3]:
            st.image(img, use_column_width=True)

# -------------------------------------------------
# FOOTER
# -------------------------------------------------
st.markdown("---")
st.caption("Built with Stable Diffusion • Streamlit • Hugging Face Diffusers")


