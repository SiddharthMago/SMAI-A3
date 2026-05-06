import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas

from model import SmallDevanagariCNN


st.set_page_config(
    page_title="Devanagari Digit Recognizer",
    page_icon="०",
    layout="centered",
)

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "models" / "devanagari_digit_cnn.pt"
META_PATH = APP_DIR / "models" / "class_metadata.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_DRAWING_TIPS = {
    0: "keep the loop closed and avoid adding tails that resemble another digit",
    1: "use a clear single main stroke with enough contrast from the background",
    2: "make the lower stroke distinct so it does not look like a rounded 3",
    3: "keep the upper curve and descending stroke clear without forming a 2-like base",
    4: "make the crossing structure and lower curve visible",
    5: "separate the top and lower strokes so it does not collapse into a 2-like shape",
    6: "keep the lower loop closed and the upper stroke compact",
    7: "use a clean diagonal/curved body without extra loops",
    8: "make both loops visible and balanced",
    9: "keep the top loop clear and avoid a lower stroke that resembles 2",
}

PAIR_FEEDBACK = {
    (2, 3): "This is a common confusion: a rounded top and right-side descending stroke can make 2 resemble 3. Emphasize the lower/base stroke of 2.",
    (3, 2): "The sample may have a stronger lower stroke than expected for 3, making it closer to 2. Try making the 3-like curve more continuous.",
    (5, 2): "The lower part may resemble the base of 2. Try making the distinctive top stroke of 5 clearer.",
    (9, 2): "The bottom stroke may be pulling the image toward 2. Try keeping the upper loop of 9 more dominant.",
    (4, 5): "The crossing and lower curve can resemble 5 if the central structure is unclear. Try separating the main strokes more clearly.",
}


@st.cache_resource
def load_model_and_metadata():
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    model = SmallDevanagariCNN(num_classes=len(metadata["class_names"]))
    try:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model, metadata


def crop_to_foreground(gray_array):
    foreground = gray_array > 20
    if not foreground.any():
        return gray_array

    rows = np.where(foreground.any(axis=1))[0]
    cols = np.where(foreground.any(axis=0))[0]
    top, bottom = rows[0], rows[-1]
    left, right = cols[0], cols[-1]
    return gray_array[top : bottom + 1, left : right + 1]


def should_invert(gray_array):
    border = np.concatenate(
        [
            gray_array[0, :],
            gray_array[-1, :],
            gray_array[:, 0],
            gray_array[:, -1],
        ]
    )
    return float(np.median(border)) > 127


def preprocess_image(image, polarity="light_on_dark"):
    gray = image.convert("L")
    arr = np.array(gray, dtype=np.uint8)

    if polarity == "auto":
        invert = should_invert(arr)
    else:
        invert = polarity == "dark_on_light"

    if invert:
        gray = ImageOps.invert(gray)
        arr = np.array(gray, dtype=np.uint8)

    cropped = crop_to_foreground(arr)

    side = max(cropped.shape)
    pad_y = side - cropped.shape[0]
    pad_x = side - cropped.shape[1]
    padded = np.pad(
        cropped,
        ((pad_y // 2, pad_y - pad_y // 2), (pad_x // 2, pad_x - pad_x // 2)),
        mode="constant",
        constant_values=0,
    )

    preview = Image.fromarray(padded).resize((32, 32), Image.Resampling.LANCZOS)
    tensor = np.array(preview, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(tensor).unsqueeze(0).unsqueeze(0)
    return tensor, preview


def predict(image, model, metadata, polarity="light_on_dark"):
    x, preview = preprocess_image(image, polarity=polarity)
    x = x.to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

    pred = int(np.argmax(probs))
    return {
        "class_index": pred,
        "digit": metadata["devanagari_digits"][pred],
        "unicode": metadata["unicode_codes"][pred],
        "confidence": float(probs[pred]),
        "probabilities": probs,
        "preview": preview,
    }


def foreground_stats(preview):
    arr = np.array(preview, dtype=np.uint8)
    foreground = arr > 20
    if not foreground.any():
        return {"ink_ratio": 0.0, "center_offset": 1.0}

    ys, xs = np.where(foreground)
    ink_ratio = float(foreground.mean())
    center_y = float(ys.mean()) / max(arr.shape[0] - 1, 1)
    center_x = float(xs.mean()) / max(arr.shape[1] - 1, 1)
    center_offset = abs(center_x - 0.5) + abs(center_y - 0.5)
    return {"ink_ratio": ink_ratio, "center_offset": center_offset}


def practice_feedback(result, target, metadata):
    pred = result["class_index"]
    probs = result["probabilities"]
    stats = foreground_stats(result["preview"])
    target_prob = float(probs[target])
    pred_prob = float(probs[pred])
    gap = pred_prob - target_prob

    target_label = f"{metadata['devanagari_digits'][target]} ({metadata['unicode_codes'][target]})"
    pred_label = f"{metadata['devanagari_digits'][pred]} ({metadata['unicode_codes'][pred]})"

    if pred == target:
        return [
            f"The drawing matches the target {target_label}. The model assigned {pred_prob:.1%} confidence to this class.",
            "For more stable predictions, keep the digit centered and use a complete, high-contrast stroke.",
        ]

    feedback = [
        f"The model favored {pred_label} over the target {target_label}: {pred_prob:.1%} vs {target_prob:.1%} confidence.",
        PAIR_FEEDBACK.get(
            (target, pred),
            f"To move the prediction toward the target, try to emphasize the target digit's distinctive structure: {CLASS_DRAWING_TIPS[target]}.",
        ),
    ]

    if gap < 0.15:
        feedback.append("The probabilities are close, so this is an ambiguous sample; a small stroke change may be enough.")
    if stats["ink_ratio"] < 0.05:
        feedback.append("The digit looks very thin or incomplete after preprocessing; draw with a fuller continuous stroke.")
    elif stats["ink_ratio"] > 0.45:
        feedback.append("The digit fills a large part of the 32x32 input; leaving more empty margin may help.")
    if stats["center_offset"] > 0.30:
        feedback.append("The digit is off-center after preprocessing; centering it can improve recognition.")

    return feedback


model, metadata = load_model_and_metadata()
digits = metadata["devanagari_digits"]
unicode_codes = metadata["unicode_codes"]

st.title("Handwritten Devanagari Digit Recognition")

with st.sidebar:
    input_mode = st.radio("Input mode", ["Draw", "Upload"])
    practice_mode = st.checkbox("Practice mode")
    target = None
    if practice_mode:
        target = st.selectbox(
            "Target digit",
            list(range(10)),
            format_func=lambda i: f"{i} -> {digits[i]} ({unicode_codes[i]})",
        )
    upload_polarity = st.selectbox(
        "Uploaded image polarity",
        ["auto", "dark_on_light", "light_on_dark"],
        format_func={
            "auto": "Auto",
            "dark_on_light": "Dark digit on light background",
            "light_on_dark": "Light digit on dark background",
        }.get,
    )

input_image = None

if input_mode == "Draw":
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 0)",
        stroke_width=18,
        stroke_color="#FFFFFF",
        background_color="#000000",
        width=280,
        height=280,
        drawing_mode="freedraw",
        key="canvas",
        update_streamlit=True,
    )

    if canvas_result.image_data is not None:
        arr = canvas_result.image_data.astype(np.uint8)
        input_image = Image.fromarray(arr).convert("RGB")
else:
    uploaded = st.file_uploader("Upload a digit image", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        input_image = Image.open(uploaded).convert("RGB")
        st.image(input_image, caption="Uploaded image", width=180)

if input_image is not None:
    result = predict(
        input_image,
        model,
        metadata,
        polarity=upload_polarity if input_mode == "Upload" else "light_on_dark",
    )

    left, right = st.columns([1, 2])
    with left:
        st.image(result["preview"].resize((128, 128)), caption="32x32 model input")

    with right:
        st.metric("Prediction", f"{result['digit']} ({result['unicode']})")
        st.write(f"Class index: **{result['class_index']}**")
        st.write(f"Confidence: **{result['confidence']:.2%}**")

        if practice_mode:
            if result["class_index"] == target:
                st.success("Correct target digit.")
            else:
                st.warning(f"Target was {digits[target]} ({unicode_codes[target]}).")

            with st.expander("Practice feedback", expanded=result["class_index"] != target):
                for note in practice_feedback(result, target, metadata):
                    st.write(f"- {note}")

    chart_data = pd.DataFrame(
        {
            "digit": [f"{i} {digits[i]}" for i in range(10)],
            "probability": result["probabilities"],
        }
    )
    st.bar_chart(chart_data, x="digit", y="probability")

st.caption(
    "Model: 3-layer PyTorch CNN trained from scratch on the UCI Devanagari digit subset."
)
