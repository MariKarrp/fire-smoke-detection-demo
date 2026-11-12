import streamlit as st
import torch
from PIL import Image, ImageDraw
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
import torchvision
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import ToTensor

REPO_ROOT = Path(__file__).parent
YOLO_DIR = REPO_ROOT / "yolo"
sys.path.insert(0, str(YOLO_DIR))

@st.cache_resource
def load_weights():
    frcnn_path = hf_hub_download(
        repo_id="MariKarrp/cvteam19-fire-smoke-detection",
        filename="faster_rcnn_best.pth"
    )
    yolo_path = hf_hub_download(
        repo_id="MariKarrp/cvteam19-fire-smoke-detection",
        filename="selected_best.pt"
    )
    return frcnn_path, yolo_path

@st.cache_resource
def load_models():
    device = torch.device("cpu")
    
    frcnn_path, yolo_path = load_weights()
    
    frcnn = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    in_f = frcnn.roi_heads.box_predictor.cls_score.in_features
    frcnn.roi_heads.box_predictor = FastRCNNPredictor(in_f, 3)
    ckpt = torch.load(frcnn_path, map_location="cpu")
    frcnn.load_state_dict(ckpt["model_state_dict"])
    frcnn.to(device).eval()
    
    from ultralytics import YOLO
    yolo_wrapper = YOLO(yolo_path)
    yolo_core = yolo_wrapper.model.to(device).float().eval()
    
    return frcnn, yolo_core, device

def run_frcnn(model, image, device):
    img_t = ToTensor()(image).to(device)
    with torch.no_grad():
        out = model([img_t])[0]
    dets = []
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
        if score.item() >= 0.3:
            dets.append({"box": box.tolist(), "score": score.item(), "label": int(label.item())})
    return dets

def run_yolo(core, image, device):
    from final_yolo_protocol import predict_with_baseline_input
    with torch.inference_mode():
        out = predict_with_baseline_input(core, [image], device=str(device), score_floor=0.3)[0]
    dets = []
    for box, score, label in zip(out["boxes"], out["scores"], out["labels"]):
        dets.append({"box": box.tolist(), "score": score.item(), "label": int(label.item()) + 1})
    return dets

def draw_boxes(image, detections, base_color):
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        label = "smoke" if d["label"] == 1 else "fire"
        
        color = "red" if label == "fire" else "orange"
        
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        draw.text((x1 + 5, y1 + 5), f"{label} {d['score']:.2f}", fill=color)
    return img

st.set_page_config(page_title="Fire & Smoke Detection", layout="wide")
st.title("Fire & Smoke Detection")
st.markdown("Upload an image to compare **Faster R-CNN** and **YOLOv8s** predictions.")

frcnn, yolo_core, device = load_models()

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(image, caption="Original", use_container_width=True)
    
    with st.spinner("Running Faster R-CNN..."):
        frcnn_dets = run_frcnn(frcnn, image, device)
    with col2:
        st.image(draw_boxes(image, frcnn_dets, "red"), caption=f"Faster R-CNN ({len(frcnn_dets)} boxes)", use_container_width=True)
    with st.spinner("Running YOLOv8s..."):
        yolo_dets = run_yolo(yolo_core, image, device)
    with col3:
        st.image(draw_boxes(image, yolo_dets, "blue"), caption=f"YOLOv8s ({len(yolo_dets)} boxes)", use_container_width=True)