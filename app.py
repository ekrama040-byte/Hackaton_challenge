import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import numpy as np
import pandas as pd
import sqlite3
import datetime
import streamlit as st

st.set_page_config(page_title="AgriVision Sovereign Command Engine", layout="wide")

# ==============================================================================
# ADVANCED ENTERPRISE CUSTOM STYLING (CSS DESIGN LAYER)
# ==============================================================================
st.markdown("""
    <style>
        .stApp {
            background-color: #0b0f19 !important;
            color: #e2e8f0 !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        h1, h2, h3 {
            color: #00f2fe !important;
            font-weight: 700 !important;
            letter-spacing: 0.5px;
            text-transform: uppercase;
            border-bottom: 1px solid rgba(0, 242, 254, 0.15);
            padding-bottom: 8px;
        }
        section[data-testid="stSidebar"] {
            background-color: #0d1527 !important;
            border-right: 2px solid #1e293b !important;
        }
        div[data-testid="metric-container"] {
            background-color: #111a2e !important;
            border: 1px solid #1e293b !important;
            border-left: 4px solid #00f2fe !important;
            padding: 15px !important;
            border-radius: 6px !important;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        }
        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
            font-size: 0.85rem !important;
            text-transform: uppercase !important;
            font-weight: 600 !important;
        }
        div[data-testid="stMetricValue"] {
            color: #ffffff !important;
            font-size: 1.6rem !important;
            font-weight: 700 !important;
        }
        .stAlert {
            background-color: #111a2e !important;
            border: 1px solid #1e293b !important;
            border-radius: 6px !important;
        }
        .stDataFrame, div[data-testid="stTable"] {
            background-color: #111a2e !important;
            border: 1px solid #1e293b !important;
            border-radius: 6px !important;
        }
    </style>
""", unsafe_allow_html=True)

# ==============================================================================
# INDUSTRIAL EXPLAINABLE AI (GRAD-CAM) ENGINE - DIRECT TENSOR HOOK METHOD
# ==============================================================================
class GradientActivationMapping:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Use only a forward hook; the backward hook is registered directly onto the tensor
        self.target_layer.register_forward_hook(self.save_activations)

    def save_activations(self, module, input, output):
        self.activations = output.detach()
        
        # Foolproof Tensor Hook: Triggers on the tensor itself, always receiving a single tensor, never a tuple
        def backward_tensor_hook(grad):
            self.gradients = grad.detach()
            return grad
            
        output.register_hook(backward_tensor_hook)

    def compute_heatmap(self, input_tensor, class_idx):
        self.gradients = None
        self.model.zero_grad()
        output = self.model(input_tensor)
        score = output[0, class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            return np.zeros((224, 224), dtype=np.float32)
            
        gradients = self.gradients.cpu().numpy()
        activations = self.activations.cpu().numpy()
        weights = np.mean(gradients, axis=(2, 3))
        cam = np.zeros(activations.shape[2:], dtype=np.float32)
        
        for i, w in enumerate(weights[0]):
            cam += w * activations[0, i, :, :]
            
        cam = np.maximum(cam, 0)
        cam_max = np.max(cam)
        if cam_max > 0:
            cam = cam / cam_max
        return cam

def generate_cam_overlay(pil_image, heatmap):
    heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8)).resize((224, 224), resample=Image.BILINEAR)
    heatmap_np = np.array(heatmap_resized)
    r = heatmap_np
    g = np.zeros_like(heatmap_np)
    b = 255 - heatmap_np
    heatmap_rgb = Image.fromarray(np.stack([r, g, b], axis=2))
    raw_resized = pil_image.resize((224, 224))
    blended_output = Image.blend(raw_resized, heatmap_rgb, alpha=0.35)
    return np.array(blended_output)

# ==============================================================================
# ENTERPRISE PERSISTENT DATA LAYER (SQLITE3 HARDWARE BOUND)
# ==============================================================================
def init_db():
    conn = sqlite3.connect("agrivision_sovereign.db", check_same_thread=False)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS field_telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            condition TEXT,
            confidence REAL,
            hectares REAL,
            soil_ph REAL,
            vpd REAL,
            required_kg REAL,
            procurement_cost REAL
        )
    """)
    conn.commit()
    return conn

db_conn = init_db()

# ==============================================================================
# INDUSTRIAL GENERATIVE AGRO-COPILOT DICTIONARY
# ==============================================================================
def get_agronomist_advice(disease_name, total_kg, cost, vpd):
    parts = disease_name.split("___")
    crop = parts[0].replace("_", " ") if len(parts) > 0 else "Crop Tissue"
    condition = parts[1].replace("_", " ") if len(parts) > 1 else "Unknown Matrix Condition"
    
    if "healthy" in condition.lower():
        return {
            "pathogen": f"None Detected ({crop} Matrix Stable)",
            "symptoms": f"Uniform chlorophyll distribution across {crop} foliage. Cellular respiration nominal.",
            "transmission": f"Microclimate air metrics (VPD: {vpd:.2f} kPa) are currently maintaining physiological balance.",
            "strategy": "No aggressive chemical intervention needed. Maintain baseline soil nitrogen feeding profiles."
        }
        
    return {
        "pathogen": f"Active Pathogen Spore/Cell Micro-colonies hitting local {crop} vascular sectors.",
        "symptoms": f"Necrotic lesion formatting clusters, spotting, or leaf rust anomalies matching {condition}.",
        "transmission": f"Pathogen reproduction track high. Localized vapor deficit ({vpd:.2f} kPa) speeds up vector spreads.",
        "strategy": f"Deploy chemical mitigation treatments immediately. Apply {total_kg:.2f} total units at a procurement supply-line cost of ${cost:.2f}."
    }

# ==============================================================================
# GATEWAY LAYER: INDUSTRIAL INPUT VALIDATION MASK
# ==============================================================================
@st.cache_resource
def load_validation_filter():
    weights = models.MobileNet_V3_Small_Weights.DEFAULT
    filter_model = models.mobilenet_v3_small(weights=weights)
    filter_model.eval()
    return filter_model, weights.meta["categories"]

filter_node, imagenet_classes = load_validation_filter()

def verify_agricultural_integrity(img):
    preprocess = models.MobileNet_V3_Small_Weights.DEFAULT.transforms()
    input_tensor = preprocess(img).unsqueeze(0)
    with torch.no_grad():
        logits = filter_node(input_tensor)
        probabilities = F.softmax(logits, dim=1).squeeze().numpy()
    top5_indices = np.argsort(probabilities)[-5:][::-1]
    
    top1_label = imagenet_classes[top5_indices[0]].lower()
    
    explicit_violations = ["identity card", "passport", "web site", "website", "screen", "monitor", "car", "truck"]
    if any(v in top1_label for v in explicit_violations):
        return False, top1_label, probabilities[top5_indices[0]]
        
    valid_keywords = ["leaf", "plant", "crop", "corn", "maize", "tree", "vegetable", "grass", "pot", "fungus", "chameleon", "lizard", "bittern"]
    for idx in top5_indices:
        if any(kw in imagenet_classes[idx].lower() for kw in valid_keywords):
            return True, imagenet_classes[idx].lower(), probabilities[idx]
    return False, top1_label, probabilities[top5_indices[0]]

# ==============================================================================
# CORE SYSTEM ENGINE: MULTI-CLASS TRANSFER LEARNING MODEL BLOCK
# ==============================================================================
PLANTVILLAGE_CLASSES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry___Powdery_mildew", "Cherry___healthy", "Corn___Cercospora_leaf_spot",
    "Corn___Common_rust", "Corn___Northern_Leaf_Blight", "Corn___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight", "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus", "Tomato___healthy"
]

@st.cache_resource
def load_trained_sovereign_engine():
    model = models.mobilenet_v3_small()
    model.classifier = nn.Sequential(
        nn.Linear(576, 1024),
        nn.Hardswish(),
        nn.Dropout(p=0.2, inplace=True),
        nn.Linear(1024, 38)
    )

    weights_path = "models/plant_disease_model.pth"
    if os.path.exists(weights_path):
        try:
            model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
            st.sidebar.success("⚡ AgriVision Core Engine Activated: Trained Weights Embedded.")
        except Exception as e:
            st.sidebar.error(f"❌ Structural loading mismatch error: {e}")
    else:
        st.sidebar.warning("⚠️ Weights file missing. Operating on baseline untrained model.")
        
    model.eval()
    return model

sovereign_engine = load_trained_sovereign_engine()

# ==============================================================================
# FRONTEND USER INTERFACE RENDER BLOCK
# ==============================================================================
st.title("🛰️ AgriVision Sovereign Command Engine")

# Sidebar Controls
st.sidebar.header("Telemetry Calibration Controls")
field_scale = st.sidebar.slider("Field Operational Scale (Hectares)", 0.5, 50.0, 5.0, step=0.5)
soil_ph = st.sidebar.slider("Current Ground Soil pH Metric", 4.0, 9.0, 6.5, step=0.1)
air_vpd = st.sidebar.slider("Microclimate Vapor Pressure Deficit (VPD - kPa)", 0.1, 3.5, 1.2, step=0.1)

col1, col2 = st.columns(2)

with col1:
    st.header("📸 Visual Target Capture")
    uploaded_file = st.file_uploader("Upload Leaf Sample Matrix File...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file).convert("RGB")
        st.image(raw_image, caption="Target Asset Sample Matrix Loaded.", use_container_width=True)
        
        is_valid, matched_lbl, integrity_score = verify_agricultural_integrity(raw_image)
        
        if not is_valid:
            st.error(f"⚠️ Validation Layer Warning: Asset rejected. Mismatched profile: ({matched_lbl})")
        else:
            st.success(f"✅ Validation Layer Passed: Confirmed plant foliage. ({matched_lbl} Match Score: {integrity_score:.2f})")
            
            preprocess_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            input_tensor = preprocess_transform(raw_image).unsqueeze(0)
            input_tensor.requires_grad = True
            
            # Extract the final convolutional block layer from the feature extractor
            target_conv_layer = sovereign_engine.features[-1]
            grad_cam = GradientActivationMapping(sovereign_engine, target_conv_layer)
            
            outputs = sovereign_engine(input_tensor)
            probabilities = F.softmax(outputs, dim=1).squeeze().detach().numpy()
            predicted_class_idx = np.argmax(probabilities)
            predicted_class_name = PLANTVILLAGE_CLASSES[predicted_class_idx]
            confidence_level = probabilities[predicted_class_idx]
            
            required_units = float(field_scale * 14.5)
            procurement_cost = float(required_units * 18.5)
            
            cursor = db_conn.cursor()
            cursor.execute("""
                INSERT INTO field_telemetry (timestamp, condition, confidence, hectares, soil_ph, vpd, required_kg, procurement_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                predicted_class_name, float(confidence_level), float(field_scale), float(soil_ph), float(air_vpd), required_units, procurement_cost
            ))
            db_conn.commit()
            
            st.subheader("🔬 Diagnostic Evaluation Matrix")
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.metric(label="Identified Crop Condition", value=predicted_class_name.replace("___", " → "))
            with m_col2:
                st.metric(label="Predictive Confidence Profile", value=f"{confidence_level*100:.2f}%")
            
            heatmap_mask = grad_cam.compute_heatmap(input_tensor, predicted_class_idx)
            cam_overlay_img = generate_cam_overlay(raw_image, heatmap_mask)
            
            st.subheader("🗺️ Explainable AI Heatmap Focus")
            st.image(cam_overlay_img, caption="Grad-CAM Layer Target Focus Breakdown Map.", use_container_width=True)

with col2:
    st.header("📋 Tactical Logistics & Guidance")
    if uploaded_file is not None and is_valid:
        copilot_advice = get_agronomist_advice(predicted_class_name, required_units, procurement_cost, air_vpd)
        
        st.subheader("🤖 EMBEDDED AGRONOMIST CO-PILOT")
        st.info(f"🔬 **Biological Pathogen ID:** {copilot_advice['pathogen']}")
        st.info(f"📋 **Clinical Symptom Profile:** {copilot_advice['symptoms']}")
        st.info(f"💨 **Vector Vectoring Dynamics:** {copilot_advice['transmission']}")
        st.info(f"⚡ **Prescription Mitigation Strategy:** {copilot_advice['strategy']}")
        
        st.subheader("📊 Supply Chain Projections")
        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.metric(label="Total Weight Requirement", value=f"{required_units:.2f} Kg")
        with p_col2:
            st.metric(label="Calculated Procurement Cost", value=f"${procurement_cost:.2f}")
    else:
        st.info("Awaiting structural crop matrix validation signals to compile guidance reports.")

# Data Matrix Logs History Display Panel
st.subheader("⏳ Persistent Field Telemetry Ledger Log History (SQLite)")
df_history = pd.read_sql_query("SELECT * FROM field_telemetry ORDER BY id DESC LIMIT 10", db_conn)
st.dataframe(df_history, use_container_width=True)

