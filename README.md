# 🛰️ AgriVision Sovereign Command Engine FOR GDG HACKATON CHALLENGE

An enterprise-grade, localized Computer Vision (CV) platform designed for precision agriculture diagnostics. The system evaluates crop foliage matrix patterns to identify biological pathogens, run real-time microclimate risk analysis, calculate supply-chain prescription costs, and maintain persistent historical telemetry ledgers without requiring continuous external cloud data pipelines.

---

## 🚀 Core Architectural Infrastructure

* **Neural Core Engine:** Built upon a fine-tuned MobileNetV3-Small deep learning architecture, trained across the PlantVillage dataset to classify 38 unique crop-pathogen states.
* **Explainable Focus Layer:** Implements a localized computer vision intensity focus matrix that highlights active necrotic lesions on the target plant asset.
* **Persistent Telemetry Ledger:** Driven by a native SQLite hardware-bound data layer that records chronological field data logs, predictive confidence vectors, and environmental scales.
* **Generative Agro-Copilot:** Translates raw neural classification indexes into clinical symptom profiles and localized transmission vector calculations.

---

## 🛠️ System Deployment & Installation

Ensure you have Python 3.10+ installed locally on your system architecture.

### 1. Clone the Codebase Matrix
```bash
git clone https://github.com/ekrama040-byte/Hackaton_challenge
cd Hackaton_challenge2
```

### 2. Provision Dependencies
Install the necessary computational frameworks and user interface libraries:
```bash
pip install torch torchvision streamlit pandas numpy PIL-compat
```

### 3. Embed Trained Parameter State Weights
Due to remote git size constraints, the binary model checkpoint weights file (`plant_disease_model.pth`) must be placed in the local system directory manually before starting up:
* Create a folder named `models/` in the root directory.
* Drop your compiled `plant_disease_model.pth` file inside the `models/` directory.

---

## 🖥️ Launching the Command Interface

Execute the execution pipeline command inside your terminal environment to start up the web dashboard server interface:

```bash
streamlit run app.py
```

Once executed, navigate your local browser stream terminal to:
👉 **`http://localhost:8501`**

---

## 📊 Database Schema Topology (SQLite)

Field parameters are recorded sequentially inside the `field_telemetry` matrix table configuration:


| Column Field Attribute | Operational Telemetry Data Type | Metrics Description |
| :--- | :--- | :--- |
| `id` | INTEGER PRIMARY KEY | Chronological hardware logging entry token index. |
| `timestamp` | TEXT | Standard ISO date-time string log markers. |
| `condition` | TEXT | Target crop class name + pathogen profile tag. |
| `confidence` | REAL | Model probability percentage distribution scale. |
| `hectares` | REAL | Field spatial matrix tracking calibration size. |
| `soil_ph` | REAL | Active ground chemical monitoring pH level value. |
| `vpd` | REAL | Microclimate Vapor Pressure Deficit calculation. |
| `required_kg` | REAL | Total required chemical mitigation volume unit mass. |
| `procurement_cost` | REAL | Supply-line financial allocation projection cost. |

---
🏁 *Developed as a high-performance prototype for the GDG Hackathon Challenge Engine Panel Presentation Evaluation.*
