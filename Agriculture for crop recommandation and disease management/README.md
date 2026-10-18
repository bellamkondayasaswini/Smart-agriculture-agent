# 🌾 Smart Agriculture Agent for Crop Recommendation and Crop Disease Advisory

An AI-powered Precision Agriculture Decision Support System developed for a **College AI Mini-Capstone Project** (First-Year B.Tech level).

This web application integrates **Machine Learning classification for crop recommendation**, **plant leaf disease detection and treatment advisory**, and an **interactive rule-based AI agriculture assistant** (*Kisan Mitra*) into a clean, responsive Streamlit dashboard.

---

## 📌 Project Objectives

1. **Intelligent Crop Selection**: Recommend the most suitable agricultural crop from 22 candidates based on soil chemistry (Nitrogen, Phosphorus, Potassium, pH) and climatic factors (Temperature, Humidity, Rainfall) using a trained Machine Learning model.
2. **Crop Disease Identification & Management**: Validate and analyze plant leaf photos, diagnose fungal/bacterial/viral pathogens, and provide structured organic remedies, chemical treatments, and preventive farm sanitation guidelines.
3. **Interactive AI Agriculture Assistant (*Kisan Mitra*)**: Provide an offline, rule-based conversational assistant that explains machine learning decisions, identifies soil nutrient gaps, and formulates integrated 4-phase seasonal farm action plans without paid API keys.
4. **Transparency & Reliability**: Display genuine model evaluation metrics (accuracy, precision, recall, confusion matrix, feature importance), validate all user inputs, and strictly avoid fabricated predictions.

---

## 🚀 System Architecture & Machine Learning Workflow

```
                                  USER INPUTS
                                       │
            ┌──────────────────────────┴──────────────────────────┐
            ▼                                                     ▼
 ┌──────────────────────┐                              ┌──────────────────────┐
 │ Soil & Climate Data  │                              │   Leaf Photograph    │
 │ (N, P, K, pH, Temp,  │                              │ (JPG/PNG/WEBP upload │
 │  Humidity, Rainfall) │                              │  or sample selector) │
 └──────────┬───────────┘                              └──────────┬───────────┘
            │                                                     │
            ▼                                                     ▼
 ┌──────────────────────┐                              ┌──────────────────────┐
 │   Input Validation   │                              │   Image Validation   │
 │ (Agri boundaries &   │                              │ (Dimensions, format, │
 │  NaN/Inf filtering)  │                              │  byte integrity)     │
 └──────────┬───────────┘                              └──────────┬───────────┘
            │                                                     │
            ▼                                                     ▼
 ┌──────────────────────┐                              ┌──────────────────────┐
 │ Random Forest Model  │                              │ Disease Classifier   │
 │ (Trained on 2,200    │                              │ (Color Histograms,   │
 │  genuine samples)    │                              │  HSV, Spatial Grid)  │
 └──────────┬───────────┘                              └──────────┬───────────┘
            │                                                     │
            ▼                                                     ▼
 ┌──────────────────────┐                              ┌──────────────────────┐
 │ Top Recommended Crop │                              │ Disease Diagnosis &  │
 │ + Soil Gap Analysis  │                              │ Full Action Advisory │
 └──────────┬───────────┘                              └──────────┬───────────┘
            │                                                     │
            └──────────────────────────┬──────────────────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │ AI Agriculture Assistant     │
                        │ - Scientific explanation     │
                        │ - Soil nutrient corrections  │
                        │ - 4-Phase Seasonal Farm Plan │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │   Streamlit Web Dashboard    │
                        └──────────────────────────────┘
```

---

## 📁 Project Directory Structure

```
├── app.py                     # Main Streamlit web application (5 dashboard pages)
├── crop_recommender.py        # Crop inference, input validation, and agronomic gap analysis
├── disease_detector.py        # Leaf image validation, model inference, and 38-class advisory
├── agriculture_agent.py       # Offline rule-based AI assistant and farm action planning
├── train_crop_model.py        # Model training pipeline comparing Random Forest, Decision Tree, Naive Bayes
├── train_disease_model.py     # Disease model training script (full PlantVillage + baseline demo)
├── requirements.txt           # Project dependencies
├── README.md                  # Comprehensive setup and project documentation
├── .gitignore                 # Cache and environment exclusion rules
│
├── data/
│   ├── crop_recommendation.csv # Genuine 2,200-row agricultural crop dataset
│   ├── README_DATA.md         # Dataset provenance, attributes, and download instructions
│   └── sample_leaves/         # Built-in demonstration leaf images for testing
│
├── models/
│   ├── crop_recommendation_model.pkl # Trained Random Forest model bundle
│   ├── crop_model_metrics.json       # Unseen test evaluation metrics & confusion matrix
│   ├── plant_disease_model.pkl       # Trained disease classification model bundle
│   └── plant_disease_classes.json    # Supported plant disease classes
│
└── tests/
    ├── __init__.py
    ├── test_crop_recommender.py  # 12 automated unit tests for crop recommendation
    ├── test_disease_detector.py  # 12 automated unit tests for disease detection
    └── test_agriculture_agent.py # 11 automated unit tests for AI assistant
```

---

## 🛠️ Role of Important Python Files

| File | Primary Function |
| :--- | :--- |
| `app.py` | Launches the multi-page Streamlit portal with the agricultural emerald green theme, interactive sliders, demonstration leaf selector, and live assistant chat. |
| `crop_recommender.py` | Contains input validation logic, loads `crop_recommendation_model.pkl`, generates top-3 probabilities, and calculates soil nutrient and rainfall deficiencies. |
| `disease_detector.py` | Handles image file validation (format, size, dimensions), checks model availability, extracts visual features, and queries the 38-class plant pathology knowledge base. |
| `agriculture_agent.py` | Powers *Kisan Mitra*, the rule-based assistant that synthesizes crop results, explains recommendations, and suggests integrated 4-phase farming schedules. |
| `train_crop_model.py` | Trains candidate classifiers on `data/crop_recommendation.csv`, evaluates test metrics (accuracy, precision, recall, F1), and saves model bundles and metrics. |
| `train_disease_model.py` | Trains leaf disease models either on real PlantVillage image folders or generates verified baseline models for demonstration. |

---

## 📊 Feature Status Matrix

| Feature | Status | Details |
| :--- | :---: | :--- |
| **Crop Recommendation** | ✅ **100% Fully Functional** | Trained on 2,200 genuine rows with **99.55% test accuracy**; includes soil gap analysis and 22 crop profiles. |
| **Model Evaluation Dashboard** | ✅ **100% Fully Functional** | Real test accuracy, precision, recall, feature importance chart, and model comparison table. |
| **AI Agriculture Assistant** | ✅ **100% Fully Functional** | Offline rule-based agent; answers questions, explains predictions, and synthesizes 4-phase farm plans. |
| **Plant Disease Knowledge Base** | ✅ **100% Fully Functional** | Searchable database of 38 disease/healthy classes across 14 crops with symptoms, organic, and chemical cures. |
| **Leaf Image Validation** | ✅ **100% Fully Functional** | Verifies JPG/PNG/WEBP formats, checks dimensions (>32px, <8000px), and flags corrupted files. |
| **Disease Model Inference** | ✅ **Operational (Baseline Demo)** | Working feature classifier across 8 core classes; includes sample images for demonstration. |
| **Full 54k PlantVillage Training** | ℹ️ **Requires Optional Download** | Full 1.5 GB dataset download instructions provided below; runs via `python train_disease_model.py`. |

---

## 💻 Installation & Execution on a Windows Laptop

### 1. Open PowerShell
Press `Win + X` and select **Terminal** or **Windows PowerShell**. Navigate to the project directory:
```powershell
cd "c:\Users\DELL\Desktop\Agriculture for crop recommandation and disease management"
```

### 2. Verify or Set Up Python Environment
Ensure Python is installed (Python 3.10 to 3.14):
```powershell
python --version
```
*(If `python` is not recognized, use `py` or the direct path to your Python executable).*

### 3. Install Dependencies
Install all required packages from `requirements.txt`:
```powershell
pip install -r requirements.txt
```

### 4. Train the Machine Learning Models
Train the crop recommendation model on the genuine dataset:
```powershell
python train_crop_model.py
```
*(Expected output: Best Algorithm: Random Forest Classifier with ~99.55% test accuracy).*

To generate or refresh the baseline disease model for image demonstration:
```powershell
python train_disease_model.py --train_baseline
```

### 5. Run Automated Tests
Execute the pytest suite to verify all 35 tests pass:
```powershell
pytest tests/ -v
```

### 6. Launch the Streamlit Web Application
Start the interactive application:
```powershell
streamlit run app.py
```
*(If port 8501 is occupied by another service on your laptop, run: `streamlit run app.py --server.port 8502`).*

Open your web browser and navigate to:
```
http://localhost:8501   (or http://localhost:8502)
```

---

## 🧪 Testing and Reliability Summary

The project includes an automated test suite containing **35 test cases** verified via `pytest`:
- **Input Validation Tests**: Boundary tests for min/max allowable ranges, negative values, excessive values, and non-numeric inputs.
- **Resilience Tests**: Handles missing model files, corrupted image bytes, unsupported formats (`.txt`, `.pdf`), and empty inputs gracefully without application crashes.
- **Integrity Rule Verification**: Tests verify that when disease model weights are absent, the system **never outputs fake predictions** or random confidence numbers.
- **Agent Reasoning Tests**: Verifies intent parsing for soil pH, N-P-K nutrients, irrigation, crop seasons, contextual session explanations, and mandatory safety disclaimers.

---

## 📖 How to Train on the Full PlantVillage Dataset (Optional)

The PlantVillage dataset contains 54,306 images across 38 classes (~1.5 GB). To train on the full dataset:

1. **Download the Dataset**:
   - Via Kaggle CLI:
     ```powershell
     pip install kaggle
     kaggle datasets download -d emmarex/plantdisease
     Expand-Archive plantdisease.zip -DestinationPath data/plantvillage/
     ```
   - Or download manually from [Kaggle: emmarex/plantdisease](https://www.kaggle.com/datasets/emmarex/plantdisease) and extract into `data/plantvillage/`.

2. **Run Full Training**:
   ```powershell
   python train_disease_model.py --data_dir "data/plantvillage/PlantVillage"
   ```
   The script extracts multi-channel color histograms, HSV moments, and spatial grid features, fits a classifier, and updates `models/plant_disease_model.pkl`.

---

## ⚠️ Limitations & Ethical AI Considerations

1. **Decision Support Nature**: Machine learning predictions are advisory aids and do not constitute a laboratory diagnosis. Complex field situations involving multi-pathogen co-infection or abiotic stresses should be verified with local agricultural officers.
2. **Responsible Chemical Usage**: Chemical fungicide recommendations should always adhere to regional label instructions, safety equipment (PPE), and approved waiting periods before harvest.
3. **Connectivity Accessibility**: The decision to keep the AI assistant rule-based ensures that the application runs completely offline on standard farmer or village computer center hardware without ongoing API billing costs.

---

## 📚 Dataset Attribution & References

1. **Crop Recommendation Dataset**: 2,200 curated agricultural records covering 22 crops based on Indian Council of Agricultural Research (ICAR) nutrient thresholds.
2. **PlantVillage Dataset**: Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *"Using Deep Learning for Image-Based Plant Disease Detection"*, Frontiers in Plant Science, 7, 1419.
3. **Scikit-Learn**: Pedregosa et al., *"Scikit-learn: Machine Learning in Python"*, JMLR 12, pp. 2825-2830, 2011.
