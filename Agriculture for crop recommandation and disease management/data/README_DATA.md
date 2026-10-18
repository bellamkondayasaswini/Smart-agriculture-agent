# Dataset Information

This directory contains the datasets used by the **Smart Agriculture Agent for Crop Recommendation and Crop Disease Advisory**.

## 1. Crop Recommendation Dataset (`crop_recommendation.csv`)

- **Status**: Included and verified (`data/crop_recommendation.csv`).
- **Rows**: 2,200 rows of genuine agricultural data.
- **Classes**: 22 crops (100 samples per crop):
  `rice, maize, chickpea, kidneybeans, pigeonpeas, mothbeans, mungbean, blackgram, lentil, pomegranate, banana, mango, grapes, watermelon, muskmelon, apple, orange, papaya, coconut, cotton, jute, coffee`
- **Features**:
  1. `N` — Nitrogen content ratio in soil (0 to 140 mg/kg)
  2. `P` — Phosphorus content ratio in soil (5 to 145 mg/kg)
  3. `K` — Potassium content ratio in soil (5 to 205 mg/kg)
  4. `temperature` — Temperature in Celsius (8.8°C to 43.7°C)
  5. `humidity` — Relative humidity in % (14.2% to 99.9%)
  6. `ph` — Soil pH value (3.5 to 9.9)
  7. `rainfall` — Rainfall in mm (20.2 mm to 298.6 mm)
  8. `label` — Target recommended crop name

- **Source / Citation**:
  Publicly available agricultural research dataset, widely cited and benchmarked across agricultural AI literature (e.g., Kaggle Crop Recommendation Dataset / Indian Agriculture ICAR benchmark parameters).

---

## 2. Plant Disease Image Dataset (PlantVillage)

- **Dataset Name**: PlantVillage Dataset (Mohanty et al., Penn State University)
- **Total Images**: 54,306 images across 38 crop disease & healthy classes.
- **Supported Crops**: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper Bell, Potato, Raspberry, Soybean, Squash, Strawberry, Tomato.
- **How to Download the Full Dataset**:
  The full dataset is approximately 1.5 GB. To obtain it for full-scale training:
  
  **Option A: Using Kaggle CLI**
  ```bash
  pip install kaggle
  kaggle datasets download -d emmarex/plantdisease
  unzip plantdisease.zip -d data/plantvillage/
  ```

  **Option B: Direct Browser Download**
  - Download from Mendeley Data: [PlantVillage Dataset on Mendeley](https://data.mendeley.com/datasets/tywbtsjrjv/1)
  - Or download from Kaggle: [PlantVillage on Kaggle](https://www.kaggle.com/datasets/emmarex/plantdisease)
  - Extract the folder into `data/plantvillage/`.

- **Training**:
  Once downloaded, run:
  ```bash
  python train_disease_model.py --data_dir data/plantvillage/
  ```
