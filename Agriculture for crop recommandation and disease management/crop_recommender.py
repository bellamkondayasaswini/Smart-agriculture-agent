"""
crop_recommender.py
===================
Crop Recommendation Inference, Input Validation, and Agronomic Advisory Engine.

Provides clean functions for:
  - Validating user inputs against realistic agronomic boundaries
  - Loading the trained Scikit-Learn crop recommendation model
  - Predicting the top recommended crop and top-k alternative probabilities
  - Calculating soil nutrient and climate gap analysis against benchmarks
  - Returning detailed agronomic profiles for all 22 crops
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, List, Optional

DEFAULT_MODEL_PATH = os.path.join("models", "crop_recommendation_model.pkl")
DEFAULT_METRICS_PATH = os.path.join("models", "crop_model_metrics.json")

# Permissible input ranges based on ICAR agricultural standards and dataset limits
INPUT_BOUNDS = {
    "N": {"min": 0.0, "max": 140.0, "unit": "mg/kg", "name": "Nitrogen (N)"},
    "P": {"min": 5.0, "max": 145.0, "unit": "mg/kg", "name": "Phosphorus (P)"},
    "K": {"min": 5.0, "max": 205.0, "unit": "mg/kg", "name": "Potassium (K)"},
    "temperature": {"min": 8.0, "max": 45.0, "unit": "°C", "name": "Temperature"},
    "humidity": {"min": 14.0, "max": 100.0, "unit": "%", "name": "Relative Humidity"},
    "ph": {"min": 3.5, "max": 10.0, "unit": "pH scale", "name": "Soil pH"},
    "rainfall": {"min": 20.0, "max": 300.0, "unit": "mm", "name": "Annual/Seasonal Rainfall"}
}

# Agronomic knowledge profiles for all 22 supported crops
CROP_PROFILES = {
    "rice": {
        "botanical_name": "Oryza sativa",
        "category": "Cereal / Food Grain",
        "season": "Kharif (Monsoon)",
        "duration": "100 - 150 days",
        "soil_type": "Clayey, alluvial loam with good water retention",
        "water_need": "Very High (standing water 5-10 cm during early growth)",
        "ideal_ph": "5.5 - 7.0",
        "tips": "Maintain adequate puddling in soil; split nitrogen application at basal, tillering, and panicle initiation stages."
    },
    "maize": {
        "botanical_name": "Zea mays",
        "category": "Cereal / Coarse Grain",
        "season": "Kharif & Rabi",
        "duration": "90 - 120 days",
        "soil_type": "Well-drained deep loamy soil rich in organic matter",
        "water_need": "Moderate (sensitive to waterlogging)",
        "ideal_ph": "5.8 - 7.5",
        "tips": "Ensure furrow drainage to avoid root rot; apply zinc sulfate at sowing if micronutrient deficiency suspected."
    },
    "chickpea": {
        "botanical_name": "Cicer arietinum",
        "category": "Pulse / Legume",
        "season": "Rabi (Winter)",
        "duration": "90 - 110 days",
        "soil_type": "Sandy loam to deep clay loam; well-aerated",
        "water_need": "Low (drought-tolerant cool season pulse)",
        "ideal_ph": "6.0 - 7.8",
        "tips": "Inoculate seeds with Rhizobium culture to fix atmospheric nitrogen; avoid excessive vegetative irrigation."
    },
    "kidneybeans": {
        "botanical_name": "Phaseolus vulgaris",
        "category": "Pulse / Legume",
        "season": "Kharif / Rabi in hills",
        "duration": "85 - 120 days",
        "soil_type": "Light, friable, well-drained loam",
        "water_need": "Moderate",
        "ideal_ph": "5.5 - 6.5",
        "tips": "Very sensitive to salinity and water stagnation; prefers cooler daytime temperatures during pod development."
    },
    "pigeonpeas": {
        "botanical_name": "Cajanus cajan",
        "category": "Pulse / Legume (Arhar/Tur)",
        "season": "Kharif",
        "duration": "140 - 180 days",
        "soil_type": "Deep well-drained loamy to black cotton soil",
        "water_need": "Low to Moderate (deep taproot system)",
        "ideal_ph": "6.5 - 7.5",
        "tips": "Ideal for intercropping with cotton or sorghum; requires minimal nitrogen fertilizer due to high nodulation."
    },
    "mothbeans": {
        "botanical_name": "Vigna aconitifolia",
        "category": "Pulse / Arid Legume",
        "season": "Kharif",
        "duration": "60 - 75 days",
        "soil_type": "Sandy, desert soils with low fertility",
        "water_need": "Very Low (extremely drought-resilient)",
        "ideal_ph": "6.5 - 8.0",
        "tips": "Acts as excellent soil cover crop preventing wind erosion; thrives in arid and semi-arid agro-climatic zones."
    },
    "mungbean": {
        "botanical_name": "Vigna radiata",
        "category": "Pulse / Green Gram",
        "season": "Kharif, Rabi & Summer (Zaid)",
        "duration": "60 - 70 days",
        "soil_type": "Fertile loam to sandy loam with good drainage",
        "water_need": "Low",
        "ideal_ph": "6.2 - 7.2",
        "tips": "Short duration crop ideal for catch-cropping; protect young seedlings from yellow mosaic virus vector whiteflies."
    },
    "blackgram": {
        "botanical_name": "Vigna mungo",
        "category": "Pulse / Urad Bean",
        "season": "Kharif & Rabi",
        "duration": "70 - 85 days",
        "soil_type": "Heavy clay loam to black soils",
        "water_need": "Low to Moderate",
        "ideal_ph": "6.5 - 7.5",
        "tips": "Improves soil fertility via symbiotic nitrogen fixation; apply single superphosphate (SSP) as basal dose."
    },
    "lentil": {
        "botanical_name": "Lens culinaris",
        "category": "Pulse / Legume (Masoor)",
        "season": "Rabi",
        "duration": "110 - 130 days",
        "soil_type": "Loam to clay loam; tolerant to light alkalinity",
        "water_need": "Low",
        "ideal_ph": "6.0 - 8.0",
        "tips": "Requires cold weather during vegetative growth and warm weather at maturity; requires one or two light irrigations."
    },
    "pomegranate": {
        "botanical_name": "Punica granatum",
        "category": "Fruit / Horticultural",
        "season": "Perennial (Ambe, Mrig, Hasta bahar flowering)",
        "duration": "Orchard perennial (harvest 150-180 days after flowering)",
        "soil_type": "Deep well-drained loam or light black soil",
        "water_need": "Moderate (responsive to drip fertigation)",
        "ideal_ph": "6.5 - 7.5",
        "tips": "Prune water shoots regularly; adopt drip irrigation to prevent fruit cracking during maturation."
    },
    "banana": {
        "botanical_name": "Musa acuminata",
        "category": "Fruit / Commercial Plant",
        "season": "Year-round planting",
        "duration": "11 - 13 months",
        "soil_type": "Deep, fertile alluvial or volcanic soil high in potassium",
        "water_need": "Very High (requires frequent uniform irrigation)",
        "ideal_ph": "6.0 - 7.5",
        "tips": "Heavy feeder of Potassium and Nitrogen; provide sturdy propping against high winds during bunch development."
    },
    "mango": {
        "botanical_name": "Mangifera indica",
        "category": "Fruit / Perennial Orchard",
        "season": "Summer harvest (flowering in winter)",
        "duration": "Long-lived perennial tree",
        "soil_type": "Deep alluvial, red or lateritic well-drained soil",
        "water_need": "Moderate (withhold water 2-3 months before flowering)",
        "ideal_ph": "5.5 - 7.5",
        "tips": "Ensure deep drainage (avoid water table within 2 meters); protect inflorescence from powdery mildew and hoppers."
    },
    "grapes": {
        "botanical_name": "Vitis vinifera",
        "category": "Fruit / Commercial Vine",
        "season": "Perennial vine",
        "duration": "Annual harvesting cycle (120-135 days after pruning)",
        "soil_type": "Sandy loam to clay loam with sub-soil gravel",
        "water_need": "Moderate (drip irrigation preferred)",
        "ideal_ph": "6.5 - 8.5",
        "tips": "Adopt Bower or Y-trellis training systems; maintain strict canopy aeration to prevent fungal downy mildew."
    },
    "watermelon": {
        "botanical_name": "Citrullus lanatus",
        "category": "Cucurbit / Fruit",
        "season": "Summer (Zaid)",
        "duration": "80 - 100 days",
        "soil_type": "Warm, deep sandy loam or riverbed soils",
        "water_need": "Moderate (reduce watering near fruit ripening)",
        "ideal_ph": "6.0 - 7.0",
        "tips": "Withhold heavy irrigation during ripening to concentrate brix sugar levels and avoid rind splitting."
    },
    "muskmelon": {
        "botanical_name": "Cucumis melo",
        "category": "Cucurbit / Fruit",
        "season": "Summer (Zaid)",
        "duration": "75 - 90 days",
        "soil_type": "Well-drained sandy loam rich in organic manure",
        "water_need": "Moderate",
        "ideal_ph": "6.0 - 7.0",
        "tips": "Requires hot, dry climate with plenty of sunshine for high sweetness; mulch soil to preserve root warmth."
    },
    "apple": {
        "botanical_name": "Malus domestica",
        "category": "Temperate Fruit",
        "season": "Autumn harvest (Rabi/Winter dormancy)",
        "duration": "Perennial tree crop",
        "soil_type": "Deep, well-aerated loamy soil with high organic matter",
        "water_need": "Moderate (requires 1000-1200 chilling hours <7°C)",
        "ideal_ph": "5.5 - 6.5",
        "tips": "Suitable strictly for temperate hill altitudes (1500m - 2700m); monitor for apple scab after monsoon rains."
    },
    "orange": {
        "botanical_name": "Citrus sinensis",
        "category": "Citrus Fruit",
        "season": "Perennial orchard crop",
        "duration": "Harvest 8 - 9 months after bloom",
        "soil_type": "Well-drained light loam to sandy loam",
        "water_need": "Moderate to High",
        "ideal_ph": "6.0 - 7.5",
        "tips": "Highly intolerant to saline or waterlogged soils; maintain regular zinc and iron micronutrient foliar sprays."
    },
    "papaya": {
        "botanical_name": "Carica papaya",
        "category": "Fruit Crop",
        "season": "Year-round (planted in spring or monsoon)",
        "duration": "9 - 11 months to first harvest",
        "soil_type": "Rich, well-drained sandy or alluvial loam",
        "water_need": "Moderate (strictly no standing water, prone to collar rot)",
        "ideal_ph": "6.0 - 7.0",
        "tips": "Provide raised bed planting to prevent collar rot; rogue out virus-infected plants promptly to arrest vector spread."
    },
    "coconut": {
        "botanical_name": "Cocos nucifera",
        "category": "Plantation / Oil Palm",
        "season": "Perennial coastal crop",
        "duration": "Perennial palm (continuous monthly harvest)",
        "soil_type": "Coastal sandy, red sandy loam or lateritic soil",
        "water_need": "High (uniform coastal humidity and sub-soil moisture)",
        "ideal_ph": "5.2 - 8.0",
        "tips": "Requires abundant sunshine and high humidity; apply muriate of potash (MOP) and magnesium sulfate in ring basins."
    },
    "cotton": {
        "botanical_name": "Gossypium hirsutum",
        "category": "Fiber / Cash Crop",
        "season": "Kharif",
        "duration": "150 - 180 days",
        "soil_type": "Deep black cotton soils (Vertisols) with high clay content",
        "water_need": "Moderate (sensitive to excessive rainfall at boll opening)",
        "ideal_ph": "6.0 - 8.5",
        "tips": "Avoid water stress during flowering and boll formation; install pheromone traps to monitor pink bollworm."
    },
    "jute": {
        "botanical_name": "Corchorus olitorius",
        "category": "Fiber / Commercial",
        "season": "Early Kharif (March - May sowing)",
        "duration": "120 - 150 days",
        "soil_type": "Alluvial deltaic clayey to sandy loam",
        "water_need": "High to Very High (standing water during retting)",
        "ideal_ph": "6.0 - 7.5",
        "tips": "Requires warm, humid climate and abundant clean running or stagnant water for quality fiber retting."
    },
    "coffee": {
        "botanical_name": "Coffea arabica / canephora",
        "category": "Plantation / Beverage",
        "season": "Perennial (harvest Nov - Feb)",
        "duration": "Perennial shade-grown shrub",
        "soil_type": "Deep, fertile, permeable volcanic or humic forest loam",
        "water_need": "High (well-distributed rainfall with 2-3 month dry blossom break)",
        "ideal_ph": "5.5 - 6.5",
        "tips": "Requires two-tier shade trees (silver oak, dadap); prune diseased shoots to restrict coffee berry borer."
    }
}


def load_crop_model(model_path: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Load serialized crop model bundle.
    Returns: (model_bundle, error_message)
    """
    path = model_path or DEFAULT_MODEL_PATH
    if not os.path.exists(path):
        return None, (
            f"Trained model not found at '{path}'. "
            "Please run 'python train_crop_model.py' to generate the model bundle."
        )
    try:
        bundle = joblib.load(path)
        if not isinstance(bundle, dict) or "model" not in bundle:
            return None, "Model file is corrupt or does not match the expected bundle format."
        return bundle, None
    except Exception as e:
        return None, f"Failed to load crop model: {str(e)}"


def load_evaluation_metrics(metrics_path: Optional[str] = None) -> Tuple[Optional[dict], Optional[str]]:
    """
    Load precomputed test evaluation metrics.
    Returns: (metrics_dict, error_message)
    """
    path = metrics_path or DEFAULT_METRICS_PATH
    if not os.path.exists(path):
        return None, f"Metrics file not found at '{path}'. Run train_crop_model.py first."
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data, None
    except Exception as e:
        return None, f"Failed to load metrics: {str(e)}"


def validate_crop_inputs(
    n: Any,
    p: Any,
    k: Any,
    temperature: Any,
    humidity: Any,
    ph: Any,
    rainfall: Any
) -> Tuple[bool, List[str], Dict[str, float]]:
    """
    Validate user inputs against physical and agronomic ranges.
    Returns: (is_valid, error_list, cleaned_values_dict)
    """
    errors = []
    cleaned = {}
    
    raw_inputs = {
        "N": n,
        "P": p,
        "K": k,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    for key, val in raw_inputs.items():
        bounds = INPUT_BOUNDS[key]
        name = bounds["name"]
        unit = bounds["unit"]
        
        # Check type
        try:
            val_float = float(val)
        except (ValueError, TypeError):
            errors.append(f"{name}: Must be a valid numeric number (received '{val}').")
            continue
        
        # Check NaN or Inf
        if np.isnan(val_float) or np.isinf(val_float):
            errors.append(f"{name}: Value cannot be NaN or infinite.")
            continue
            
        # Check Range
        min_val = bounds["min"]
        max_val = bounds["max"]
        if val_float < min_val or val_float > max_val:
            errors.append(
                f"{name}: Out of valid range ({min_val} to {max_val} {unit}). Entered: {val_float} {unit}."
            )
        else:
            cleaned[key] = val_float

    is_valid = len(errors) == 0
    return is_valid, errors, cleaned


def calculate_soil_gaps(crop_name: str, user_inputs: Dict[str, float], benchmarks: Optional[dict] = None) -> List[Dict[str, Any]]:
    """
    Compare user inputs against crop's ideal benchmarks to provide agronomic advice.
    """
    crop_lower = crop_name.lower().strip()
    gaps = []

    if benchmarks and crop_lower in benchmarks:
        crop_bm = benchmarks[crop_lower]
        for feat in ["N", "P", "K", "ph", "rainfall", "temperature", "humidity"]:
            if feat not in user_inputs or feat not in crop_bm:
                continue
            entered = user_inputs[feat]
            ideal_min = crop_bm[feat]["min"]
            ideal_max = crop_bm[feat]["max"]
            ideal_mean = crop_bm[feat]["mean"]
            unit = INPUT_BOUNDS[feat]["unit"]
            feat_name = INPUT_BOUNDS[feat]["name"]

            if entered < ideal_min:
                diff = round(ideal_min - entered, 1)
                status = "Below Optimal"
                if feat == "N":
                    tip = f"Deficient by ~{diff} {unit}. Apply nitrogen-rich fertilizer like Urea (46% N) or well-decomposed farmyard manure."
                elif feat == "P":
                    tip = f"Deficient by ~{diff} {unit}. Apply Single Super Phosphate (SSP) or Di-Ammonium Phosphate (DAP) at sowing."
                elif feat == "K":
                    tip = f"Deficient by ~{diff} {unit}. Apply Muriate of Potash (MOP) to enhance stalk strength and pest tolerance."
                elif feat == "ph":
                    tip = f"Soil is more acidic than optimal. Apply agricultural lime (calcium carbonate) or dolomite to raise pH."
                elif feat == "rainfall":
                    tip = f"Local rainfall is below natural requirement. Supplemental irrigation (drip or sprinkler) is required."
                else:
                    tip = f"Lower than standard requirement for {crop_name.title()}."
            elif entered > ideal_max:
                diff = round(entered - ideal_max, 1)
                status = "Above Optimal"
                if feat == "N":
                    tip = f"Excessive by ~{diff} {unit}. Avoid excess nitrogen to prevent delayed maturity and lodging."
                elif feat == "P":
                    tip = f"Excessive by ~{diff} {unit}. Curtail phosphorus applications to avoid zinc/iron lockup."
                elif feat == "K":
                    tip = f"High by ~{diff} {unit}. Soil has adequate potassium reserves; skip potassium top-dressing."
                elif feat == "ph":
                    tip = f"Soil is slightly alkaline. Incorporate gypsum or organic mulch / peat moss to buffer pH downwards."
                elif feat == "rainfall":
                    tip = f"Higher rainfall than ideal. Ensure raised field beds and proper drainage channels to prevent root rot."
                else:
                    tip = f"Higher than standard requirement for {crop_name.title()}."
            else:
                status = "Optimal"
                tip = f"Well-suited for {crop_name.title()} (within ideal range {ideal_min}-{ideal_max} {unit})."

            gaps.append({
                "feature": feat,
                "name": feat_name,
                "entered": entered,
                "ideal_mean": ideal_mean,
                "ideal_range": f"{ideal_min} - {ideal_max} {unit}",
                "unit": unit,
                "status": status,
                "recommendation": tip
            })
    return gaps


def predict_crop(
    model_bundle: dict,
    inputs: Dict[str, float],
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Run prediction given a valid dictionary of inputs.
    Returns dictionary with recommended crop, probabilities, agronomic profile, and soil gaps.
    """
    model = model_bundle["model"]
    feature_names = model_bundle.get("feature_names", list(INPUT_BOUNDS.keys()))
    benchmarks = model_bundle.get("agronomic_benchmarks", {})

    # Create 2D DataFrame for scikit-learn
    input_df = pd.DataFrame([[inputs[col] for col in feature_names]], columns=feature_names)

    # Predict class & probabilities
    prediction = model.predict(input_df)[0]
    
    top_candidates = []
    confidence = 1.0
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(input_df)[0]
        classes = model.classes_
        
        # Sort classes by probability descending
        sorted_indices = np.argsort(probabilities)[::-1]
        confidence = float(probabilities[sorted_indices[0]])
        
        for idx in sorted_indices[:top_k]:
            top_candidates.append({
                "crop": str(classes[idx]).title(),
                "probability": round(float(probabilities[idx]) * 100, 2)
            })

    # Retrieve profile & soil gaps
    crop_profile = CROP_PROFILES.get(
        str(prediction).lower(),
        {
            "botanical_name": "N/A",
            "category": "Agricultural Crop",
            "season": "Seasonal",
            "duration": "100-140 days",
            "soil_type": "Loam",
            "water_need": "Moderate",
            "ideal_ph": "6.0 - 7.0",
            "tips": "Follow standard regional agricultural extension practices."
        }
    )

    gaps = calculate_soil_gaps(str(prediction), inputs, benchmarks)

    return {
        "recommended_crop": str(prediction).title(),
        "crop_code": str(prediction).lower(),
        "confidence": round(confidence * 100, 2),
        "top_candidates": top_candidates,
        "profile": crop_profile,
        "soil_gap_analysis": gaps,
        "inputs": inputs
    }
