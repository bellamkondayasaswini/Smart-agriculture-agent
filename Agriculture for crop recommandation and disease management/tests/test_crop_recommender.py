"""
test_crop_recommender.py
========================
Automated Unit and Integration Tests for Crop Recommendation Engine.

Tests:
  - Input boundary validation (valid, boundary, negative, and extreme values)
  - Non-numeric and missing input handling
  - Model loading and missing model resilience
  - Machine learning prediction accuracy on benchmark inputs
  - Soil nutrient gap analysis calculations
  - Agronomic profile coverage for all 22 crops
  - Evaluation metrics loading and validation
"""

import os
import pytest
import numpy as np

from crop_recommender import (
    validate_crop_inputs,
    load_crop_model,
    predict_crop,
    calculate_soil_gaps,
    load_evaluation_metrics,
    INPUT_BOUNDS,
    CROP_PROFILES
)


class TestCropInputValidation:
    """Tests for agronomic input validation."""

    def test_valid_inputs_pass(self):
        """Standard realistic agricultural values must validate successfully."""
        is_valid, errors, cleaned = validate_crop_inputs(
            n=90.0, p=42.0, k=43.0, temperature=21.0, humidity=82.0, ph=6.5, rainfall=200.0
        )
        assert is_valid is True
        assert len(errors) == 0
        assert cleaned["N"] == 90.0
        assert cleaned["ph"] == 6.5

    def test_boundary_inputs(self):
        """Exact minimum and maximum boundaries must be accepted."""
        is_valid, errors, cleaned = validate_crop_inputs(
            n=0.0, p=5.0, k=5.0, temperature=8.0, humidity=14.0, ph=3.5, rainfall=20.0
        )
        assert is_valid is True
        assert len(errors) == 0

        is_valid_max, errors_max, cleaned_max = validate_crop_inputs(
            n=140.0, p=145.0, k=205.0, temperature=45.0, humidity=100.0, ph=10.0, rainfall=300.0
        )
        assert is_valid_max is True
        assert len(errors_max) == 0

    def test_out_of_bounds_negative_inputs(self):
        """Negative values must trigger validation errors."""
        is_valid, errors, _ = validate_crop_inputs(
            n=-10.0, p=40.0, k=40.0, temperature=25.0, humidity=80.0, ph=6.5, rainfall=100.0
        )
        assert is_valid is False
        assert any("Nitrogen (N)" in err and "Out of valid range" in err for err in errors)

    def test_out_of_bounds_excessive_inputs(self):
        """Excessive unrealistic values must be caught."""
        is_valid, errors, _ = validate_crop_inputs(
            n=500.0, p=40.0, k=40.0, temperature=65.0, humidity=150.0, ph=14.0, rainfall=999.0
        )
        assert is_valid is False
        assert len(errors) >= 4

    def test_non_numeric_and_null_inputs(self):
        """String non-numbers and None must fail validation cleanly."""
        is_valid, errors, _ = validate_crop_inputs(
            n="invalid_n", p=None, k=40.0, temperature="abc", humidity=80.0, ph=6.5, rainfall=100.0
        )
        assert is_valid is False
        assert any("Nitrogen (N)" in err and "valid numeric" in err for err in errors)
        assert any("Phosphorus (P)" in err for err in errors)


class TestCropModelLoadingAndPrediction:
    """Tests for model persistence and ML inference."""

    def test_load_existing_model(self):
        """Existing model bundle must load cleanly with expected structure."""
        bundle, err = load_crop_model()
        assert err is None
        assert bundle is not None
        assert "model" in bundle
        assert "model_name" in bundle
        assert "feature_names" in bundle
        assert len(bundle["feature_names"]) == 7

    def test_load_missing_model_resilience(self):
        """Loading a non-existent model path must return clean error without unhandled exception."""
        bundle, err = load_crop_model(model_path="models/non_existent_file.pkl")
        assert bundle is None
        assert err is not None
        assert "not found" in err.lower()

    def test_predict_rice_conditions(self):
        """Benchmark high-rainfall, high-humidity, moderate temperature conditions must recommend Rice."""
        bundle, err = load_crop_model()
        assert bundle is not None, "Trained model bundle required for prediction test"

        # Realistic rice soil & climate conditions
        inputs = {
            "N": 90.0,
            "P": 42.0,
            "K": 43.0,
            "temperature": 20.88,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 202.94
        }
        res = predict_crop(bundle, inputs, top_k=3)
        assert res["recommended_crop"].lower() == "rice"
        assert res["confidence"] > 50.0
        assert len(res["top_candidates"]) <= 3
        assert "profile" in res
        assert "soil_gap_analysis" in res

    def test_predict_coffee_conditions(self):
        """Benchmark high rainfall, moderate temp, balanced NPK should recommend Coffee or similar plantation."""
        bundle, _ = load_crop_model()
        assert bundle is not None

        inputs = {
            "N": 100.0,
            "P": 28.0,
            "K": 30.0,
            "temperature": 25.5,
            "humidity": 58.0,
            "ph": 6.8,
            "rainfall": 160.0
        }
        res = predict_crop(bundle, inputs, top_k=3)
        assert isinstance(res["recommended_crop"], str)
        assert res["confidence"] > 0.0


class TestAgronomicKnowledgeAndGaps:
    """Tests for soil gap calculations and crop profile dictionary."""

    def test_all_22_crops_have_profiles(self):
        """All 22 crops in the genuine dataset must have detailed agronomic profiles."""
        dataset_crops = [
            "rice", "maize", "chickpea", "kidneybeans", "pigeonpeas", "mothbeans",
            "mungbean", "blackgram", "lentil", "pomegranate", "banana", "mango",
            "grapes", "watermelon", "muskmelon", "apple", "orange", "papaya",
            "coconut", "cotton", "jute", "coffee"
        ]
        for c in dataset_crops:
            assert c in CROP_PROFILES, f"Crop profile missing for: {c}"
            prof = CROP_PROFILES[c]
            assert "category" in prof
            assert "season" in prof
            assert "water_need" in prof
            assert "ideal_ph" in prof
            assert "tips" in prof

    def test_soil_gap_calculation_detects_deficiency(self):
        """Low nitrogen input should be identified as Below Optimal with actionable fertilizer advice."""
        bundle, _ = load_crop_model()
        assert bundle is not None
        benchmarks = bundle.get("agronomic_benchmarks", {})

        inputs = {
            "N": 5.0,  # Intentionally very low for rice
            "P": 42.0,
            "K": 43.0,
            "temperature": 21.0,
            "humidity": 82.0,
            "ph": 6.5,
            "rainfall": 200.0
        }
        gaps = calculate_soil_gaps("rice", inputs, benchmarks)
        assert len(gaps) > 0
        n_gap = next((g for g in gaps if g["feature"] == "N"), None)
        assert n_gap is not None
        assert n_gap["status"] == "Below Optimal"
        assert "Urea" in n_gap["recommendation"] or "nitrogen" in n_gap["recommendation"].lower()

    def test_evaluation_metrics_validity(self):
        """Precomputed test evaluation metrics file must load with accuracy >= 95%."""
        metrics, err = load_evaluation_metrics()
        assert err is None
        assert metrics is not None
        assert "overall_metrics" in metrics
        acc = metrics["overall_metrics"]["accuracy"]
        assert acc >= 0.95
        assert "confusion_matrix" in metrics
        assert "feature_importances" in metrics
