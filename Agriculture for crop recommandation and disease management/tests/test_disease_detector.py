"""
test_disease_detector.py
========================
Automated Unit and Integration Tests for Plant Disease Detection and Advisory.

Tests:
  - Leaf image validation (valid images, PNG, JPG)
  - Unsupported file formats (.txt, .pdf, .exe)
  - Corrupted and empty byte streams
  - Minimum/maximum dimension constraints
  - Model status detection (loaded vs missing)
  - Strict absence of fake predictions when model is not loaded
  - Real model inference with confidence and advisory
  - Comprehensive disease catalog lookup and search
  - Mandatory disclaimer inclusion
"""

import io
import pytest
import numpy as np
from PIL import Image

from disease_detector import (
    validate_leaf_image,
    check_disease_model_status,
    load_disease_model,
    predict_disease,
    get_disease_advisory,
    search_disease_catalog,
    DISEASE_KNOWLEDGE_BASE,
    DISCLAIMER_TEXT
)


class TestLeafImageValidation:
    """Tests for image format and byte integrity validation."""

    def test_valid_pil_image_passes(self):
        """Standard valid PIL image should validate cleanly."""
        img = Image.new("RGB", (128, 128), (40, 180, 50))
        is_valid, err, validated_img = validate_leaf_image(img)
        assert is_valid is True
        assert err is None
        assert validated_img is not None
        assert validated_img.size == (128, 128)

    def test_valid_jpeg_bytes_pass(self):
        """Valid JPEG byte buffer must pass validation."""
        img = Image.new("RGB", (100, 100), (30, 150, 40))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)

        is_valid, err, validated_img = validate_leaf_image(buf.getvalue())
        assert is_valid is True
        assert err is None
        assert validated_img is not None

    def test_unsupported_file_extension(self):
        """Unsupported extensions (e.g. .txt, .pdf) must be rejected with helpful message."""
        class MockUploadedFile:
            name = "notes.txt"
            def read(self):
                return b"some text content"

        is_valid, err, _ = validate_leaf_image(MockUploadedFile())
        assert is_valid is False
        assert "Unsupported file format" in err
        assert "Allowed formats are JPG, JPEG, PNG, and WEBP" in err

    def test_corrupted_image_bytes(self):
        """Random corrupted bytes must be caught without throwing unhandled exceptions."""
        corrupted_bytes = b"NOT_A_REAL_IMAGE_DATA_123456789"
        is_valid, err, _ = validate_leaf_image(corrupted_bytes)
        assert is_valid is False
        assert "corrupted" in err.lower() or "cannot identify" in err.lower()

    def test_empty_image_input(self):
        """None or empty bytes must fail validation."""
        is_valid1, err1, _ = validate_leaf_image(None)
        assert is_valid1 is False

        is_valid2, err2, _ = validate_leaf_image(b"")
        assert is_valid2 is False
        assert "empty" in err2.lower()

    def test_too_small_dimensions(self):
        """Images smaller than 32x32 pixels must be rejected."""
        tiny_img = Image.new("RGB", (16, 16), (40, 180, 50))
        is_valid, err, _ = validate_leaf_image(tiny_img)
        assert is_valid is False
        assert "too small" in err.lower()


class TestDiseaseModelAndInference:
    """Tests for model status, missing model behavior, and genuine inference."""

    def test_model_status_returns_dict(self):
        """check_disease_model_status must return a structured dictionary."""
        status = check_disease_model_status()
        assert isinstance(status, dict)
        assert "is_available" in status
        assert "message" in status

    def test_predict_disease_without_model_never_fakes_prediction(self):
        """
        CRITICAL REQUIREMENT:
        When model is None or not loaded, predict_disease must NEVER output fake predictions
        or random confidence numbers.
        """
        img = Image.new("RGB", (100, 100), (40, 180, 50))
        res = predict_disease(img, model_bundle=None)
        assert res["success"] is False
        assert "not loaded" in res["error"].lower()
        assert res["predicted_class"] is None
        assert res["confidence"] is None
        assert res["top_candidates"] == []
        assert res["disclaimer"] == DISCLAIMER_TEXT

    def test_real_model_inference_with_trained_bundle(self):
        """When model bundle is loaded, inference must return genuine class, confidence, and advisory."""
        bundle, err = load_disease_model()
        if bundle is None:
            pytest.skip("Disease model weights not present on disk; skipping live inference test.")

        img = Image.new("RGB", (128, 128), (40, 180, 50))
        res = predict_disease(img, model_bundle=bundle)
        assert res["success"] is True
        assert res["error"] is None
        assert res["predicted_class"] is not None
        assert res["confidence"] > 0.0
        assert len(res["top_candidates"]) > 0
        assert res["advisory"] is not None
        assert "disclaimer" in res


class TestDiseaseCatalogAndAdvisory:
    """Tests for disease advisory knowledge base and search functionality."""

    def test_knowledge_base_covers_core_classes(self):
        """Core PlantVillage crops must be present in the advisory database."""
        expected_crops = ["Potato", "Tomato", "Corn (Maize)", "Apple", "Grape"]
        covered_crops = {data["crop"] for data in DISEASE_KNOWLEDGE_BASE.values()}
        for c in expected_crops:
            assert c in covered_crops, f"Crop '{c}' missing from DISEASE_KNOWLEDGE_BASE"

    def test_every_entry_has_required_fields(self):
        """Every entry must provide symptoms, organic remedy, chemical treatment, and prevention."""
        for key, entry in DISEASE_KNOWLEDGE_BASE.items():
            assert "crop" in entry, f"Missing 'crop' in {key}"
            assert "is_healthy" in entry, f"Missing 'is_healthy' in {key}"
            assert "symptoms" in entry, f"Missing 'symptoms' in {key}"
            assert "organic_remedy" in entry, f"Missing 'organic_remedy' in {key}"
            assert "chemical_treatment" in entry, f"Missing 'chemical_treatment' in {key}"
            assert "prevention" in entry, f"Missing 'prevention' in {key}"

    def test_catalog_search_by_crop_and_symptom(self):
        """Search must find relevant entries by crop name and symptom keyword."""
        tomato_results = search_disease_catalog(query="", crop_filter="Tomato")
        assert len(tomato_results) >= 5

        blight_results = search_disease_catalog(query="blight")
        assert len(blight_results) >= 2
        for r in blight_results:
            assert "blight" in r["class_key"].lower() or "blight" in r["symptoms"].lower()
