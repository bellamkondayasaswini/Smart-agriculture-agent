"""
test_agriculture_agent.py
=========================
Automated Unit and Integration Tests for AI Agriculture Assistant Engine.

Tests:
  - Empty query handling and suggested next actions
  - Greeting and help intent
  - Soil pH advisory and amendment recommendations
  - N-P-K nutrient guidance (Urea, DAP, MOP)
  - Irrigation and water conservation advisory
  - Cropping seasons (Kharif, Rabi, Zaid)
  - Contextual crop recommendation reasoning (with and without session context)
  - Contextual disease diagnosis explanation (healthy vs infected)
  - Integrated 4-phase seasonal farm action plan
  - Agricultural safety disclaimer inclusion across responses
"""

import pytest
from agriculture_agent import AgricultureAgent, AGENT_DISCLAIMER


@pytest.fixture
def agent():
    return AgricultureAgent()


class TestAgricultureAgentIntents:
    """Tests for individual intent handlers."""

    def test_empty_query(self, agent):
        """Empty or whitespace query should return friendly prompt and suggestions."""
        res = agent.answer_query("")
        assert res["intent"] == "EMPTY_QUERY"
        assert len(res["suggested_next_steps"]) > 0

    def test_greeting_intent(self, agent):
        """Greetings should return welcoming introduction and capabilities overview."""
        res = agent.answer_query("Hello! Who are you?")
        assert res["intent"] == "GREETING"
        assert "Kisan Mitra" in res["response_text"] or "Smart Agriculture Agent" in res["response_text"]
        assert len(res["suggested_next_steps"]) > 0

    def test_soil_ph_query(self, agent):
        """Queries about soil pH must explain lime, gypsum, and nutrient lockup."""
        res = agent.answer_query("How do I manage high alkaline soil pH?")
        assert res["intent"] == "SOIL_PH_MANAGEMENT"
        assert "Gypsum" in res["response_text"]
        assert "Lime" in res["response_text"] or "Limestone" in res["response_text"]

    def test_nutrient_npk_query(self, agent):
        """Queries on nitrogen/potassium/phosphorus must explain deficiency and fertilizers."""
        res = agent.answer_query("What does potassium do and how to fix deficiency?")
        assert res["intent"] == "SOIL_NUTRIENTS_NPK"
        assert "Potassium (K)" in res["response_text"]
        assert "MOP" in res["response_text"] or "Muriate of Potash" in res["response_text"]

    def test_irrigation_query(self, agent):
        """Irrigation queries should explain drip, sprinkler, and waterlogging."""
        res = agent.answer_query("What is the best irrigation method for saving water?")
        assert res["intent"] == "IRRIGATION_WATER"
        assert "Drip" in res["response_text"]

    def test_seasons_query(self, agent):
        """Season queries should explain Kharif, Rabi, and Zaid cycles."""
        res = agent.answer_query("Tell me about Kharif and Rabi cropping seasons")
        assert res["intent"] == "AGRONOMIC_SEASONS"
        assert "Kharif" in res["response_text"]
        assert "Rabi" in res["response_text"]


class TestContextualSessionSynthesis:
    """Tests for synthesizing session state results into natural language explanations."""

    def test_crop_explanation_without_session_context(self, agent):
        """Asking why a crop was chosen without prior recommendation prompts user to run recommender."""
        res = agent.answer_query("Why was this crop recommended?", session_context={})
        assert res["intent"] == "CROP_EXPLANATION_NO_CONTEXT"
        assert "Crop Recommendation" in res["response_text"]

    def test_crop_explanation_with_session_context(self, agent):
        """When session context contains a recommendation, agent must explain the specific crop."""
        mock_context = {
            "last_crop_recommendation": {
                "recommended_crop": "Rice",
                "confidence": 94.2,
                "inputs": {
                    "N": 90.0, "P": 42.0, "K": 43.0,
                    "temperature": 21.0, "humidity": 82.0, "ph": 6.5, "rainfall": 202.0
                },
                "profile": {
                    "category": "Cereal / Food Grain",
                    "season": "Kharif (Monsoon)",
                    "duration": "100 - 150 days",
                    "soil_type": "Clayey alluvial",
                    "water_need": "Very High",
                    "ideal_ph": "5.5 - 7.0",
                    "tips": "Maintain puddling."
                },
                "soil_gap_analysis": [
                    {
                        "feature": "N",
                        "name": "Nitrogen (N)",
                        "status": "Optimal",
                        "recommendation": "Well-suited for Rice."
                    }
                ]
            }
        }
        res = agent.answer_query("Why was this crop chosen for my farm?", session_context=mock_context)
        assert res["intent"] == "CROP_EXPLANATION_SUCCESS"
        assert "Rice" in res["response_text"]
        assert "94.2%" in res["response_text"]
        assert "Kharif" in res["response_text"]

    def test_disease_explanation_with_infected_context(self, agent):
        """When session context has an active disease detection, agent explains symptoms and treatments."""
        mock_context = {
            "last_disease_detection": {
                "success": True,
                "predicted_class": "Potato___Early_blight",
                "display_name": "Potato — Early Blight (Alternaria solani)",
                "confidence": 88.5,
                "is_healthy": False,
                "advisory": {
                    "pathogen": "Fungus (Alternaria solani)",
                    "symptoms": "Concentric rings target spot.",
                    "organic_remedy": "Neem oil spray (0.5%).",
                    "chemical_treatment": "Mancozeb 75% WP (2 g/L).",
                    "prevention": "3-year crop rotation."
                }
            }
        }
        res = agent.answer_query("How should I treat this plant disease?", session_context=mock_context)
        assert res["intent"] == "DISEASE_EXPLANATION_SUCCESS"
        assert "Early Blight" in res["response_text"]
        assert "Neem oil" in res["response_text"]
        assert "Mancozeb" in res["response_text"]

    def test_integrated_seasonal_farm_action_plan(self, agent):
        """Asking for an action plan synthesizes both crop and disease context into 4 phases."""
        mock_context = {
            "last_crop_recommendation": {
                "recommended_crop": "Tomato",
                "confidence": 89.0,
                "inputs": {"N": 60, "P": 50, "K": 50, "ph": 6.5, "rainfall": 80},
                "profile": {"season": "Kharif/Rabi"},
                "soil_gap_analysis": []
            },
            "last_disease_detection": {
                "success": True,
                "display_name": "Tomato — Early Blight",
                "is_healthy": False,
                "advisory": {
                    "organic_remedy": "Neem oil spray.",
                    "chemical_treatment": "Chlorothalonil."
                }
            }
        }
        res = agent.answer_query("Give me an integrated seasonal action plan", session_context=mock_context)
        assert res["intent"] == "INTEGRATED_ACTION_PLAN"
        assert "Phase 1" in res["response_text"]
        assert "Phase 2" in res["response_text"]
        assert "Phase 3" in res["response_text"]
        assert "Phase 4" in res["response_text"]
        assert "Tomato" in res["response_text"]

    def test_disclaimer_presence(self, agent):
        """All agricultural advisory responses must conclude with a responsible scientific disclaimer."""
        queries = [
            "Why was this crop recommended?",
            "How to treat early blight?",
            "What is the best pH for cotton?",
            "Give me a seasonal farm plan"
        ]
        for q in queries:
            res = agent.answer_query(q)
            assert "Krishi Vigyan Kendra" in res["response_text"] or "educational" in res["response_text"].lower()
