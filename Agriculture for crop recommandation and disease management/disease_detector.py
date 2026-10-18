"""
disease_detector.py
===================
Plant Disease Image Validation, Classification Inference, and Advisory Engine.

Supports 38 disease & healthy classes across 14 agricultural crops from the PlantVillage taxonomy:
  - Tomato, Potato, Apple, Corn (Maize), Grape, Pepper Bell, Peach, Cherry,
    Strawberry, Orange, Blueberry, Raspberry, Soybean, Squash.

Key Principles:
  1. Strict Image Validation: Verifies file formats (JPG/PNG/WEBP), dimensions, and byte integrity.
  2. Transparent Model Management: Checks whether genuine trained weights exist.
  3. No Fake Predictions: NEVER fabricates or guesses disease classes when weights are absent.
  4. Comprehensive Advisory: Detailed pathogen, symptoms, organic, chemical, and preventive guidance.
  5. Scientific Disclaimer: Clearly states results are for educational advisory and not a certified lab diagnosis.
"""

import os
import io
import json
import joblib
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List

MODEL_DIR = "models"
DEFAULT_PKL_PATH = os.path.join(MODEL_DIR, "plant_disease_model.pkl")
DEFAULT_CLASSES_PATH = os.path.join(MODEL_DIR, "plant_disease_classes.json")

# Mandatory disclaimer required for agricultural AI systems
DISCLAIMER_TEXT = (
    "Disclaimer: This AI disease detection and advisory system is designed for educational "
    "and preliminary guidance purposes. Environmental factors, nutrient deficiencies, and multiple "
    "concurrent pathogens can cause overlapping visual symptoms. For definitive confirmation and "
    "regulated chemical pesticide application, consult your local Agricultural Extension Officer, "
    "Krishi Vigyan Kendra (KVK), or an accredited plant pathology laboratory."
)

# Supported image mime types and extensions
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGE_SIZE_MB = 15
MIN_IMAGE_DIMENSION = 32
MAX_IMAGE_DIMENSION = 8000

# Comprehensive Agricultural Disease Advisory Database (PlantVillage Taxonomy)
DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    # ---------------- POTATO ----------------
    "Potato___Early_blight": {
        "crop": "Potato",
        "display_name": "Potato — Early Blight (Alternaria solani)",
        "is_healthy": False,
        "pathogen": "Fungus (Alternaria solani)",
        "symptoms": "Dark brown to black necrotic spots with characteristic target-board concentric rings, usually appearing first on older lower leaves.",
        "organic_remedy": "Foliar spray of cold-pressed Neem oil (0.5%) or Trichoderma harzianum bio-fungicide. Prune and safely burn heavily infected lower leaves.",
        "chemical_treatment": "Mancozeb 75% WP (2 g/L) or Chlorothalonil 75% WP (2 g/L) at initial onset. Rotate with Azoxystrobin to prevent resistance.",
        "prevention": "Follow 3-year crop rotation with non-solanaceous crops. Use certified disease-free seed tubers and avoid overhead sprinkler irrigation."
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "display_name": "Potato — Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "pathogen": "Oomycete / Water Mold (Phytophthora infestans)",
        "symptoms": "Water-soaked, dark lesions on leaf tips and margins; white velvety mildew appears on undersides during cool, humid or foggy mornings.",
        "organic_remedy": "Apply copper hydroxide / Bordeaux mixture (1%) as a preventive prophylactic spray. Remove and destroy infected vines before tuber harvest.",
        "chemical_treatment": "Metalaxyl 8% + Mancozeb 64% WP (Ridomil MZ, 2.5 g/L) or Cymoxanil + Mancozeb. Immediate spray is essential upon first sign.",
        "prevention": "Plant resistant cultivars (e.g., Kufri Girdhari). Ensure raised ridges to prevent tuber infection and avoid planting in low-lying fog-prone hollows."
    },
    "Potato___healthy": {
        "crop": "Potato",
        "display_name": "Potato — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Vibrant uniform green foliage, upright turgid stems, and absence of necrotic spotting or wilting.",
        "organic_remedy": "Continue routine application of well-composted farmyard manure and balanced N-P-K nutrition.",
        "chemical_treatment": "No chemical treatment required.",
        "prevention": "Scout field weekly for aphid vectors and maintain balanced potassium nutrition for natural disease resistance."
    },

    # ---------------- TOMATO ----------------
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "display_name": "Tomato — Early Blight (Alternaria solani)",
        "is_healthy": False,
        "pathogen": "Fungus (Alternaria solani)",
        "symptoms": "Dark brown circular spots with concentric rings, yellow chlorotic halo around lesions; leaf defoliation proceeds upward from bottom of canopy.",
        "organic_remedy": "Spray bio-control agent Bacillus subtilis or copper octanoate. Mulch soil heavily to prevent rain-splash of fungal spores.",
        "chemical_treatment": "Chlorothalonil (2 g/L) or Mancozeb (2.5 g/L) applied at 7-10 day intervals during humid weather.",
        "prevention": "Stake plants to keep leaves off soil. Adopt drip irrigation rather than overhead sprinklers. Space plants at least 45-60 cm apart."
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "display_name": "Tomato — Late Blight (Phytophthora infestans)",
        "is_healthy": False,
        "pathogen": "Oomycete (Phytophthora infestans)",
        "symptoms": "Rapidly enlarging pale green to brown water-soaked blotches on leaves and stems; white mold on leaf undersides under high humidity.",
        "organic_remedy": "Copper oxychloride (3 g/L) applied proactively prior to sustained rainy weather. Uproot and burn severely blighted plants.",
        "chemical_treatment": "Dimethomorph + Mancozeb (Acrobat MZ) or Metalaxyl-M. Rotate chemical classes to forestall resistance.",
        "prevention": "Avoid planting tomatoes adjacent to potato fields. Ensure superior air circulation in polyhouses and open fields."
    },
    "Tomato___Bacterial_spot": {
        "crop": "Tomato",
        "display_name": "Tomato — Bacterial Spot (Xanthomonas spp.)",
        "is_healthy": False,
        "pathogen": "Bacterium (Xanthomonas campestris pv. vesicatoria)",
        "symptoms": "Small (2-3 mm) dark brown, water-soaked, angular spots on leaves and scabby raised spots on green fruits.",
        "organic_remedy": "Fixed copper formulations combined with biological sprays. Disinfect pruning shears in 70% alcohol between plants.",
        "chemical_treatment": "Copper Hydroxide + Streptocycline (100 ppm / 1g per 10 L water) applied at early vegetative stages.",
        "prevention": "Use hot-water treated or certified disease-free seeds. Never work in wet fields to prevent mechanical transmission."
    },
    "Tomato___Leaf_Mold": {
        "crop": "Tomato",
        "display_name": "Tomato — Leaf Mold (Passalora fulva)",
        "is_healthy": False,
        "pathogen": "Fungus (Passalora fulva)",
        "symptoms": "Pale greenish-yellow patches on upper leaf surface, corresponding to olive-green to brown velvety mold on lower surface.",
        "organic_remedy": "Bio-fungicides like Trichoderma or sulfur dust. Improve greenhouse venting to lower relative humidity below 80%.",
        "chemical_treatment": "Difenoconazole 25% EC (0.5 ml/L) or Azoxystrobin.",
        "prevention": "Prune excess suckers to facilitate cross-ventilation. Select resistant greenhouse hybrids."
    },
    "Tomato___Septoria_leaf_spot": {
        "crop": "Tomato",
        "display_name": "Tomato — Septoria Leaf Spot (Septoria lycopersici)",
        "is_healthy": False,
        "pathogen": "Fungus (Septoria lycopersici)",
        "symptoms": "Numerous small circular spots with grayish-white centers and dark brown margins, speckled with tiny black pycnidia dots.",
        "organic_remedy": "Strip infected lower leaves promptly and burn. Apply liquid copper or sulfur sprays.",
        "chemical_treatment": "Mancozeb or Chlorothalonil every 7 to 10 days until weather dries.",
        "prevention": "Rotate crops away from nightshades for 2 years. Destroy all post-harvest solanaceous residue."
    },
    "Tomato___Spider_mites_Two-spotted_spider_mite": {
        "crop": "Tomato",
        "display_name": "Tomato — Two-Spotted Spider Mite (Tetranychus urticae)",
        "is_healthy": False,
        "pathogen": "Arachnid Pest (Tetranychus urticae)",
        "symptoms": "Stippled yellow speckling on upper leaf surface, fine silken webbing on leaf undersides, foliage bronzing in hot dry weather.",
        "organic_remedy": "Release predatory mites (Phytoseiulus persimilis) or spray insecticidal potassium soap / Neem oil (1%).",
        "chemical_treatment": "Propargite 57% EC (2 ml/L) or Spiromesifen 22.9% SC (1 ml/L). Ensure thorough coverage on leaf undersides.",
        "prevention": "Avoid water stress in crops. Wet pathways in greenhouses to increase ambient humidity and deter mite multiplication."
    },
    "Tomato___Target_Spot": {
        "crop": "Tomato",
        "display_name": "Tomato — Target Spot (Corynespora cassiicola)",
        "is_healthy": False,
        "pathogen": "Fungus (Corynespora cassiicola)",
        "symptoms": "Pinpoint brown spots that expand into circular lesions with light brown centers and dark concentric rings.",
        "organic_remedy": "Copper soap fungicide spray. Remove diseased plant residue at end of season.",
        "chemical_treatment": "Azoxystrobin or Pyraclostrobin rotated with Chlorothalonil.",
        "prevention": "Avoid excessive nitrogen fertilization which produces excessively dense, humid canopies."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "crop": "Tomato",
        "display_name": "Tomato — Yellow Leaf Curl Virus (TYLCV)",
        "is_healthy": False,
        "pathogen": "Begomovirus (transmitted by Whitefly Bemisia tabaci)",
        "symptoms": "Marked upward curling and cupping of leaflets, yellowing (chlorosis) along margins, plant stunting, and flower abortion.",
        "organic_remedy": "Install bright yellow sticky traps (15-20 per acre) to trap whiteflies. Spray Neem oil (10,000 ppm, 2 ml/L).",
        "chemical_treatment": "Imidacloprid 17.8% SL (0.5 ml/L) or Thiamethoxam 25% WG (0.3 g/L) to control insect vector.",
        "prevention": "Use 40-50 mesh insect-proof netting in nurseries. Rogue out virus-infected seedlings immediately."
    },
    "Tomato___Tomato_mosaic_virus": {
        "crop": "Tomato",
        "display_name": "Tomato — Mosaic Virus (ToMV)",
        "is_healthy": False,
        "pathogen": "Tobamovirus (Mechanically transmitted)",
        "symptoms": "Mottling with alternating light and dark green areas on leaves, blistering, and fern-like strapping of foliage.",
        "organic_remedy": "No direct curative cure exists for plant viruses. Wash hands in skimmed milk or trisodium phosphate before handling plants.",
        "chemical_treatment": "No chemical pesticide cures virus once inside tissue; eradicate vectors and infected plants.",
        "prevention": "Strictly prohibit tobacco smoking near greenhouses. Use certified virus-free seed."
    },
    "Tomato___healthy": {
        "crop": "Tomato",
        "display_name": "Tomato — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Lush dark green foliage, well-developed serrated leaves without chlorosis, necrotic spotting, or curling.",
        "organic_remedy": "Support balanced growth with vermicompost tea and foliar seaweed extract.",
        "chemical_treatment": "No chemical intervention needed.",
        "prevention": "Maintain regular watering schedule and support with trellising to prevent soil contact."
    },

    # ---------------- CORN (MAIZE) ----------------
    "Corn_(maize)___Cercospora_leaf_spot_Gray_leaf_spot": {
        "crop": "Corn (Maize)",
        "display_name": "Corn — Gray Leaf Spot (Cercospora zeae-maydis)",
        "is_healthy": False,
        "pathogen": "Fungus (Cercospora zeae-maydis)",
        "symptoms": "Small, tan to rectangular lesions delimited by leaf veins; lesions expand parallel to veins into long gray-brown streaks.",
        "organic_remedy": "Incorporate crop residues deeply into soil using chisel plowing to accelerate residue breakdown.",
        "chemical_treatment": "Azoxystrobin + Difenoconazole or Pyraclostrobin applied at tasseling.",
        "prevention": "Plant resistant corn hybrids; practice 2-year rotation with non-host crops like soybean or cotton."
    },
    "Corn_(maize)___Common_rust_": {
        "crop": "Corn (Maize)",
        "display_name": "Corn — Common Rust (Puccinia sorghi)",
        "is_healthy": False,
        "pathogen": "Fungus (Puccinia sorghi)",
        "symptoms": "Oval to elongate reddish-brown cinnamon pustules (uredinia) scattered across both upper and lower leaf surfaces.",
        "organic_remedy": "Apply sulfur-based organic dusts early in the season if rust appears on seedlings.",
        "chemical_treatment": "Mancozeb 75% WP (2 g/L) or Tebuconazole 25.9% EC (1 ml/L).",
        "prevention": "Select rust-resistant hybrids; sow early in the season to evade peak airborne spore flights."
    },
    "Corn_(maize)___Northern_Leaf_Blight": {
        "crop": "Corn (Maize)",
        "display_name": "Corn — Northern Leaf Blight (Exserohilum turcicum)",
        "is_healthy": False,
        "pathogen": "Fungus (Exserohilum turcicum)",
        "symptoms": "Large, cigar-shaped or elliptical grayish-green to tan lesions (2.5 to 15 cm long) on leaves.",
        "organic_remedy": "Burn or compost infected stover; rotate fields with broadleaf crops.",
        "chemical_treatment": "Propiconazole 25% EC (1 ml/L) or Mancozeb at early whorl or tasseling stage.",
        "prevention": "Plant hybrids with high quantitative polygenic resistance; avoid no-till continuous monoculture."
    },
    "Corn_(maize)___healthy": {
        "crop": "Corn (Maize)",
        "display_name": "Corn — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Broad, vibrant green leaves with smooth margins and clear veins, free of pustules or necrotic lesions.",
        "organic_remedy": "Apply balanced basal doses of nitrogen and zinc.",
        "chemical_treatment": "No chemical treatment required.",
        "prevention": "Ensure weed-free field during the critical first 30 days after germination."
    },

    # ---------------- APPLE ----------------
    "Apple___Apple_scab": {
        "crop": "Apple",
        "display_name": "Apple — Apple Scab (Venturia inaequalis)",
        "is_healthy": False,
        "pathogen": "Fungus (Venturia inaequalis)",
        "symptoms": "Olive-green to dull black velvety spots on leaves, becoming scabby, puckered, and twisted.",
        "organic_remedy": "Apply lime sulfur or potassium bicarbonate sprays during pink-bud and petal-fall stages.",
        "chemical_treatment": "Captan 50% WP (2.5 g/L) or Difenoconazole 25% EC (0.3 ml/L).",
        "prevention": "Shred fallen leaves with a flail mower in autumn to accelerate decomposition of overwintering ascospores."
    },
    "Apple___Black_rot": {
        "crop": "Apple",
        "display_name": "Apple — Black Rot / Frogeye Leaf Spot (Botryosphaeria obtusa)",
        "is_healthy": False,
        "pathogen": "Fungus (Botryosphaeria obtusa)",
        "symptoms": "Small purple spots on leaves expanding into circular 'frogeye' lesions with tan centers and purple rings.",
        "organic_remedy": "Prune out dead wood, cankers, and mummified fruits from branches during winter.",
        "chemical_treatment": "Thiophanate-methyl 70% WP or Captan.",
        "prevention": "Keep trees vigorous through balanced fertilization; avoid mechanical bark wounding."
    },
    "Apple___Cedar_apple_rust": {
        "crop": "Apple",
        "display_name": "Apple — Cedar Apple Rust (Gymnosporangium juniperi-virginianae)",
        "is_healthy": False,
        "pathogen": "Heteroecious Rust Fungus",
        "symptoms": "Bright yellow-orange spots on upper leaf surfaces, developing tiny black pycnia followed by tube-like aecia below.",
        "organic_remedy": "Prune out cedar galls from nearby red cedar (Juniperus virginiana) trees within 1 km radius.",
        "chemical_treatment": "Myclobutanil or Mancozeb sprays applied from pink bud stage through early summer.",
        "prevention": "Eradicate or separate eastern red cedars from orchard vicinity; plant rust-resistant apple varieties."
    },
    "Apple___healthy": {
        "crop": "Apple",
        "display_name": "Apple — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Uniform deep green leaves with crisp margins and no fungal discoloration or deformation.",
        "organic_remedy": "Maintain balanced micronutrient foliar nutrition (Boron, Zinc).",
        "chemical_treatment": "No chemical treatment required.",
        "prevention": "Maintain winter orchard sanitation and balanced annual pruning for canopy light interception."
    },

    # ---------------- GRAPE ----------------
    "Grape___Black_rot": {
        "crop": "Grape",
        "display_name": "Grape — Black Rot (Guignardia bidwellii)",
        "is_healthy": False,
        "pathogen": "Fungus (Guignardia bidwellii)",
        "symptoms": "Small circular reddish-brown leaf spots with dark borders and tiny black fruiting bodies (pycnidia).",
        "organic_remedy": "Apply sulfur or copper formulations before shoot emergence and post-bloom.",
        "chemical_treatment": "Mancozeb 75% WP (2.5 g/L) or Myclobutanil 10% WP (0.5 g/L).",
        "prevention": "Remove all mummified berries from trellises during dormant pruning; maintain open vine canopy."
    },
    "Grape___Esca_(Black_Measles)": {
        "crop": "Grape",
        "display_name": "Grape — Esca / Black Measles (Phaeomoniella chlamydospora)",
        "is_healthy": False,
        "pathogen": "Fungal Complex",
        "symptoms": "'Tiger-stripe' chlorotic and necrotic patterns between leaf veins; dark speckles on fruit skins.",
        "organic_remedy": "Paint dormant pruning wounds with biological Trichoderma paste or wound protectant.",
        "chemical_treatment": "No curative systemic fungicide; manage by trunk renewal and double pruning techniques.",
        "prevention": "Avoid heavy pruning during wet weather. Disinfect pruning shears between individual vines."
    },
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": {
        "crop": "Grape",
        "display_name": "Grape — Leaf Blight (Pseudocercospora vitis)",
        "is_healthy": False,
        "pathogen": "Fungus (Pseudocercospora vitis)",
        "symptoms": "Irregular dark brown patches on leaves, leading to premature leaf drop and weakened cane maturation.",
        "organic_remedy": "Foliar spray of Bordeaux mixture (1%) after fruit harvest.",
        "chemical_treatment": "Copper oxychloride (3 g/L) or Carbendazim (1 g/L).",
        "prevention": "Adopt bower or Y-trellis system to elevate canopy away from ground humidity."
    },
    "Grape___healthy": {
        "crop": "Grape",
        "display_name": "Grape — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Vigorous, broad green palmate leaves with healthy serrated lobes and clean vein structures.",
        "organic_remedy": "Apply organic humic acid and potassium-rich vinasse.",
        "chemical_treatment": "No chemical treatment needed.",
        "prevention": "Ensure good trellis shoot positioning and weed management along vine rows."
    },

    # ---------------- PEPPER BELL ----------------
    "Pepper,_bell___Bacterial_spot": {
        "crop": "Pepper Bell (Capsicum)",
        "display_name": "Bell Pepper — Bacterial Spot (Xanthomonas euvesicatoria)",
        "is_healthy": False,
        "pathogen": "Bacterium (Xanthomonas spp.)",
        "symptoms": "Small, circular to irregular water-soaked spots turning dark brown with pale centers on leaves.",
        "organic_remedy": "Copper soap or Bacillus subtilis biopesticide spray.",
        "chemical_treatment": "Copper Hydroxide mixed with Streptocycline (100 ppm).",
        "prevention": "Use certified pathogen-free seeds; avoid handling plants when foliage is wet from morning dew."
    },
    "Pepper,_bell___healthy": {
        "crop": "Pepper Bell (Capsicum)",
        "display_name": "Bell Pepper — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Smooth, glossy, dark green foliage free from chlorosis, curling, or brown spots.",
        "organic_remedy": "Apply neem cake and vermicompost in root basin.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Mulch soil with reflective silver-black plastic to conserve moisture and deter thrips."
    },

    # ---------------- CITRUS / ORANGE ----------------
    "Orange___Haunglongbing_(Citrus_greening)": {
        "crop": "Orange (Citrus)",
        "display_name": "Citrus — Huanglongbing / Citrus Greening (Candidatus Liberibacter)",
        "is_healthy": False,
        "pathogen": "Fastidious Bacterium (Vector: Asian Citrus Psyllid)",
        "symptoms": "Asymmetrical blotchy mottle on leaves, yellow veins, small lopsided bitter fruits that remain green at bottom.",
        "organic_remedy": "Strictly control psyllid vectors with petroleum horticultural oils or entomopathogenic fungi.",
        "chemical_treatment": "Systemic neonicotinoids (Imidacloprid, Thiamethoxam) to suppress psyllid populations. No cure for infected tree.",
        "prevention": "Plant only certified disease-free budwood from registered screenhouse nurseries. Rogue infected trees."
    },

    # ---------------- PEACH ----------------
    "Peach___Bacterial_spot": {
        "crop": "Peach",
        "display_name": "Peach — Bacterial Spot (Xanthomonas arboricola pv. pruni)",
        "is_healthy": False,
        "pathogen": "Bacterium (Xanthomonas arboricola pv. pruni)",
        "symptoms": "Angular purple-black spots on leaves; dead centers drop out creating a 'shot-hole' appearance.",
        "organic_remedy": "Apply dormant fixed copper sprays before bud swell in early spring.",
        "chemical_treatment": "Oxytetracycline or copper sprays at low rates during early cover.",
        "prevention": "Select resistant varieties; provide windbreaks to minimize sandblasting injury."
    },
    "Peach___healthy": {
        "crop": "Peach",
        "display_name": "Peach — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Elongated lanceolate green leaves with clean margins and no shot-hole spotting.",
        "organic_remedy": "Maintain balanced zinc and nitrogen fertilization.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Prune to open-center vase system for maximum sunlight penetration."
    },

    # ---------------- CHERRY ----------------
    "Cherry_(including_sour)___Powdery_mildew": {
        "crop": "Cherry",
        "display_name": "Cherry — Powdery Mildew (Podosphaera clandestina)",
        "is_healthy": False,
        "pathogen": "Fungus (Podosphaera clandestina)",
        "symptoms": "White powdery fungal patches on young leaves and shoot tips, causing leaf curling and distortion.",
        "organic_remedy": "Spray wettable sulfur (2 g/L) or potassium bicarbonate.",
        "chemical_treatment": "Myclobutanil or Tebuconazole applied at petal fall and shuck fall.",
        "prevention": "Prune dense canopies to promote air drying; avoid excessive early nitrogen fertilization."
    },
    "Cherry_(including_sour)___healthy": {
        "crop": "Cherry",
        "display_name": "Cherry — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Glossy ovate dark green leaves with serrated margins and intact cuticle.",
        "organic_remedy": "Continue good irrigation and orchard weed sanitation.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Scout weekly during warm spring flushes."
    },

    # ---------------- STRAWBERRY ----------------
    "Strawberry___Leaf_scorch": {
        "crop": "Strawberry",
        "display_name": "Strawberry — Leaf Scorch (Diplocarpon earlianum)",
        "is_healthy": False,
        "pathogen": "Fungus (Diplocarpon earlianum)",
        "symptoms": "Numerous small, dark purple irregular blotches that coalesce, giving entire leaf a dry, scorched burnt appearance.",
        "organic_remedy": "Mow and destroy older infected foliage immediately after harvest.",
        "chemical_treatment": "Captan 50% WP or Azoxystrobin.",
        "prevention": "Plant on raised beds with drip irrigation to avoid wet leaves."
    },
    "Strawberry___healthy": {
        "crop": "Strawberry",
        "display_name": "Strawberry — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Trifoliate vibrant green leaves with crisp toothed margins.",
        "organic_remedy": "Apply organic compost and pine needle mulch.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Renovate beds every 3-4 years to avoid disease build-up."
    },

    # ---------------- SQUASH ----------------
    "Squash___Powdery_mildew": {
        "crop": "Squash / Cucurbits",
        "display_name": "Squash — Powdery Mildew (Podosphaera xanthii)",
        "is_healthy": False,
        "pathogen": "Fungus (Podosphaera xanthii)",
        "symptoms": "White talcum-like powdery fungal growth covering upper and lower surfaces of leaves, causing yellowing and premature leaf senescence.",
        "organic_remedy": "Spray diluted cow milk (10% solution in water) or potassium bicarbonate (3 g/L) in bright sunlight.",
        "chemical_treatment": "Difenoconazole (0.5 ml/L) or Dinocap.",
        "prevention": "Choose resistant squash varieties and allow generous plant spacing (1.5 - 2 meters)."
    },

    # ---------------- BLUEBERRY ----------------
    "Blueberry___healthy": {
        "crop": "Blueberry",
        "display_name": "Blueberry — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Elliptical dark green leathery leaves with smooth margins and no chlorosis.",
        "organic_remedy": "Maintain acidic soil pH (4.5 - 5.5) using elemental sulfur and peat moss.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Maintain pine bark mulch around bushes."
    },

    # ---------------- RASPBERRY ----------------
    "Raspberry___healthy": {
        "crop": "Raspberry",
        "display_name": "Raspberry — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Compound serrated green leaves with silver undersides, free of cane blights or rust.",
        "organic_remedy": "Apply well-composted organic mulch.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Prune out spent floricanes after fruiting to promote good airflow."
    },

    # ---------------- SOYBEAN ----------------
    "Soybean___healthy": {
        "crop": "Soybean",
        "display_name": "Soybean — Healthy Leaf",
        "is_healthy": True,
        "pathogen": "None (Plant is healthy)",
        "symptoms": "Trifoliate leaves of uniform medium green color, sturdy petiole, and absence of rust or frogeye spots.",
        "organic_remedy": "Inoculate seeds with Bradyrhizobium japonicum for robust nodulation.",
        "chemical_treatment": "No chemical required.",
        "prevention": "Rotate crops with corn or sorghum; monitor for defoliating caterpillars."
    }
}


def validate_leaf_image(image_input: Any) -> Tuple[bool, Optional[str], Optional[Image.Image]]:
    """
    Validate uploaded image file format, size, byte integrity, and dimensions.
    
    Accepts:
      - File path (str)
      - Bytes / BytesIO
      - Streamlit UploadedFile object
      
    Returns:
      (is_valid, error_message, PIL_Image_or_None)
    """
    if image_input is None:
        return False, "No image file provided. Please upload a leaf photo.", None

    if isinstance(image_input, Image.Image):
        try:
            width, height = image_input.size
            if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
                return (
                    False,
                    f"Image dimensions ({width}x{height}) are too small. Minimum required is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels.",
                    None
                )
            if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
                return (
                    False,
                    f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION} pixels.",
                    None
                )
            img = image_input.convert("RGB") if image_input.mode != "RGB" else image_input
            return True, None, img
        except Exception as e:
            return False, f"Failed to process image: {str(e)}", None

    # Handle Streamlit UploadedFile or file-like object
    filename = getattr(image_input, "name", None)
    if filename:
        ext = os.path.splitext(filename)[1].lower()
        if ext not in SUPPORTED_EXTENSIONS:
            return (
                False,
                f"Unsupported file format '{ext}'. Allowed formats are JPG, JPEG, PNG, and WEBP.",
                None
            )

    try:
        # Read bytes
        if hasattr(image_input, "read"):
            image_bytes = image_input.read()
            # Reset seek position if possible
            if hasattr(image_input, "seek"):
                image_input.seek(0)
        elif isinstance(image_input, (bytes, bytearray)):
            image_bytes = image_input
        elif isinstance(image_input, str):
            if not os.path.exists(image_input):
                return False, f"Image path '{image_input}' does not exist.", None
            with open(image_input, "rb") as f:
                image_bytes = f.read()
        else:
            return False, "Invalid image input type. Please provide a valid image file, path, or bytes.", None

        if len(image_bytes) == 0:
            return False, "Uploaded image file is empty (0 bytes).", None

        # Check maximum size
        size_mb = len(image_bytes) / (1024 * 1024)
        if size_mb > MAX_IMAGE_SIZE_MB:
            return (
                False,
                f"Image file size ({size_mb:.2f} MB) exceeds maximum limit of {MAX_IMAGE_SIZE_MB} MB.",
                None
            )

        # Open and verify via PIL
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()  # Verifies byte integrity

        # Reopen after verify (PIL requirement)
        img = Image.open(io.BytesIO(image_bytes))
        width, height = img.size

        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return (
                False,
                f"Image dimensions ({width}x{height}) are too small. Minimum required is {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION} pixels.",
                None
            )

        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return (
                False,
                f"Image dimensions ({width}x{height}) exceed maximum allowed dimension of {MAX_IMAGE_DIMENSION} pixels.",
                None
            )

        # Convert to RGB mode if necessary (handles RGBA, Grayscale, CMYK, Palette)
        if img.mode != "RGB":
            img = img.convert("RGB")

        return True, None, img

    except Exception as e:
        return False, f"Invalid or corrupted image file: {str(e)}", None


def check_disease_model_status() -> Dict[str, Any]:
    """
    Check if a trained disease classification model is present on disk.
    Ensures transparent communication with the user.
    """
    has_pkl = os.path.exists(DEFAULT_PKL_PATH)
    pth_path = os.path.join(MODEL_DIR, "plant_disease_model.pth")
    has_pth = os.path.exists(pth_path)

    if has_pkl:
        return {
            "is_available": True,
            "model_type": "Scikit-Learn Image Feature Classifier",
            "model_path": DEFAULT_PKL_PATH,
            "message": "Trained disease classification model bundle is loaded and operational."
        }
    elif has_pth:
        return {
            "is_available": True,
            "model_type": "PyTorch Deep Learning Model",
            "model_path": pth_path,
            "message": "Trained PyTorch plant disease model is available."
        }
    else:
        return {
            "is_available": False,
            "model_type": None,
            "model_path": None,
            "message": (
                "Model weights file ('models/plant_disease_model.pkl' or '.pth') is not yet trained or loaded. "
                "Per project integrity rules, disease detection requires real model weights and will not output fake predictions."
            )
        }


def load_disease_model(model_path: Optional[str] = None) -> Tuple[Optional[Any], Optional[str]]:
    """
    Load trained disease classification model bundle.
    Returns: (model_bundle_or_None, error_string)
    """
    path = model_path or DEFAULT_PKL_PATH
    if not os.path.exists(path):
        return None, (
            f"Disease model file not found at '{path}'. "
            "Please run 'python train_disease_model.py' to generate model weights."
        )
    try:
        bundle = joblib.load(path)
        return bundle, None
    except Exception as e:
        return None, f"Failed to load disease model: {str(e)}"


def extract_image_features(image: Image.Image) -> np.ndarray:
    """
    Extract color, texture, and spatial features from a PIL Image
    compatible with the trained image classifier.
    Features:
      - 3-channel RGB histogram (32 bins each = 96 features)
      - 3-channel HSV histogram (32 bins each = 96 features)
      - Channel mean & std dev for RGB & HSV (12 features)
      - 4x4 spatial grid average RGB values (48 features)
    Total features: 252 features
    """
    img_resized = image.resize((128, 128))
    rgb_arr = np.array(img_resized, dtype=np.float32) / 255.0

    # Convert to HSV
    img_hsv = img_resized.convert("HSV")
    hsv_arr = np.array(img_hsv, dtype=np.float32) / 255.0

    features = []

    # 1. RGB Histograms
    for ch in range(3):
        hist, _ = np.histogram(rgb_arr[:, :, ch], bins=32, range=(0.0, 1.0), density=True)
        features.extend(hist)

    # 2. HSV Histograms
    for ch in range(3):
        hist, _ = np.histogram(hsv_arr[:, :, ch], bins=32, range=(0.0, 1.0), density=True)
        features.extend(hist)

    # 3. Channel Means & Stds
    for ch in range(3):
        features.append(float(np.mean(rgb_arr[:, :, ch])))
        features.append(float(np.std(rgb_arr[:, :, ch])))
        features.append(float(np.mean(hsv_arr[:, :, ch])))
        features.append(float(np.std(hsv_arr[:, :, ch])))

    # 4. 4x4 Spatial Grid Average RGB
    grid_h = 128 // 4
    grid_w = 128 // 4
    for i in range(4):
        for j in range(4):
            patch = rgb_arr[i * grid_h:(i + 1) * grid_h, j * grid_w:(j + 1) * grid_w]
            features.append(float(np.mean(patch[:, :, 0])))
            features.append(float(np.mean(patch[:, :, 1])))
            features.append(float(np.mean(patch[:, :, 2])))

    return np.array(features, dtype=np.float32).reshape(1, -1)


def predict_disease(
    image: Image.Image,
    model_bundle: Optional[dict] = None,
    top_k: int = 3
) -> Dict[str, Any]:
    """
    Perform disease prediction on a leaf image.
    
    If model_bundle is None or missing, returns an explicit status error.
    NEVER returns fake or random predictions!
    """
    if model_bundle is None or "model" not in model_bundle:
        return {
            "success": False,
            "error": "Model weights are not loaded. Cannot perform inference without a trained model.",
            "predicted_class": None,
            "display_name": None,
            "prediction": None,
            "confidence": None,
            "top_candidates": [],
            "advisory": None,
            "disclaimer": DISCLAIMER_TEXT
        }

    try:
        model = model_bundle["model"]
        classes = model_bundle.get("classes", list(DISEASE_KNOWLEDGE_BASE.keys()))
        
        # Extract features
        features = extract_image_features(image)

        # Run inference
        probabilities = model.predict_proba(features)[0]
        model_classes = model.classes_

        # Sort top-k
        sorted_indices = np.argsort(probabilities)[::-1]
        top_idx = sorted_indices[0]
        predicted_class = str(model_classes[top_idx])
        top_confidence = float(probabilities[top_idx])

        top_candidates = []
        for idx in sorted_indices[:min(top_k, len(model_classes))]:
            c_name = str(model_classes[idx])
            profile = DISEASE_KNOWLEDGE_BASE.get(c_name, {})
            disp = profile.get("display_name", c_name.replace("___", " - ").replace("_", " "))
            top_candidates.append({
                "class_code": c_name,
                "display_name": disp,
                "probability": round(float(probabilities[idx]) * 100, 2),
                "is_healthy": profile.get("is_healthy", "healthy" in c_name.lower())
            })

        advisory = get_disease_advisory(predicted_class)

        return {
            "success": True,
            "error": None,
            "predicted_class": predicted_class,
            "display_name": advisory.get("display_name", predicted_class),
            "confidence": round(top_confidence * 100, 2),
            "is_healthy": advisory.get("is_healthy", "healthy" in predicted_class.lower()),
            "crop": advisory.get("crop", "Unknown"),
            "top_candidates": top_candidates,
            "advisory": advisory,
            "disclaimer": DISCLAIMER_TEXT
        }
    except Exception as e:
        return {
            "success": False,
            "error": f"Error executing inference: {str(e)}",
            "prediction": None,
            "confidence": None,
            "top_candidates": [],
            "advisory": None,
            "disclaimer": DISCLAIMER_TEXT
        }


def get_disease_advisory(class_key: str) -> Dict[str, Any]:
    """Retrieve structured advisory dictionary for any disease class."""
    if class_key in DISEASE_KNOWLEDGE_BASE:
        return DISEASE_KNOWLEDGE_BASE[class_key]
    
    # Fallback template if custom class
    return {
        "crop": class_key.split("___")[0].replace("_", " ") if "___" in class_key else "Crop",
        "display_name": class_key.replace("___", " — ").replace("_", " "),
        "is_healthy": "healthy" in class_key.lower(),
        "pathogen": "Diagnosed Plant Condition",
        "symptoms": "Leaf tissue discoloration, lesions, or morphological alterations.",
        "organic_remedy": "Isolate affected plant parts; apply organic bio-fungicide or neem-based spray.",
        "chemical_treatment": "Consult local Krishi Vigyan Kendra (KVK) for recommended approved active ingredients.",
        "prevention": "Maintain clean field sanitation, avoid waterlogging, and rotate crops."
    }


def search_disease_catalog(query: str = "", crop_filter: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Search through the 38 PlantVillage disease entries by keyword, crop, or symptom.
    """
    query_clean = query.lower().strip()
    results = []

    for key, data in DISEASE_KNOWLEDGE_BASE.items():
        # Apply crop filter if provided
        if crop_filter and crop_filter.lower() != "all":
            if data["crop"].lower() != crop_filter.lower():
                continue

        # Check keyword matches
        if not query_clean:
            results.append({"class_key": key, **data})
            continue

        searchable_text = f"{key} {data['crop']} {data['display_name']} {data['pathogen']} {data['symptoms']} {data['organic_remedy']}".lower()
        if query_clean in searchable_text:
            results.append({"class_key": key, **data})

    return results
