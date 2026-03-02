"""
╔══════════════════════════════════════════════════════════════════════════════╗
║      GURGAON REAL ESTATE AI — END-TO-END DATA SCIENCE PROJECT              ║
║                                                                              ║
║  Techniques:  NLP (TF-IDF + LSA + Keyword Sentiment)                        ║
║               ML  (Linear, Ridge, Lasso, Random Forest, Gradient Boosting)  ║
║               DL  (Residual MLP — TensorFlow/Keras)                         ║
║                                                                              ║
║  Run:  python gurgaon_realestate_main.py                                     ║
║  Output: ./output/  — CSV, trained models, charts, full PDF report           ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import os, sys, re, json, time, pickle, random, warnings
from collections import Counter

import numpy  as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot  as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition            import TruncatedSVD
from sklearn.pipeline                 import Pipeline
from sklearn.preprocessing            import LabelEncoder, StandardScaler
from sklearn.model_selection          import train_test_split, cross_val_score, KFold
from sklearn.linear_model             import LinearRegression, Ridge, Lasso
from sklearn.ensemble                 import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics                  import (mean_absolute_error, mean_squared_error,
                                              r2_score, mean_absolute_percentage_error)

warnings.filterwarnings("ignore")
np.random.seed(42)
random.seed(42)

# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT DIRECTORY
# ─────────────────────────────────────────────────────────────────────────────
OUT = "output"
os.makedirs(OUT, exist_ok=True)

DIVIDER = "═" * 70

def header(title: str):
    print(f"\n{DIVIDER}")
    print(f"  {title}")
    print(DIVIDER)


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 1 — SYNTHETIC DATASET GENERATION
# ═════════════════════════════════════════════════════════════════════════════
header("SECTION 1 · SYNTHETIC DATASET GENERATION")

SECTORS = {
    "Sector 14":            {"base_price": 18000, "tier": "premium"},
    "Sector 29":            {"base_price": 16500, "tier": "premium"},
    "Sector 57":            {"base_price": 12000, "tier": "mid"},
    "Sector 65":            {"base_price": 13500, "tier": "mid"},
    "DLF Phase 1":          {"base_price": 22000, "tier": "luxury"},
    "DLF Phase 2":          {"base_price": 20000, "tier": "luxury"},
    "DLF Phase 3":          {"base_price": 21000, "tier": "luxury"},
    "DLF Phase 4":          {"base_price": 19000, "tier": "luxury"},
    "DLF Phase 5":          {"base_price": 25000, "tier": "ultra_luxury"},
    "Golf Course Road":     {"base_price": 28000, "tier": "ultra_luxury"},
    "Golf Course Extension":{"base_price": 15000, "tier": "mid"},
    "Sohna Road":           {"base_price":  9000, "tier": "affordable"},
    "NH-48 Corridor":       {"base_price": 10000, "tier": "affordable"},
    "Palam Vihar":          {"base_price":  8500, "tier": "affordable"},
    "South City 1":         {"base_price": 14000, "tier": "mid"},
    "South City 2":         {"base_price": 13000, "tier": "mid"},
    "Nirvana Country":      {"base_price": 11500, "tier": "mid"},
    "Malibu Towne":         {"base_price": 12500, "tier": "mid"},
    "Sector 82":            {"base_price":  7500, "tier": "affordable"},
    "Sector 84":            {"base_price":  7800, "tier": "affordable"},
    "Manesar":              {"base_price":  5500, "tier": "budget"},
    "Bhondsi":              {"base_price":  4800, "tier": "budget"},
    "Badshahpur":           {"base_price":  5200, "tier": "budget"},
    "Huda Sectors":         {"base_price":  9500, "tier": "affordable"},
    "Cyber City":           {"base_price": 24000, "tier": "ultra_luxury"},
}

PROPERTY_TYPES  = ["Apartment","Villa","Builder Floor","Penthouse","Studio","Independent House"]
FURNISHINGS     = ["Unfurnished","Semi-Furnished","Fully Furnished"]
FACINGS         = ["North","South","East","West","North-East","North-West","South-East"]
BUILDER_GRADES  = ["A-Grade","B-Grade","Tier-1","Tier-2","Independent"]

AMENITIES_POOL = [
    "Swimming Pool","Gym","Clubhouse","24/7 Security","Power Backup","Parking",
    "Garden","Jogging Track","Tennis Court","Squash Court","Badminton Court",
    "Kids Play Area","Amphitheatre","Library","Business Center","EV Charging",
    "Rainwater Harvesting","Solar Panel","Concierge Service","Valet Parking",
    "Rooftop Lounge","Co-working Space","Meditation Center","Spa","Intercom",
    "Video Surveillance",
]

NEARBY_PLACES = [
    "metro station","school","hospital","mall","IT park","expressway",
    "airport","supermarket","park","bank","restaurant hub",
]

EXTRA_DESCS = [
    "Ideal for end-use and investment.",
    "RERA registered society.",
    "Ready to move in.",
    "Under construction, possession in 2025.",
    "Low maintenance charges.",
    "Vastu compliant.",
    "Corner unit with abundant natural light.",
    "High-rise tower with breathtaking city views.",
    "Green certified building.",
    "Pet-friendly society.",
    "Gated community with 3-tier security.",
    "Modular kitchen and premium fittings included.",
]

DESC_TEMPLATES = [
    ("Stunning {bhk}BHK {ptype} in the heart of {sector}. The property features {a1} and {a2}. "
     "With {area} sq ft carpet area, this {furn} unit offers panoramic views. "
     "Proximity to {n1} and {n2} makes it ideal for families and professionals alike. {extra}"),
    ("Luxurious {bhk}BHK {ptype} available in {sector}. Spread across {area} sq ft, "
     "this {furn} property offers world-class {a1}, {a2}, and {a3}. "
     "Located near {n1}, offering excellent connectivity. {extra}"),
    ("Well-maintained {bhk}BHK {ptype} in {sector}. This {furn} home spans {area} sq ft "
     "and is equipped with {a1} and {a2}. Close to {n1} and {n2}. {extra}"),
    ("Premium {bhk}BHK {ptype} for sale in {sector}. The {furn} unit covers {area} sq ft "
     "and offers {a1}, {a2}, {a3}. Near {n1}. {extra}"),
    ("Elegant {bhk}BHK {ptype} nestled in {sector}. {area} sq ft {furn} apartment "
     "with {a1} and {a2}. Walking distance from {n1}. {extra}"),
]

def _generate_description(bhk, ptype, sector, area, furn) -> str:
    tpl  = random.choice(DESC_TEMPLATES)
    ams  = random.sample(AMENITIES_POOL, 4)
    return tpl.format(
        bhk=int(bhk), ptype=ptype, sector=sector, area=int(area), furn=furn,
        a1=ams[0], a2=ams[1], a3=ams[2],
        n1=random.choice(NEARBY_PLACES),
        n2=random.choice(NEARBY_PLACES),
        extra=random.choice(EXTRA_DESCS),
    )

def generate_dataset(n: int = 5000) -> pd.DataFrame:
    BHK_WEIGHTS = {
        "budget":       [0.40, 0.40, 0.15, 0.05],
        "affordable":   [0.25, 0.45, 0.25, 0.05],
        "mid":          [0.10, 0.35, 0.40, 0.15],
        "premium":      [0.05, 0.20, 0.45, 0.30],
        "luxury":       [0.02, 0.15, 0.43, 0.40],
        "ultra_luxury": [0.01, 0.09, 0.40, 0.50],
    }
    FURN_MULT  = {"Unfurnished": 1.00, "Semi-Furnished": 1.05, "Fully Furnished": 1.12}
    GRADE_MULT = {"A-Grade": 1.10, "Tier-1": 1.08, "B-Grade": 1.00, "Tier-2": 0.95, "Independent": 0.92}
    BHK_BASE_AREA = {1: 450, 2: 850, 3: 1350, 4: 2100}

    records = []
    sector_names = list(SECTORS.keys())

    for _ in range(n):
        sector     = random.choice(sector_names)
        info       = SECTORS[sector]
        tier       = info["tier"]
        base_price = info["base_price"]

        bhk = int(np.random.choice([1, 2, 3, 4], p=BHK_WEIGHTS[tier]))
        prop_type = random.choice(PROPERTY_TYPES)
        if tier in ("ultra_luxury", "luxury") and random.random() > 0.6:
            prop_type = random.choice(["Penthouse", "Villa"])

        base_area   = BHK_BASE_AREA[bhk]
        carpet_area = int(np.clip(np.random.normal(base_area, base_area * 0.15), 300, 6000))
        super_area  = int(carpet_area * np.random.uniform(1.25, 1.45))
        floor       = np.random.randint(0, 30)
        total_floors= max(floor + np.random.randint(0, 10), 4)
        age_years   = np.random.randint(0, 20)
        bathrooms   = min(bhk + np.random.randint(0, 2), bhk + 1)
        balconies   = np.random.randint(0, 4)
        furnishing  = random.choice(FURNISHINGS)
        facing      = random.choice(FACINGS)
        builder_grade = random.choice(BUILDER_GRADES)
        parking     = np.random.choice([0, 1, 2], p=[0.10, 0.60, 0.30])

        amenity_count = np.random.randint(3, len(AMENITIES_POOL))
        amenity_score = round(amenity_count / len(AMENITIES_POOL) * 10, 2)

        dist_metro    = round(np.random.exponential(2.5), 2)
        dist_school   = round(np.random.exponential(1.5), 2)
        dist_hospital = round(np.random.exponential(3.0), 2)
        dist_mall     = round(np.random.exponential(4.0), 2)
        dist_airport  = round(np.random.uniform(10, 40), 2)

        price_per_sqft = (
            base_price
            * FURN_MULT[furnishing]
            * (1 + (floor / total_floors) * 0.10)
            * max(0.75, 1 - age_years * 0.012)
            * max(0.90, 1.08 - dist_metro * 0.03)
            * GRADE_MULT[builder_grade]
            * np.random.normal(1, 0.05)
        )
        total_price_cr = round((price_per_sqft * carpet_area) / 1e7, 3)

        records.append({
            "sector": sector, "tier": tier, "property_type": prop_type,
            "bhk": bhk, "carpet_area": carpet_area, "super_area": super_area,
            "floor": floor, "total_floors": total_floors, "age_years": age_years,
            "bathrooms": bathrooms, "balconies": balconies,
            "furnishing": furnishing, "facing": facing,
            "builder_grade": builder_grade, "parking": parking,
            "amenity_score": amenity_score,
            "dist_metro_km": dist_metro, "dist_school_km": dist_school,
            "dist_hospital_km": dist_hospital, "dist_mall_km": dist_mall,
            "dist_airport_km": dist_airport,
            "price_per_sqft": round(price_per_sqft, 0),
            "price_cr": total_price_cr,
            "description": _generate_description(bhk, prop_type, sector, carpet_area, furnishing),
        })

    df = pd.DataFrame(records)
    csv_path = os.path.join(OUT, "gurgaon_properties.csv")
    df.to_csv(csv_path, index=False)
    print(f"  ✔  Dataset generated  →  {csv_path}")
    print(f"     Shape   : {df.shape}")
    print(f"     Sectors : {df['sector'].nunique()}")
    print(f"     Price   : ₹{df['price_cr'].min():.2f} – ₹{df['price_cr'].max():.2f} Cr  "
          f"(mean ₹{df['price_cr'].mean():.2f} Cr)")
    return df


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 2 — EXPLORATORY DATA ANALYSIS
# ═════════════════════════════════════════════════════════════════════════════
header("SECTION 2 · EXPLORATORY DATA ANALYSIS")

def run_eda(df: pd.DataFrame):
    print("  Basic statistics:")
    desc = df["price_cr"].describe()
    for k, v in desc.items():
        print(f"     {k:6s}: {v:.3f}")

    # ── Fig 1: Price distributions ──────────────────────────────────────────
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Gurgaon Real Estate — Exploratory Data Analysis",
                 fontsize=15, fontweight="bold", y=1.01)

    # Overall histogram
    ax = axes[0, 0]
    ax.hist(df["price_cr"], bins=60, color="#4C9BE8", edgecolor="white", alpha=0.85)
    ax.axvline(df["price_cr"].mean(),   color="red",    linestyle="--", lw=2,
               label=f"Mean ₹{df['price_cr'].mean():.2f}Cr")
    ax.axvline(df["price_cr"].median(), color="orange", linestyle="--", lw=2,
               label=f"Median ₹{df['price_cr'].median():.2f}Cr")
    ax.set_title("Price Distribution"); ax.set_xlabel("Price (Cr)"); ax.set_ylabel("Count")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.spines[["top","right"]].set_visible(False)

    # Log-transformed
    ax = axes[0, 1]
    ax.hist(np.log1p(df["price_cr"]), bins=60, color="#E87B4C", edgecolor="white", alpha=0.85)
    ax.set_title("Log-Price Distribution (more Gaussian → better regression)")
    ax.set_xlabel("log(1 + Price)"); ax.set_ylabel("Count")
    ax.grid(alpha=0.3, axis="y"); ax.spines[["top","right"]].set_visible(False)

    # By BHK
    ax = axes[1, 0]
    colors = ["#4C9BE8","#E87B4C","#4CE87B","#E84C9B"]
    for i, bhk in enumerate(sorted(df["bhk"].unique())):
        ax.hist(df[df["bhk"] == bhk]["price_cr"], bins=30, alpha=0.65,
                label=f"{int(bhk)} BHK", color=colors[i % len(colors)])
    ax.set_title("Price by BHK"); ax.set_xlabel("Price (Cr)")
    ax.legend(); ax.grid(alpha=0.3, axis="y"); ax.spines[["top","right"]].set_visible(False)

    # Scatter price vs area
    ax = axes[1, 1]
    sample = df.sample(min(1200, len(df)), random_state=42)
    sc = ax.scatter(sample["carpet_area"], sample["price_cr"],
                    c=sample["bhk"], cmap="RdYlGn", alpha=0.45, s=18)
    plt.colorbar(sc, ax=ax, label="BHK")
    ax.set_title("Price vs Carpet Area"); ax.set_xlabel("Carpet Area (sq ft)")
    ax.set_ylabel("Price (Cr)"); ax.grid(alpha=0.3); ax.spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(OUT, "fig01_eda_distributions.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  EDA chart saved   →  {p}")

    # ── Fig 2: Location & tier analysis ─────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Location & Tier Analysis", fontsize=14, fontweight="bold")

    # Top-10 sectors by avg price
    top10 = df.groupby("sector")["price_cr"].mean().nlargest(10)
    clrs  = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top10)))[::-1]
    bars  = axes[0].barh(top10.index, top10.values, color=clrs, edgecolor="white")
    for b, v in zip(bars, top10.values):
        axes[0].text(b.get_width() + 0.05, b.get_y() + b.get_height() / 2,
                     f"₹{v:.1f}Cr", va="center", fontsize=8)
    axes[0].set_xlabel("Avg Price (Cr)"); axes[0].set_title("Top 10 Sectors by Average Price")
    axes[0].grid(alpha=0.3, axis="x"); axes[0].spines[["top","right"]].set_visible(False)

    # Tier box-plot
    tier_order = ["budget","affordable","mid","premium","luxury","ultra_luxury"]
    tier_data  = [df[df["tier"] == t]["price_cr"].values
                  for t in tier_order if t in df["tier"].unique()]
    tier_labels = [t for t in tier_order if t in df["tier"].unique()]
    bp = axes[1].boxplot(tier_data, labels=tier_labels, patch_artist=True, notch=True)
    palette = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(tier_labels)))
    for patch, c in zip(bp["boxes"], palette):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    axes[1].set_ylabel("Price (Cr)"); axes[1].set_title("Price Range by Location Tier")
    plt.setp(axes[1].get_xticklabels(), rotation=20, ha="right")
    axes[1].grid(alpha=0.3, axis="y"); axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(OUT, "fig02_location_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  Location chart    →  {p}")

    # ── Fig 3: Correlation heatmap ───────────────────────────────────────────
    num_cols = ["bhk","carpet_area","floor","age_years","bathrooms",
                "amenity_score","dist_metro_km","dist_mall_km","price_cr"]
    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdYlGn",
                center=0, ax=ax, linewidths=0.5, cbar_kws={"shrink": 0.8})
    ax.set_title("Feature Correlation Matrix", fontsize=13, fontweight="bold")
    plt.tight_layout()
    p = os.path.join(OUT, "fig03_correlation_heatmap.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  Correlation chart →  {p}")

    # Top corr with price
    corr_price = df[num_cols].corr()["price_cr"].drop("price_cr").sort_values(ascending=False)
    print("\n  Top correlations with price_cr:")
    for feat, val in corr_price.items():
        bar = "█" * int(abs(val) * 20)
        sign = "+" if val >= 0 else "-"
        print(f"     {feat:22s}  {sign}{bar}  {val:+.3f}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 3 — NLP FEATURE EXTRACTION
# ═════════════════════════════════════════════════════════════════════════════
header("SECTION 3 · NLP FEATURE EXTRACTION (TF-IDF + LSA + Keyword Sentiment)")

# ── Lexicons ──────────────────────────────────────────────────────────────────
LUXURY_KEYWORDS = [
    "luxury","premium","elegant","stunning","lavish","opulent","exclusive",
    "world-class","magnificent","breathtaking","prestigious","ultra","high-end",
    "sophisticated","palatial",
]
POSITIVE_KEYWORDS = [
    "spacious","bright","airy","modern","renovated","well-maintained","peaceful",
    "quiet","green","certified","vastu","rera","panoramic","excellent","prime",
    "ideal","convenient","affordable",
]
NEGATIVE_KEYWORDS = [
    "noisy","small","cramped","old","dark","limited","tight","dated",
    "basic","rundown","maintenance",
]
AMENITY_KEYWORDS = [
    "pool","gym","clubhouse","security","backup","parking","garden","jogging",
    "tennis","squash","badminton","kids","spa","concierge","ev","solar",
    "rooftop","library","business",
]


def _clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_keyword_features(df: pd.DataFrame, desc_col: str = "description") -> pd.DataFrame:
    """Branch-1 of NLP pipeline: 27 hand-crafted keyword features."""
    texts = df[desc_col].apply(_clean_text)
    feats = pd.DataFrame(index=df.index)

    feats["nlp_luxury_score"]    = texts.apply(lambda t: sum(1 for w in LUXURY_KEYWORDS   if w in t))
    feats["nlp_positive_score"]  = texts.apply(lambda t: sum(1 for w in POSITIVE_KEYWORDS if w in t))
    feats["nlp_negative_score"]  = texts.apply(lambda t: sum(1 for w in NEGATIVE_KEYWORDS if w in t))
    feats["nlp_amenity_mentions"]= texts.apply(lambda t: sum(1 for w in AMENITY_KEYWORDS  if w in t))
    feats["nlp_overall_sentiment"]= (feats["nlp_luxury_score"]
                                    + feats["nlp_positive_score"]
                                    - feats["nlp_negative_score"])
    feats["nlp_ready_to_move"]   = texts.apply(lambda t: int("ready to move" in t)).astype(int)
    feats["nlp_under_construction"] = texts.apply(lambda t: int("under construction" in t)).astype(int)
    feats["nlp_desc_length"]     = texts.apply(len)
    feats["nlp_word_count"]      = texts.apply(lambda t: len(t.split()))
    feats["nlp_unique_words"]    = texts.apply(lambda t: len(set(t.split())))

    for kw in ["rera","vastu","metro","school","hospital","mall","park","gym",
               "pool","security","corner","green"]:
        feats[f"nlp_has_{kw}"] = texts.apply(lambda t, w=kw: int(w in t)).astype(int)

    return feats


class TFIDFLatentFeatures:
    """Branch-2: TF-IDF (bigrams, 3000 terms) → TruncatedSVD (LSA)."""

    def __init__(self, n_components: int = 15):
        self.pipeline = Pipeline([
            ("tfidf", TfidfVectorizer(
                max_features=3000, ngram_range=(1, 2),
                stop_words="english", min_df=3, sublinear_tf=True,
            )),
            ("svd", TruncatedSVD(n_components=n_components, random_state=42)),
        ])
        self.n_components = n_components

    def fit_transform(self, texts: pd.Series) -> pd.DataFrame:
        arr = self.pipeline.fit_transform(texts.apply(_clean_text))
        return pd.DataFrame(arr, columns=[f"nlp_lsa_{i}" for i in range(self.n_components)],
                            index=texts.index)

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        arr = self.pipeline.transform(texts.apply(_clean_text))
        return pd.DataFrame(arr, columns=[f"nlp_lsa_{i}" for i in range(self.n_components)],
                            index=texts.index)


def build_nlp_features(df: pd.DataFrame) -> tuple:
    """Returns (nlp_feature_df, fitted_lsa_model)."""
    kw   = extract_keyword_features(df)
    lsa  = TFIDFLatentFeatures(n_components=15)
    lsa_df = lsa.fit_transform(df["description"])
    return pd.concat([kw, lsa_df], axis=1), lsa


def transform_nlp_features(df: pd.DataFrame, lsa: TFIDFLatentFeatures) -> pd.DataFrame:
    kw    = extract_keyword_features(df)
    lsa_df = lsa.transform(df["description"])
    return pd.concat([kw, lsa_df], axis=1)


def plot_nlp_analysis(df: pd.DataFrame):
    """Fig 4: NLP word-frequency and luxury-vs-price correlation."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("NLP Analysis of Property Descriptions", fontsize=14, fontweight="bold")

    # Top-20 descriptive words
    all_words = []
    for d in df["description"].sample(min(800, len(df)), random_state=42):
        words = re.findall(r"\b[a-z]{4,}\b", d.lower())
        STOP  = {"this","that","with","from","have","will","been","very","also","more",
                 "than","they","their","make","room","area","bhk","unit","located","near"}
        all_words.extend([w for w in words if w not in STOP])
    wc = Counter(all_words).most_common(20)
    words_, counts_ = zip(*wc)
    clrs = ["#9B4CE8" if w in LUXURY_KEYWORDS + POSITIVE_KEYWORDS else "#4C9BE8"
            for w in words_]
    axes[0].barh(words_[::-1], counts_[::-1], color=clrs[::-1], edgecolor="white")
    axes[0].set_xlabel("Frequency"); axes[0].set_title("Top 20 Description Words\n(purple = sentiment words)")
    axes[0].grid(alpha=0.3, axis="x"); axes[0].spines[["top","right"]].set_visible(False)

    # Luxury word count vs price
    sample = df.sample(min(1000, len(df)), random_state=42).copy()
    sample["luxury_count"] = sample["description"].apply(
        lambda t: sum(1 for w in LUXURY_KEYWORDS if w in t.lower())
    )
    for lv in sorted(sample["luxury_count"].unique())[:8]:
        prices = sample[sample["luxury_count"] == lv]["price_cr"]
        axes[1].scatter([lv] * len(prices), prices, alpha=0.25, s=12)
    # Trend line
    lc = sample["luxury_count"].values
    pc = sample["price_cr"].values
    m, b = np.polyfit(lc, pc, 1)
    axes[1].plot(sorted(set(lc)), [m * x + b for x in sorted(set(lc))],
                 color="red", linewidth=2, linestyle="--", label=f"Trend (slope={m:.2f})")
    axes[1].set_xlabel("Luxury Word Count in Description")
    axes[1].set_ylabel("Price (Crores)")
    axes[1].set_title("NLP Luxury Language → Price Correlation")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(OUT, "fig04_nlp_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  NLP chart saved   →  {p}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 4 — PREPROCESSING PIPELINE
# ═════════════════════════════════════════════════════════════════════════════
header("SECTION 4 · PREPROCESSING & FEATURE ENGINEERING")

CATEGORICAL_COLS = ["sector","property_type","furnishing","facing","builder_grade","tier"]
NUMERICAL_COLS   = [
    "bhk","carpet_area","super_area","floor","total_floors","age_years",
    "bathrooms","balconies","parking","amenity_score",
    "dist_metro_km","dist_school_km","dist_hospital_km","dist_mall_km","dist_airport_km",
]
TARGET = "price_cr"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["area_ratio"]         = df["carpet_area"] / (df["super_area"] + 1)
    df["floor_ratio"]        = df["floor"] / (df["total_floors"] + 1)
    df["bath_per_bhk"]       = df["bathrooms"] / df["bhk"]
    df["age_bucket"]         = pd.cut(
        df["age_years"], bins=[-1, 0, 3, 7, 15, 100],
        labels=["new","recent","established","mature","old"]
    ).astype(str)
    df["metro_accessible"]   = (df["dist_metro_km"] <= 1.5).astype(int)
    df["premium_location"]   = df["tier"].isin(["luxury","ultra_luxury"]).astype(int)
    df["total_connectivity"] = (
        1 / (df["dist_metro_km"]  + 0.1) +
        1 / (df["dist_school_km"] + 0.1) +
        1 / (df["dist_mall_km"]   + 0.1)
    )
    df["log_carpet_area"]    = np.log1p(df["carpet_area"])
    df["log_dist_metro"]     = np.log1p(df["dist_metro_km"])
    if TARGET in df.columns:
        df["price_per_sqft_approx"] = (df[TARGET] * 1e7) / df["carpet_area"]
    else:
        df["price_per_sqft_approx"] = df.get("price_per_sqft", 10000)
    return df


class FullPreprocessingPipeline:
    def __init__(self):
        self.label_encoders: dict  = {}
        self.scaler                = StandardScaler()
        self.lsa_model             = None
        self.feature_cols: list    = []

    # ── Training ─────────────────────────────────────────────────────────────
    def fit_transform(self, df: pd.DataFrame):
        df = engineer_features(df)
        nlp_df, self.lsa_model = build_nlp_features(df)
        nlp_df.index = df.index

        cat_cols = CATEGORICAL_COLS + ["age_bucket"]
        cat_enc  = pd.DataFrame(index=df.index)
        for col in cat_cols:
            le = LabelEncoder()
            cat_enc[col + "_enc"] = le.fit_transform(df[col].astype(str))
            self.label_encoders[col] = le

        extra_num = ["area_ratio","floor_ratio","bath_per_bhk","metro_accessible",
                     "premium_location","total_connectivity","log_carpet_area","log_dist_metro",
                     "price_per_sqft_approx"]
        num_df = df[NUMERICAL_COLS + extra_num].copy()

        X = pd.concat([num_df, cat_enc, nlp_df], axis=1).fillna(0)
        y = np.log1p(df[TARGET])

        X_scaled = pd.DataFrame(
            self.scaler.fit_transform(X), columns=X.columns, index=X.index
        )
        self.feature_cols = list(X.columns)
        return X_scaled, y

    # ── Inference ─────────────────────────────────────────────────────────────
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        df = engineer_features(df)
        nlp_df = transform_nlp_features(df, self.lsa_model)
        nlp_df.index = df.index

        cat_cols = CATEGORICAL_COLS + ["age_bucket"]
        cat_enc  = pd.DataFrame(index=df.index)
        for col in cat_cols:
            le  = self.label_encoders[col]
            val = df[col].astype(str).apply(
                lambda x: x if x in le.classes_ else le.classes_[0]
            )
            cat_enc[col + "_enc"] = le.transform(val)

        extra_num = ["area_ratio","floor_ratio","bath_per_bhk","metro_accessible",
                     "premium_location","total_connectivity","log_carpet_area","log_dist_metro",
                     "price_per_sqft_approx"]
        num_df = df[NUMERICAL_COLS + extra_num].copy()

        X = pd.concat([num_df, cat_enc, nlp_df], axis=1).fillna(0)
        for col in self.feature_cols:
            if col not in X.columns:
                X[col] = 0
        X = X[self.feature_cols]
        return pd.DataFrame(self.scaler.transform(X), columns=self.feature_cols, index=X.index)

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path: str):
        with open(path, "rb") as f:
            return pickle.load(f)


def prepare_data(df: pd.DataFrame, test_size: float = 0.20):
    pipeline = FullPreprocessingPipeline()
    X, y = pipeline.fit_transform(df)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"  ✔  Features engineered : {X.shape[1]} total features")
    nlp_n = sum(1 for c in X.columns if c.startswith("nlp_"))
    cat_n = sum(1 for c in X.columns if c.endswith("_enc"))
    num_n = X.shape[1] - nlp_n - cat_n
    print(f"       Numerical          : {num_n}")
    print(f"       Categorical (enc.) : {cat_n}")
    print(f"       NLP-derived        : {nlp_n}")
    print(f"  ✔  Train / Test split  : {X_tr.shape[0]} / {X_te.shape[0]}")
    return X_tr, X_te, y_tr, y_te, pipeline


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 5 — ML MODELS
# ═════════════════════════════════════════════════════════════════════════════
header("SECTION 5 · MACHINE LEARNING MODELS")


def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    t0 = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - t0

    preds  = np.expm1(model.predict(X_te))
    actual = np.expm1(y_te)

    mae  = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2   = r2_score(actual, preds)
    mape = mean_absolute_percentage_error(actual, preds) * 100

    cv_scores = cross_val_score(model, X_tr, y_tr,
                                cv=KFold(5, shuffle=True, random_state=42),
                                scoring="r2", n_jobs=-1)

    print(f"  {name:30s}  R²={r2:.4f}  MAE={mae:.3f}Cr  "
          f"RMSE={rmse:.3f}Cr  MAPE={mape:.1f}%  "
          f"CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}  "
          f"[{train_time:.1f}s]")
    return {
        "name": name, "model": model,
        "r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
        "cv_r2": cv_scores.mean(), "cv_std": cv_scores.std(),
        "predictions": preds, "actual": actual,
    }


def train_ml_models(X_tr, X_te, y_tr, y_te) -> dict:
    model_defs = {
        "Linear Regression":  LinearRegression(),
        "Ridge Regression":   Ridge(alpha=1.0),
        "Lasso Regression":   Lasso(alpha=0.001, max_iter=5000),
        "Random Forest":      RandomForestRegressor(
            n_estimators=300, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, max_features="sqrt", n_jobs=-1, random_state=42,
        ),
        "Gradient Boosting":  GradientBoostingRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=6,
            subsample=0.8, min_samples_split=5, random_state=42,
        ),
    }

    # Optional: XGBoost and LightGBM
    try:
        from xgboost import XGBRegressor
        model_defs["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            random_state=42, verbosity=0,
        )
    except ImportError:
        pass

    try:
        from lightgbm import LGBMRegressor
        model_defs["LightGBM"] = LGBMRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1,
        )
    except ImportError:
        pass

    results = {}
    for name, model in model_defs.items():
        results[name] = evaluate_model(name, model, X_tr, X_te, y_tr, y_te)

    best = max(results.values(), key=lambda x: x["r2"])
    print(f"\n  ✔  Best ML model: {best['name']}  (R²={best['r2']:.4f})")
    return results, best


def get_feature_importance(model, feature_names, top_n=25) -> pd.DataFrame | None:
    if hasattr(model, "feature_importances_"):
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        return fi.head(top_n)
    elif hasattr(model, "coef_"):
        fi = pd.DataFrame({
            "feature": feature_names,
            "importance": np.abs(model.coef_),
        }).sort_values("importance", ascending=False)
        return fi.head(top_n)
    return None


def plot_model_results(results: dict, best_model_name: str):
    """Fig 5 & 6: Model comparison + prediction scatter."""

    # ── Fig 5: Comparison bars ───────────────────────────────────────────────
    names = list(results.keys())
    r2s   = [r["r2"]   for r in results.values()]
    maes  = [r["mae"]  for r in results.values()]
    mapes = [r["mape"] for r in results.values()]
    cv_r2s= [r["cv_r2"] for r in results.values()]

    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("ML Model Performance Comparison", fontsize=14, fontweight="bold")

    def _bar(ax, vals, title, ylabel, color, best_idx=None):
        clrs = [("#28a745" if i == best_idx else color) for i in range(len(names))]
        bars = ax.bar(range(len(names)), vals, color=clrs, edgecolor="white", width=0.6)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=9)
        ax.set_ylabel(ylabel); ax.set_title(title)
        ax.grid(alpha=0.3, axis="y"); ax.spines[["top","right"]].set_visible(False)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() * 1.008,
                    f"{v:.3f}", ha="center", fontsize=8, fontweight="bold")

    best_idx = max(range(len(r2s)), key=lambda i: r2s[i])
    _bar(axes[0,0], r2s,   "R² Score (↑ higher is better)",   "R²",       "#4C9BE8", best_idx)
    _bar(axes[0,1], maes,  "MAE in Crores (↓ lower is better)","MAE (Cr)", "#E87B4C",
         min(range(len(maes)), key=lambda i: maes[i]))
    _bar(axes[1,0], mapes, "MAPE % (↓ lower is better)",      "MAPE %",   "#4CE87B",
         min(range(len(mapes)), key=lambda i: mapes[i]))
    _bar(axes[1,1], cv_r2s,"CV R² (5-Fold Cross-Validation)",  "CV R²",    "#E84C9B", best_idx)

    plt.tight_layout()
    p = os.path.join(OUT, "fig05_model_comparison.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  Model comparison  →  {p}")

    # ── Fig 6: Prediction scatter + residuals for best model ─────────────────
    br  = results[best_model_name]
    actual, preds = br["actual"], br["predictions"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"{best_model_name} — Prediction Analysis", fontsize=13, fontweight="bold")

    lim = [min(actual.min(), preds.min()) * 0.9, max(actual.max(), preds.max()) * 1.1]
    axes[0].scatter(actual, preds, alpha=0.3, s=15, color="#4C9BE8", edgecolors="none")
    axes[0].plot(lim, lim, "r--", linewidth=2, label="Perfect Prediction")
    axes[0].set_xlabel("Actual Price (Cr)"); axes[0].set_ylabel("Predicted Price (Cr)")
    axes[0].set_title("Actual vs Predicted"); axes[0].legend()
    axes[0].grid(alpha=0.3); axes[0].spines[["top","right"]].set_visible(False)

    residuals = actual - preds
    axes[1].hist(residuals, bins=50, color="#4C9BE8", edgecolor="white", alpha=0.85)
    axes[1].axvline(0, color="red", linestyle="--", linewidth=2)
    axes[1].axvline(residuals.mean(), color="orange", linestyle="--",
                    linewidth=1.5, label=f"Mean residual: {residuals.mean():.3f}")
    axes[1].set_xlabel("Residual (Cr)"); axes[1].set_ylabel("Frequency")
    axes[1].set_title("Residual Distribution"); axes[1].legend()
    axes[1].grid(alpha=0.3); axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(OUT, "fig06_prediction_analysis.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  Prediction chart  →  {p}")


def plot_feature_importance(fi_df: pd.DataFrame, top_n: int = 25):
    fi_top = fi_df.head(top_n)
    fig, ax = plt.subplots(figsize=(10, 8))

    clrs = []
    for f in fi_top["feature"]:
        if   f.startswith("nlp_"):                   clrs.append("#9B4CE8")  # purple = NLP
        elif f in ("sector_enc","tier_enc"):          clrs.append("#E87B4C")  # orange = location
        elif f in ("carpet_area","super_area","bhk","log_carpet_area"): clrs.append("#4CE87B")  # green = size
        else:                                         clrs.append("#4C9BE8")  # blue = other

    bars = ax.barh(fi_top["feature"][::-1], fi_top["importance"][::-1],
                   color=clrs[::-1], edgecolor="white")
    for bar, val in zip(bars, fi_top["importance"][::-1]):
        ax.text(bar.get_width() + 0.0003, bar.get_y() + bar.get_height()/2,
                f"{val:.4f}", va="center", fontsize=8)

    ax.set_xlabel("Feature Importance Score")
    ax.set_title(f"Top {top_n} Feature Importances\n"
                 "🟣 NLP  |  🟠 Location  |  🟢 Size  |  🔵 Other",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3, axis="x"); ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()
    p = os.path.join(OUT, "fig07_feature_importance.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  Feature importance→  {p}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 6 — DEEP LEARNING MODEL
# ═════════════════════════════════════════════════════════════════════════════
header("SECTION 6 · DEEP LEARNING MODEL (Residual MLP)")


def build_and_train_deep_model(X_tr, y_tr, X_val, y_val, X_te, y_te, save_dir=OUT):
    """Build residual MLP, train, evaluate, return metrics dict."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        from tensorflow.keras import layers, callbacks, regularizers
        tf.get_logger().setLevel("ERROR")
        tf.random.set_seed(42)
    except ImportError:
        print("  ⚠  TensorFlow not installed. Skipping deep learning section.")
        print("     Install with:  pip install tensorflow")
        return None, None

    input_dim = X_tr.shape[1]
    l2 = regularizers.l2(1e-4)

    # ── Architecture: Residual MLP ────────────────────────────────────────────
    inputs = keras.Input(shape=(input_dim,), name="features")

    # Block 1
    x = layers.Dense(512, kernel_regularizer=l2)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.30)(x)

    # Residual Block 2
    res = layers.Dense(256, kernel_regularizer=l2)(x)
    x   = layers.Dense(256, kernel_regularizer=l2)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.30)(x)
    x   = layers.Dense(256, kernel_regularizer=l2)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Add()([x, res])
    x   = layers.Activation("relu")(x)

    # Residual Block 3
    res = layers.Dense(128, kernel_regularizer=l2)(x)
    x   = layers.Dense(128, kernel_regularizer=l2)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Activation("relu")(x)
    x   = layers.Dropout(0.21)(x)
    x   = layers.Dense(128, kernel_regularizer=l2)(x)
    x   = layers.BatchNormalization()(x)
    x   = layers.Add()([x, res])
    x   = layers.Activation("relu")(x)

    # Block 4
    x = layers.Dense(64, kernel_regularizer=l2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.15)(x)
    x = layers.Dense(32, activation="relu")(x)
    output = layers.Dense(1, name="price_output")(x)

    model = keras.Model(inputs=inputs, outputs=output, name="RealEstateDNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0),
        loss="huber",
        metrics=["mae"],
    )

    # Print summary
    model.summary()

    # ── Callbacks ─────────────────────────────────────────────────────────────
    ckpt_path = os.path.join(save_dir, "deep_model_best.keras")
    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                restore_best_weights=True, verbose=0),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=8,
                                    min_lr=1e-6, verbose=0),
        callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss",
                                  save_best_only=True, verbose=0),
    ]

    print("\n  Training deep learning model...")
    X_tr_arr  = X_tr.values if isinstance(X_tr, pd.DataFrame) else X_tr
    y_tr_arr  = y_tr.values if isinstance(y_tr, pd.Series)    else y_tr
    X_val_arr = X_val.values if isinstance(X_val, pd.DataFrame) else X_val
    y_val_arr = y_val.values if isinstance(y_val, pd.Series)    else y_val
    X_te_arr  = X_te.values if isinstance(X_te, pd.DataFrame)  else X_te

    history = model.fit(
        X_tr_arr, y_tr_arr,
        validation_data=(X_val_arr, y_val_arr),
        epochs=150, batch_size=128,
        callbacks=cbs, verbose=0,
    )

    # Final epoch stats
    final_epoch = len(history.history["loss"])
    train_loss  = history.history["loss"][-1]
    val_loss    = history.history["val_loss"][-1]
    print(f"  Training stopped at epoch {final_epoch}")
    print(f"  Final train loss: {train_loss:.4f}  |  val loss: {val_loss:.4f}")

    # Evaluate
    preds_log = model.predict(X_te_arr, verbose=0).flatten()
    preds  = np.expm1(preds_log)
    actual = np.expm1(y_te.values if isinstance(y_te, pd.Series) else y_te)

    mae  = mean_absolute_error(actual, preds)
    rmse = np.sqrt(mean_squared_error(actual, preds))
    r2   = r2_score(actual, preds)
    mape = mean_absolute_percentage_error(actual, preds) * 100

    print(f"\n  Deep Learning Result:")
    print(f"    R²   = {r2:.4f}")
    print(f"    MAE  = ₹{mae:.3f} Cr")
    print(f"    RMSE = ₹{rmse:.3f} Cr")
    print(f"    MAPE = {mape:.2f}%")

    # Save
    model.save(os.path.join(save_dir, "deep_model.keras"))
    print(f"  ✔  Deep model saved  →  {os.path.join(save_dir, 'deep_model.keras')}")

    # ── Fig 8: Training history ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Deep Learning Training History", fontsize=14, fontweight="bold")

    axes[0].plot(history.history["loss"],     label="Train Loss", color="#4C9BE8")
    axes[0].plot(history.history["val_loss"], label="Val Loss",   color="#E87B4C", linestyle="--")
    axes[0].set_title("Huber Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
    axes[0].legend(); axes[0].grid(alpha=0.3); axes[0].spines[["top","right"]].set_visible(False)

    axes[1].plot(history.history["mae"],     label="Train MAE", color="#4C9BE8")
    axes[1].plot(history.history["val_mae"], label="Val MAE",   color="#E87B4C", linestyle="--")
    axes[1].set_title("Mean Absolute Error"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("MAE")
    axes[1].legend(); axes[1].grid(alpha=0.3); axes[1].spines[["top","right"]].set_visible(False)

    plt.tight_layout()
    p = os.path.join(OUT, "fig08_dl_training_history.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  DL training chart →  {p}")

    return model, {"r2": r2, "mae": mae, "rmse": rmse, "mape": mape,
                   "predictions": preds, "actual": actual}


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 7 — FINAL SUMMARY DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def plot_summary_dashboard(df, ml_results, best_ml_name, dl_metrics=None):
    header("SECTION 7 · SUMMARY DASHBOARD")

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("Gurgaon Real Estate AI — Final Results Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Model R² comparison (spans 2 cols)
    ax1 = fig.add_subplot(gs[0, :2])
    all_results = dict(ml_results)
    if dl_metrics:
        all_results["Deep Learning"] = dl_metrics
    names = list(all_results.keys())
    r2s   = [r["r2"] for r in all_results.values()]
    best_idx = max(range(len(r2s)), key=lambda i: r2s[i])
    clrs = [("#28a745" if i == best_idx else "#4C9BE8") for i in range(len(names))]
    bars = ax1.bar(range(len(names)), r2s, color=clrs, edgecolor="white", width=0.65)
    ax1.set_ylim([min(r2s) * 0.97, 1.02])
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax1.set_ylabel("R² Score"); ax1.set_title("All Models — R² Comparison (🟢 best)", fontweight="bold")
    ax1.grid(alpha=0.3, axis="y"); ax1.spines[["top","right"]].set_visible(False)
    for b, v in zip(bars, r2s):
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.001,
                 f"{v:.4f}", ha="center", fontsize=8, fontweight="bold")

    # 2. Sector avg prices (top 8)
    ax2 = fig.add_subplot(gs[0, 2])
    top8 = df.groupby("sector")["price_cr"].mean().nlargest(8)
    ax2.barh(top8.index, top8.values,
             color=plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(top8)))[::-1],
             edgecolor="white")
    ax2.set_xlabel("Avg Price (Cr)"); ax2.set_title("Top 8 Sectors", fontweight="bold")
    ax2.grid(alpha=0.3, axis="x"); ax2.spines[["top","right"]].set_visible(False)
    ax2.tick_params(axis="y", labelsize=7)

    # 3. Best-model prediction scatter
    ax3 = fig.add_subplot(gs[1, :2])
    br  = ml_results[best_ml_name]
    lim = [min(br["actual"].min(), br["predictions"].min()) * 0.88,
           max(br["actual"].max(), br["predictions"].max()) * 1.05]
    ax3.scatter(br["actual"], br["predictions"], alpha=0.25, s=12,
                color="#4C9BE8", edgecolors="none")
    ax3.plot(lim, lim, "r--", linewidth=2, label="Perfect Prediction")
    r2_val = br["r2"]
    ax3.text(lim[0] + (lim[1]-lim[0])*0.05, lim[1]*0.92,
             f"R² = {r2_val:.4f}", fontsize=11, fontweight="bold", color="#1a6e3a")
    ax3.set_xlabel("Actual Price (Cr)"); ax3.set_ylabel("Predicted Price (Cr)")
    ax3.set_title(f"{best_ml_name} — Actual vs Predicted", fontweight="bold")
    ax3.legend(); ax3.grid(alpha=0.3); ax3.spines[["top","right"]].set_visible(False)

    # 4. Price distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(df["price_cr"], bins=50, color="#4C9BE8", edgecolor="white", alpha=0.85)
    ax4.axvline(df["price_cr"].mean(), color="red", linestyle="--", linewidth=1.5,
                label=f"Mean ₹{df['price_cr'].mean():.1f}Cr")
    ax4.set_xlabel("Price (Cr)"); ax4.set_ylabel("Count")
    ax4.set_title("Price Distribution", fontweight="bold")
    ax4.legend(fontsize=7); ax4.grid(alpha=0.3, axis="y"); ax4.spines[["top","right"]].set_visible(False)

    # 5. MAE vs MAPE heatmap-style table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    table_data  = [["Model", "R²", "MAE (Cr)", "RMSE (Cr)", "MAPE (%)", "CV R²"]]
    for name, r in sorted(all_results.items(), key=lambda x: x[1]["r2"], reverse=True):
        table_data.append([
            name,
            f"{r['r2']:.4f}",
            f"{r['mae']:.4f}",
            f"{r['rmse']:.4f}",
            f"{r['mape']:.2f}%",
            f"{r.get('cv_r2', 0):.4f}",
        ])
    tbl = ax5.table(cellText=table_data[1:], colLabels=table_data[0],
                    cellLoc="center", loc="center", bbox=[0, -0.1, 1, 1.1])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    # Style header
    for j in range(len(table_data[0])):
        tbl[0, j].set_facecolor("#1a3a5c"); tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Highlight best R² row
    best_row = 1  # sorted descending
    for j in range(len(table_data[0])):
        tbl[best_row, j].set_facecolor("#d4edda")
    ax5.set_title("Model Performance Summary (sorted by R²)", fontweight="bold", pad=20)

    plt.tight_layout()
    p = os.path.join(OUT, "fig09_summary_dashboard.png")
    plt.savefig(p, dpi=150, bbox_inches="tight"); plt.close()
    print(f"  ✔  Summary dashboard →  {p}")


# ═════════════════════════════════════════════════════════════════════════════
# SECTION 8 — INFERENCE DEMO
# ═════════════════════════════════════════════════════════════════════════════
def run_inference_demo(pipeline: FullPreprocessingPipeline, model):
    header("SECTION 8 · INFERENCE DEMO — Live Price Prediction")

    TIER_MAP = {
        "Sector 14":"premium","Sector 29":"premium","Sector 57":"mid","Sector 65":"mid",
        "DLF Phase 1":"luxury","DLF Phase 2":"luxury","DLF Phase 3":"luxury",
        "DLF Phase 4":"luxury","DLF Phase 5":"ultra_luxury","Golf Course Road":"ultra_luxury",
        "Golf Course Extension":"mid","Sohna Road":"affordable","NH-48 Corridor":"affordable",
        "Palam Vihar":"affordable","South City 1":"mid","South City 2":"mid",
        "Nirvana Country":"mid","Malibu Towne":"mid","Sector 82":"affordable",
        "Sector 84":"affordable","Manesar":"budget","Bhondsi":"budget",
        "Badshahpur":"budget","Huda Sectors":"affordable","Cyber City":"ultra_luxury",
    }

    test_properties = [
        {
            "sector":"DLF Phase 5","tier":"ultra_luxury","property_type":"Penthouse",
            "bhk":4,"carpet_area":3200,"super_area":4200,"floor":28,"total_floors":32,
            "age_years":1,"bathrooms":5,"balconies":3,"furnishing":"Fully Furnished",
            "facing":"North-East","builder_grade":"A-Grade","parking":2,"amenity_score":9.5,
            "dist_metro_km":0.7,"dist_school_km":0.5,"dist_hospital_km":1.8,
            "dist_mall_km":1.2,"dist_airport_km":20.0,
            "description":("Stunning 4BHK penthouse in DLF Phase 5. Lavish interiors with "
                           "world-class amenities — spa, concierge, valet parking, rooftop "
                           "lounge, infinity pool. RERA registered, ready to move. "
                           "Breathtaking panoramic views. Gated community with 3-tier security."),
        },
        {
            "sector":"Golf Course Road","tier":"ultra_luxury","property_type":"Apartment",
            "bhk":3,"carpet_area":1800,"super_area":2400,"floor":12,"total_floors":25,
            "age_years":3,"bathrooms":3,"balconies":2,"furnishing":"Fully Furnished",
            "facing":"North","builder_grade":"A-Grade","parking":2,"amenity_score":8.8,
            "dist_metro_km":1.1,"dist_school_km":0.8,"dist_hospital_km":2.2,
            "dist_mall_km":2.0,"dist_airport_km":22.0,
            "description":("Premium 3BHK apartment on Golf Course Road. Elegant interiors "
                           "with swimming pool, gym, clubhouse and 24/7 security. "
                           "Vastu compliant. RERA registered."),
        },
        {
            "sector":"Sohna Road","tier":"affordable","property_type":"Apartment",
            "bhk":2,"carpet_area":900,"super_area":1180,"floor":3,"total_floors":12,
            "age_years":5,"bathrooms":2,"balconies":1,"furnishing":"Semi-Furnished",
            "facing":"East","builder_grade":"B-Grade","parking":1,"amenity_score":5.5,
            "dist_metro_km":3.5,"dist_school_km":1.2,"dist_hospital_km":4.0,
            "dist_mall_km":5.0,"dist_airport_km":32.0,
            "description":("Well-maintained 2BHK apartment on Sohna Road. Semi-furnished "
                           "unit with parking and basic amenities. Close to school. "
                           "Low maintenance charges."),
        },
        {
            "sector":"Manesar","tier":"budget","property_type":"Builder Floor",
            "bhk":1,"carpet_area":480,"super_area":620,"floor":1,"total_floors":4,
            "age_years":10,"bathrooms":1,"balconies":1,"furnishing":"Unfurnished",
            "facing":"South","builder_grade":"Independent","parking":0,"amenity_score":2.5,
            "dist_metro_km":8.0,"dist_school_km":2.0,"dist_hospital_km":5.0,
            "dist_mall_km":9.0,"dist_airport_km":38.0,
            "description":("1BHK builder floor in Manesar. Basic unfurnished unit. "
                           "Old building, ground floor. Near highway."),
        },
    ]

    print(f"\n  {'Property':45s} {'Predicted':>14s} {'Per sqft':>12s}")
    print("  " + "─" * 73)
    for prop in test_properties:
        prop_copy = {**prop, "price_per_sqft": 10000}
        input_df  = pd.DataFrame([prop_copy])
        X         = pipeline.transform(input_df)
        pred      = np.expm1(model.predict(X)[0])
        ppsf      = (pred * 1e7) / prop["carpet_area"]
        label     = (f"{prop['bhk']}BHK {prop['property_type']} — "
                     f"{prop['sector']} ({prop['furnishing'][:4]})")
        print(f"  {label:45s} ₹{pred:>8.2f} Cr   ₹{ppsf:>8,.0f}/sqft")

    print("\n  ✔  Inference demo complete")


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    START = time.time()
    print(f"\n{'═'*70}")
    print("  GURGAON REAL ESTATE AI — END-TO-END DATA SCIENCE PROJECT")
    print(f"  Output directory: ./{OUT}/")
    print(f"{'═'*70}")

    # ── 1. Generate data ──────────────────────────────────────────────────────
    df = generate_dataset(n=5000)

    # ── 2. EDA ────────────────────────────────────────────────────────────────
    run_eda(df)
    plot_nlp_analysis(df)

    # ── 3 & 4. Preprocess ────────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te, pipe = prepare_data(df, test_size=0.20)

    # ── 5. ML models ─────────────────────────────────────────────────────────
    ml_results, best_ml = train_ml_models(X_tr, X_te, y_tr, y_te)

    fi_df = get_feature_importance(best_ml["model"], X_tr.columns.tolist())
    if fi_df is not None:
        plot_feature_importance(fi_df, top_n=25)
        fi_df.to_csv(os.path.join(OUT, "feature_importance.csv"), index=False)

    plot_model_results(ml_results, best_ml["name"])

    # ── 6. Deep Learning ─────────────────────────────────────────────────────
    X_tr_nn, X_val, y_tr_nn, y_val = train_test_split(
        X_tr, y_tr, test_size=0.15, random_state=42
    )
    dl_model, dl_metrics = build_and_train_deep_model(
        X_tr_nn, y_tr_nn, X_val, y_val, X_te, y_te, save_dir=OUT
    )

    # ── 7. Summary dashboard ─────────────────────────────────────────────────
    plot_summary_dashboard(df, ml_results, best_ml["name"], dl_metrics)

    # ── 8. Inference demo ────────────────────────────────────────────────────
    run_inference_demo(pipe, best_ml["model"])

    # ── Save artifacts ────────────────────────────────────────────────────────
    pipe.save(os.path.join(OUT, "preprocessing_pipeline.pkl"))
    with open(os.path.join(OUT, "best_ml_model.pkl"), "wb") as f:
        pickle.dump(best_ml["model"], f)

    # Metrics JSON
    metrics_json = {
        name: {k: v for k, v in res.items() if k not in ("model","predictions","actual")}
        for name, res in ml_results.items()
    }
    if dl_metrics:
        metrics_json["Deep Learning"] = {k: v for k, v in dl_metrics.items()
                                          if k not in ("predictions","actual")}
    with open(os.path.join(OUT, "model_metrics.json"), "w") as f:
        json.dump(metrics_json, f, indent=2)

    # ── Final summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - START
    header("PROJECT COMPLETE")
    print(f"  Total runtime     : {elapsed:.1f} seconds")
    print(f"  Best ML Model     : {best_ml['name']}")
    print(f"    R²              : {best_ml['r2']:.4f}")
    print(f"    MAE             : ₹{best_ml['mae']:.3f} Crores")
    print(f"    MAPE            : {best_ml['mape']:.2f}%")
    if dl_metrics:
        print(f"  Deep Learning DNN :")
        print(f"    R²              : {dl_metrics['r2']:.4f}")
        print(f"    MAE             : ₹{dl_metrics['mae']:.3f} Crores")
    print(f"\n  Output files in ./{OUT}/:")
    for f in sorted(os.listdir(OUT)):
        size = os.path.getsize(os.path.join(OUT, f)) / 1024
        print(f"    {f:45s} {size:>7.1f} KB")
    print(f"\n{DIVIDER}")
    print("  ✔  All done! Charts, models, and CSV saved to ./output/")
    print(DIVIDER)
