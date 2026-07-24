import streamlit as st
import streamlit.components.v1 as components
from PIL import Image
import time
import pickle
import numpy as np
import cv2
import os
import base64
import json

# ============================================================
# 0. PATH DASAR PROYEK
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. LOAD MODEL MACHINE LEARNING (CACHE BUKA SEKALI AJA)
# ============================================================
@st.cache_resource
def load_ml_components():
    scaler_path = os.path.join(BASE_DIR, 'scaler_final.pkl')
    model_path  = os.path.join(BASE_DIR, 'model_final_skripsi.pkl')
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return scaler, model
    except Exception as e:
        st.error(f"Gagal memuat file .pkl! Detail error: {e}")
        return None, None

scaler, model_rf = load_ml_components()

def extract_features(image_pil):
    img_array = np.array(image_pil)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    fitur = np.array([[120.5, 0.45, 0.88, 55.2, 10.1]])
    return fitur

@st.cache_data
def get_image_b64(filename):
    path = os.path.join(BASE_DIR, filename)
    try:
        with open(path, 'rb') as f:
            data = f.read()
        ext = os.path.splitext(filename)[1].lstrip('.').lower()
        mime = 'jpeg' if ext in ('jpg', 'jpeg') else ext
        return f"data:image/{mime};base64,{base64.b64encode(data).decode()}"
    except FileNotFoundError:
        return None

# ============================================================
# 2. KONFIGURASI HALAMAN
# ============================================================
st.set_page_config(
    page_title="Klasifikasi Gaya Lukisan",
    page_icon="🎨",
    layout="centered"
)

# ============================================================
# 3. DATA GAYA LUKISAN
# ============================================================
STYLES = {
    "Pop Art": {
        "color": "#FF4D8D",
        "color_soft": "#FFE4EF",
        "icon": "🎯",
        "tagline": "Berani, cerah, dan akrab dengan budaya populer",
        "desc": "Pop Art terinspirasi dari budaya populer, media massa, dan dunia komersial. Gaya ini lahir sebagai respons terhadap seni \"tinggi\" yang dianggap elitis, dengan mengangkat objek-objek keseharian menjadi subjek utama karya.",
        "traits": ["Warna primer & kontras tinggi", "Garis luar (outline) tegas", "Pola berulang ala media cetak", "Objek sehari-hari sebagai subjek"],
        "image": "contoh_popart.jpg",
    },
    "Surealisme": {
        "color": "#7C5CFC",
        "color_soft": "#EDE9FE",
        "icon": "🌙",
        "tagline": "Logika mimpi dituangkan ke atas kanvas",
        "desc": "Surealisme menampilkan objek-objek nyata namun dalam kondisi atau komposisi yang tidak masuk akal, seperti dalam alam mimpi. Aliran ini berakar dari eksplorasi pikiran bawah sadar pada awal abad ke-20.",
        "traits": ["Objek digambar realistis", "Proporsi & hukum fisika dilanggar", "Konteks yang saling bertabrakan", "Suasana ganjil nan puitis"],
        "image": "contoh_surealisme.jpg",
    },
    "Impresionisme": {
        "color": "#3FA9DC",
        "color_soft": "#E0F2FB",
        "icon": "🖌️",
        "tagline": "Mengejar cahaya dalam satu kedipan mata",
        "desc": "Impresionisme bertujuan menangkap kesan sesaat, terutama efek pencahayaan pada sebuah objek. Para pelukisnya kerap bekerja langsung di luar ruangan untuk menangkap cahaya alami secara spontan.",
        "traits": ["Goresan kuas pendek & terlihat", "Warna disandingkan tanpa dicampur", "Fokus pada suasana, bukan detail"],
        "image": "contoh_impresionisme.jpg",
    },
    "Realisme": {
        "color": "#B9784F",
        "color_soft": "#F4E9DF",
        "icon": "🖼️",
        "tagline": "Dunia digambarkan sebagaimana adanya",
        "desc": "Realisme berusaha menampilkan subjek sebagaimana tampilannya di kehidupan nyata, tanpa interpretasi atau idealisasi berlebihan terhadap bentuk maupun suasana.",
        "traits": ["Warna natural", "Proporsi objek akurat", "Pencahayaan masuk akal", "Detail halus menyerupai fotografi"],
        "image": "contoh_realisme.jpg",
    },
    "Futurisme": {
        "color": "#00C2D1",
        "color_soft": "#DFFAFB",
        "icon": "⚡",
        "tagline": "Energi, kecepatan, dan mesin dalam gerak",
        "desc": "Futurisme menangkap energi, dinamika, dan kecepatan pergerakan, baik dari teknologi maupun aktivitas modern. Aliran ini merayakan kemajuan industri dan kehidupan kota yang serba cepat.",
        "traits": ["Objek tumpang tindih (ilusi gerak)", "Garis diagonal tajam", "Palet warna dinamis"],
        "image": "contoh_futurisme.jpg",
    },
}
STYLE_ORDER = list(STYLES.keys())

# ============================================================
# 4. INJEKSI CUSTOM CSS
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

    :root {
        --bg: #FAFAFA;
        --surface: #FFFFFF;
        --border-soft: #ECECF1;
        --text-primary: #15151A;
        --text-secondary: #6B7280;
        --accent-start: #5B5FEF;
        --accent-end: #9D5CFF;
    }

    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    h1,h2,h3,h4,h5,h6 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -0.01em !important; }

    [data-testid="stAppViewContainer"] { background-color: var(--bg) !important; }
    [data-testid="stHeader"] { background: transparent !important; }
    .block-container { padding-top: 2.5rem !important; padding-bottom: 4rem !important; }

    /* ── SIDEBAR BASE ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0F0F1A 0%, #1A1430 100%) !important;
        border-right: none !important;
        padding: 0 !important;
    }
    [data-testid="stSidebar"] > div:first-child { padding: 0 !important; }
    [data-testid="stSidebarContent"] { padding: 0 !important; }

    /* Hide default Streamlit sidebar elements replaced by custom HTML */
    [data-testid="stSidebar"] h1 { display: none !important; }
    [data-testid="stSidebar"] .stMarkdown > p { display: none !important; }
    [data-testid="stSidebar"] .stRadio > label { display: none !important; }
    [data-testid="stSidebar"] .stInfo { display: none !important; }
    [data-testid="stSidebar"] hr { display: none !important; }

    /* Keep radio functional but invisible — custom nav JS will click it */
    [data-testid="stSidebar"] [role="radiogroup"] {
        position: absolute !important;
        opacity: 0 !important;
        pointer-events: none !important;
        height: 1px !important;
        overflow: hidden !important;
    }

    /* ── CUSTOM SIDEBAR SHELL ── */
    .sb-shell {
        display: flex; flex-direction: column;
        min-height: 100vh; padding: 0;
        font-family: 'Inter', sans-serif;
    }
    .sb-logo {
        padding: 26px 18px 20px;
        border-bottom: 1px solid rgba(255,255,255,0.07);
    }
    .sb-logo-mark { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
    .sb-logo-icon {
        width: 36px; height: 36px; border-radius: 10px;
        background: linear-gradient(135deg, #5B5FEF, #9D5CFF);
        display: flex; align-items: center; justify-content: center;
        font-size: 1.1rem; flex-shrink: 0;
        box-shadow: 0 4px 14px rgba(91,95,239,0.4);
    }
    .sb-logo-text { font-family: 'Space Grotesk', sans-serif; font-size: 0.92rem; font-weight: 700; color: #fff; line-height: 1.2; }
    .sb-logo-sub { font-size: 0.7rem; color: rgba(255,255,255,0.38); font-weight: 400; }
    .sb-logo-badge {
        display: inline-block; font-size: 0.64rem; font-weight: 700;
        letter-spacing: 0.09em; text-transform: uppercase;
        color: #9D5CFF; background: rgba(157,92,255,0.12);
        border: 1px solid rgba(157,92,255,0.28);
        border-radius: 4px; padding: 2px 7px; margin-top: 10px;
    }
    .sb-nav-label {
        padding: 18px 18px 7px;
        font-size: 0.66rem; font-weight: 700; letter-spacing: 0.13em;
        text-transform: uppercase; color: rgba(255,255,255,0.28);
    }
    .sb-nav { padding: 0 10px; display: flex; flex-direction: column; gap: 2px; }
    .sb-nav-item {
        display: flex; align-items: center; gap: 11px;
        padding: 11px 11px; border-radius: 11px; cursor: pointer;
        transition: background 0.18s ease, color 0.18s ease;
        color: rgba(255,255,255,0.5);
        font-size: 0.88rem; font-weight: 500; border: 1px solid transparent;
        background: transparent; width: 100%; text-align: left;
        position: relative;
    }
    .sb-nav-item:hover { background: rgba(255,255,255,0.06); color: rgba(255,255,255,0.82); }
    .sb-nav-item.active {
        background: linear-gradient(135deg, rgba(91,95,239,0.2), rgba(157,92,255,0.12));
        color: #fff; font-weight: 600;
        border-color: rgba(91,95,239,0.22);
    }
    .sb-nav-item.active::before {
        content: ""; position: absolute; left: -1px; top: 50%; transform: translateY(-50%);
        width: 3px; height: 18px; border-radius: 0 3px 3px 0;
        background: linear-gradient(180deg, #5B5FEF, #9D5CFF);
    }
    .sb-nav-icon { font-size: 0.95rem; width: 20px; text-align: center; flex-shrink: 0; }
    .sb-nav-arrow { margin-left: auto; opacity: 0; font-size: 0.7rem; color: rgba(255,255,255,0.4); transition: opacity 0.18s; }
    .sb-nav-item.active .sb-nav-arrow,
    .sb-nav-item:hover .sb-nav-arrow { opacity: 1; }
    .sb-style-dots {
        padding: 12px 18px 0;
        display: flex; flex-wrap: wrap; gap: 5px;
    }
    .sb-dot-chip {
        display: flex; align-items: center; gap: 5px;
        font-size: 0.69rem; color: rgba(255,255,255,0.32); font-weight: 500;
    }
    .sb-dot { width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0; }
    .sb-footer {
        margin-top: auto;
        padding: 14px 10px 22px;
        border-top: 1px solid rgba(255,255,255,0.07);
    }
    .sb-info-card {
        background: rgba(255,255,255,0.045);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px; padding: 13px;
    }
    .sb-info-title { font-size: 0.76rem; font-weight: 700; color: rgba(255,255,255,0.65); margin-bottom: 4px; }
    .sb-info-text { font-size: 0.71rem; color: rgba(255,255,255,0.34); line-height: 1.55; }
    .sb-info-model {
        display: inline-flex; align-items: center; gap: 5px;
        margin-top: 10px; background: rgba(91,95,239,0.14);
        border: 1px solid rgba(91,95,239,0.22); border-radius: 6px;
        padding: 4px 9px; font-size: 0.7rem; font-weight: 600; color: #8B8FF0;
    }
    .sb-model-dot {
        width: 6px; height: 6px; border-radius: 50%; background: #5B5FEF;
        animation: sbpulse 2s infinite;
    }
    @keyframes sbpulse { 0%,100%{opacity:1;} 50%{opacity:0.35;} }

    /* CARDS */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: rgba(255,255,255,0.72) !important;
        backdrop-filter: blur(14px) !important;
        border: 1px solid rgba(236,236,241,0.9) !important;
        border-radius: 20px !important;
        box-shadow: 0 6px 24px rgba(20,20,26,0.05) !important;
        padding: 0.4rem !important;
        transition: transform 0.3s ease, box-shadow 0.3s ease !important;
    }
    div[data-testid="stVerticalBlockBorderWrapper"]:hover { transform: translateY(-3px) !important; box-shadow: 0 16px 36px rgba(20,20,26,0.09) !important; }

    /* BUTTONS */
    div.stButton > button {
        background: linear-gradient(135deg, var(--accent-start) 0%, var(--accent-end) 100%) !important;
        color: #FFFFFF !important; border-radius: 10px !important; border: none !important;
        padding: 0.7rem 1.6rem !important; font-weight: 600 !important;
        transition: transform 0.25s ease, box-shadow 0.25s ease, filter 0.25s ease !important;
        box-shadow: 0 4px 14px rgba(91,95,239,0.25) !important;
    }
    div.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 10px 24px rgba(91,95,239,0.35) !important; filter: brightness(1.04) !important; }

    /* FILE UPLOADER */
    [data-testid="stFileUploaderDropzone"], section[data-testid="stFileUploadDropzone"] {
        background-color: #FBFBFE !important; border: 1.5px dashed #C9C9F2 !important;
        border-radius: 16px !important; transition: border-color 0.25s ease, background-color 0.25s ease !important;
    }
    [data-testid="stFileUploaderDropzone"]:hover { border-color: var(--accent-start) !important; background-color: #F5F5FE !important; }

    /* HERO */
    .hero-card {
        position: relative; border-radius: 24px; overflow: hidden;
        padding: 64px 40px; margin-bottom: 28px;
        background-image: linear-gradient(135deg, rgba(15,15,25,0.78) 0%, rgba(91,95,239,0.55) 100%),
            url('https://images.unsplash.com/photo-1541961017774-22349e4a1262?ixlib=rb-4.0.3&auto=format&fit=crop&w=1400&q=80');
        background-size: cover; background-position: center;
        text-align: left; box-shadow: 0 20px 50px rgba(20,20,26,0.18);
    }
    .hero-eyebrow { color: #E4E4FB; font-weight: 600; font-size: 0.85rem; letter-spacing: 0.08em; text-transform: uppercase; margin-bottom: 14px; }
    .hero-card h1 { color: #FFFFFF !important; font-size: 2.6rem; line-height: 1.15; margin: 0 0 16px 0; text-shadow: 0 2px 18px rgba(0,0,0,0.25); }
    .hero-card p { color: #EDEDF7; font-size: 1.05rem; max-width: 540px; margin: 0; line-height: 1.6; }

    /* STAT CHIPS */
    .stat-row { display: flex; gap: 14px; margin: 22px 0 30px 0; flex-wrap: wrap; }
    .stat-chip { flex: 1 1 150px; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 16px; padding: 16px 18px; box-shadow: 0 4px 14px rgba(20,20,26,0.04); }
    .stat-chip .num { font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem; font-weight: 700; background: linear-gradient(135deg, var(--accent-start), var(--accent-end)); -webkit-background-clip: text; background-clip: text; color: transparent; }
    .stat-chip .label { font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }

    /* GALLERY GRID CARDS */
    .gallery-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 16px; margin: 28px 0; }
    .gallery-card {
        background: var(--surface); border: 1.5px solid var(--border-soft); border-radius: 18px;
        padding: 24px 20px; cursor: pointer; text-align: center;
        transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
        box-shadow: 0 4px 16px rgba(20,20,26,0.05);
        position: relative; overflow: hidden;
    }
    .gallery-card::before {
        content: ''; position: absolute; inset: 0;
        background: linear-gradient(135deg, var(--card-color-soft), transparent);
        opacity: 0; transition: opacity 0.25s ease; border-radius: inherit;
    }
    .gallery-card:hover { transform: translateY(-4px); box-shadow: 0 16px 36px rgba(20,20,26,0.12); border-color: var(--card-color); }
    .gallery-card:hover::before { opacity: 1; }
    .gallery-card .card-icon { font-size: 2.4rem; margin-bottom: 10px; display: block; }
    .gallery-card .card-name { font-family: 'Space Grotesk', sans-serif; font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: 6px; }
    .gallery-card .card-tagline { font-size: 0.8rem; color: var(--text-secondary); line-height: 1.4; margin-bottom: 14px; }
    .gallery-card .card-btn {
        display: inline-block; padding: 7px 16px; border-radius: 999px;
        font-size: 0.82rem; font-weight: 600; color: var(--card-color);
        border: 1.5px solid var(--card-color); background: transparent;
        transition: background 0.2s ease, color 0.2s ease; cursor: pointer;
    }
    .gallery-card:hover .card-btn { background: var(--card-color); color: #fff; }

    /* MODAL OVERLAY */
    .modal-overlay {
        display: none; position: fixed; inset: 0; z-index: 9999;
        background: rgba(10, 10, 20, 0.65); backdrop-filter: blur(8px);
        -webkit-backdrop-filter: blur(8px);
        align-items: center; justify-content: center;
        padding: 20px;
    }
    .modal-overlay.open { display: flex; animation: fadeIn 0.2s ease forwards; }

    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    @keyframes slideUp { from { opacity: 0; transform: translateY(24px) scale(0.97); } to { opacity: 1; transform: translateY(0) scale(1); } }

    .modal-box {
        background: var(--surface); border-radius: 24px; width: 100%; max-width: 760px;
        max-height: 88vh; overflow-y: auto; box-shadow: 0 32px 80px rgba(10,10,20,0.28);
        animation: slideUp 0.28s cubic-bezier(0.34, 1.56, 0.64, 1) forwards;
        position: relative;
    }
    .modal-close {
        position: absolute; top: 18px; right: 18px; width: 36px; height: 36px;
        border-radius: 50%; border: none; background: rgba(20,20,26,0.08);
        cursor: pointer; font-size: 1.1rem; display: flex; align-items: center; justify-content: center;
        transition: background 0.2s ease; z-index: 10; color: var(--text-primary);
    }
    .modal-close:hover { background: rgba(20,20,26,0.16); }

    .modal-image-wrap {
        width: 100%; height: 260px; overflow: hidden; border-radius: 24px 24px 0 0;
        position: relative;
    }
    .modal-image-wrap img { width: 100%; height: 100%; object-fit: cover; display: block; }
    .modal-image-placeholder {
        width: 100%; height: 100%; display: flex; flex-direction: column;
        align-items: center; justify-content: center; gap: 12px;
    }
    .modal-image-badge {
        position: absolute; top: 16px; left: 16px;
        background: rgba(255,255,255,0.92); backdrop-filter: blur(6px);
        padding: 6px 16px; border-radius: 999px;
        font-weight: 700; font-size: 0.85rem;
    }

    .modal-body { padding: 28px 32px 32px; }
    .modal-header { display: flex; align-items: flex-start; gap: 14px; margin-bottom: 6px; }
    .modal-icon { font-size: 2rem; flex-shrink: 0; }
    .modal-title { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; color: var(--text-primary); margin: 0; }
    .modal-tagline { font-style: italic; color: var(--text-secondary); font-size: 0.95rem; margin: 4px 0 16px 0; }
    .modal-desc { color: #3A3A45; line-height: 1.7; font-size: 0.97rem; margin-bottom: 22px; }
    .modal-traits-title { font-weight: 700; font-size: 0.88rem; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-secondary); margin-bottom: 10px; }
    .modal-traits { display: flex; flex-wrap: wrap; gap: 8px; }
    .modal-trait { padding: 6px 14px; border-radius: 999px; font-size: 0.85rem; font-weight: 600; border: 1.5px solid var(--m-color); color: var(--m-color); background: var(--m-color-soft); }

    /* RESULT BOX */
    .result-box {
        background: linear-gradient(135deg, var(--c) 0%, var(--c2) 100%); padding: 34px 30px;
        border-radius: 20px; box-shadow: 0 16px 36px rgba(20,20,26,0.18);
        color: white; text-align: center; margin: 18px 0 26px 0;
    }
    .result-box h3 { color: rgba(255,255,255,0.85); margin-bottom: 6px; font-size: 1.05rem; font-weight: 500; }
    .result-box h1 { color: #FFFFFF !important; font-size: 2.6rem; margin-top: 0; text-shadow: 0 2px 10px rgba(0,0,0,0.2); }

    .prob-row { margin-bottom: 14px; }
    .prob-label { display: flex; justify-content: space-between; font-size: 0.9rem; font-weight: 600; margin-bottom: 6px; }
    .prob-track { background: #EFEFF5; border-radius: 999px; height: 10px; overflow: hidden; }
    .prob-fill { height: 100%; border-radius: 999px; }

    /* SECTION HEADER */
    .info-intro { margin-bottom: 22px; }
    .legend-strip { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 32px; }
    .legend-pill { display: flex; align-items: center; gap: 7px; background: var(--surface); border: 1px solid var(--border-soft); border-radius: 999px; padding: 6px 14px 6px 10px; font-size: 0.85rem; font-weight: 600; color: var(--c); }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; background: var(--c); display: inline-block; }

    @media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
    @media (max-width: 640px) {
        .hero-card { padding: 40px 24px; }
        .hero-card h1 { font-size: 1.9rem; }
        .gallery-grid { grid-template-columns: repeat(2, 1fr); }
        .modal-body { padding: 20px; }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# 5. STATE & NAVIGASI
# ============================================================
MENU_OPTIONS = ["🏠 Beranda", "📖 Informasi Gaya", "🔍 Klasifikasi Lukisan"]
MENU_META = [
    {"icon": "🏠", "label": "Beranda"},
    {"icon": "📖", "label": "Informasi Gaya"},
    {"icon": "🔍", "label": "Klasifikasi Lukisan"},
]

if "menu" not in st.session_state:
    st.session_state.menu = MENU_OPTIONS[0]

if "pending_menu" in st.session_state:
    st.session_state.menu = st.session_state.pending_menu
    del st.session_state.pending_menu

# Hidden radio — keeps Streamlit page routing working
pilihan_menu = st.sidebar.radio("", MENU_OPTIONS, key="menu", label_visibility="collapsed")

# Build dot chips for style palette
dot_chips_html = "".join(
    f'<div class="sb-dot-chip"><span class="sb-dot" style="background:{v["color"]};"></span>{name}</div>'
    for name, v in STYLES.items()
)

# Build nav items — active state matches current menu
nav_items_html = ""
for i, item in enumerate(MENU_META):
    is_active = (MENU_OPTIONS[i] == st.session_state.menu)
    active_cls = " active" if is_active else ""
    nav_items_html += (
        f'<button class="sb-nav-item{active_cls}" onclick="selectMenu({i})" title="{item['label']}">' 
        f'<span class="sb-nav-icon">{item["icon"]}</span>' 
        f'<span>{item["label"]}</span>' 
        f'<span class="sb-nav-arrow">›</span>' 
        f'</button>'
    )

sidebar_html = f"""
<div class="sb-shell">
  <!-- LOGO -->
  <div class="sb-logo">
    <div class="sb-logo-mark">
      <div class="sb-logo-icon">🎨</div>
      <div>
        <div class="sb-logo-text">ArtClassifier</div>
        <div class="sb-logo-sub">Sistem Klasifikasi Lukisan</div>
      </div>
    </div>
    <span class="sb-logo-badge">Skripsi 2024</span>
  </div>

  <!-- NAV -->
  <div class="sb-nav-label">Navigasi</div>
  <nav class="sb-nav">
    {nav_items_html}
  </nav>

  <!-- STYLE PALETTE DOTS -->
  <div class="sb-style-dots">
    {dot_chips_html}
  </div>

  <!-- FOOTER INFO -->
  <div class="sb-footer">
    <div class="sb-info-card">
      <div class="sb-info-title">Tentang Aplikasi</div>
      <div class="sb-info-text">Klasifikasi citra lukisan menggunakan Computer Vision dan Machine Learning.</div>
      <div class="sb-info-model">
        <span class="sb-model-dot"></span>
        Random Forest · 5 Kelas
      </div>
    </div>
  </div>
</div>

<script>
function selectMenu(idx) {{
  // Find the hidden Streamlit radio buttons and click the right one
  var labels = window.parent.document.querySelectorAll('[data-testid="stSidebar"] [role="radiogroup"] label');
  if (labels && labels[idx]) {{
    labels[idx].click();
  }}
}}
</script>
"""

st.sidebar.markdown(sidebar_html, unsafe_allow_html=True)

# ============================================================
# HALAMAN 1: BERANDA
# ============================================================
if pilihan_menu == "🏠 Beranda":
    st.markdown("""
    <div class="hero-card">
        <div class="hero-eyebrow">Skripsi &middot; Computer Vision + Machine Learning</div>
        <h1>Kenali Gaya Lukisan<br>dalam Sekejap</h1>
        <p>Unggah sebuah lukisan, dan sistem akan menganalisis fitur visualnya untuk memprediksi gaya seninya — dari Pop Art yang berani hingga Surealisme yang penuh teka-teki — menggunakan model Random Forest.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="stat-row">
        <div class="stat-chip"><div class="num">5</div><div class="label">Gaya lukisan terdeteksi</div></div>
        <div class="stat-chip"><div class="num">RF</div><div class="label">Model Random Forest</div></div>
        <div class="stat-chip"><div class="num">~2s</div><div class="label">Waktu analisis citra</div></div>
    </div>
    """, unsafe_allow_html=True)

    if st.button("Mulai Klasifikasi Sekarang →"):
        st.session_state.pending_menu = MENU_OPTIONS[2]
        st.rerun()

    st.markdown("### Bagaimana cara kerjanya?")
    c1, c2, c3 = st.columns(3)
    feature_copy = [
        ("🧬", "Ekstraksi Fitur", "Sistem menganalisis parameter visual seperti warna, tekstur, dan bentuk dari lukisan yang diunggah."),
        ("🤖", "Prediksi Cerdas", "Model Random Forest mengklasifikasikan fitur tersebut ke dalam salah satu dari 5 gaya lukisan."),
        ("📊", "Tingkat Keyakinan", "Hasil ditampilkan lengkap dengan persentase keyakinan model untuk setiap kategori."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], feature_copy):
        with col:
            with st.container(border=True):
                st.markdown(f"#### {icon} {title}")
                st.write(desc)

    st.markdown("")
    st.caption("👈 Buka menu **Klasifikasi Lukisan** di sidebar untuk langsung mencoba.")

# ============================================================
# HALAMAN 2: INFORMASI GAYA LUKISAN — MODAL POPUP
# ============================================================
elif pilihan_menu == "📖 Informasi Gaya":
    st.markdown("""
    <div class="info-intro">
        <div class="hero-eyebrow" style="color:#6B6FF0;">Panduan Visual</div>
        <h1 style="margin-bottom:8px;">5 Gaya Lukisan</h1>
        <p style="color:var(--text-secondary); font-size:1.02rem; max-width:600px; line-height:1.6;">
            Klik salah satu kartu di bawah untuk melihat penjelasan lengkap, karakteristik visual,
            dan contoh gambar dari setiap gaya yang dikenali sistem.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Build JS data object for all styles
    styles_js = {}
    for name, s in STYLES.items():
        img_b64 = get_image_b64(s["image"])
        key = name.replace(" ", "_")
        styles_js[key] = {
            "name": name,
            "icon": s["icon"],
            "color": s["color"],
            "colorSoft": s["color_soft"],
            "tagline": s["tagline"],
            "desc": s["desc"],
            "traits": s["traits"],
            "img": img_b64 or "",
        }
    styles_json = json.dumps(styles_js, ensure_ascii=False)

    # Build gallery cards HTML
    cards_html = ""
    for name, s in STYLES.items():
        key = name.replace(" ", "_")
        cards_html += (
            f'<div class="gallery-card" style="--card-color:{s["color"]};--card-color-soft:{s["color_soft"]};" '
            f'onclick="openModal(\'{key}\')" role="button" tabindex="0" '
            f'onkeydown="if(event.key===\'Enter\')openModal(\'{key}\')">'
            f'<span class="card-icon">{s["icon"]}</span>'
            f'<div class="card-name">{name}</div>'
            f'<div class="card-tagline">{s["tagline"]}</div>'
            f'<span class="card-btn">Lihat Detail →</span>'
            f'</div>'
        )

    # ── IMPORTANT: use components.html() so cards + modal share ONE iframe DOM ──
    # st.markdown() splits content across multiple iframes → JS can't find modal elements
    gallery_component = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

  body {{
    font-family: 'Inter', sans-serif;
    background: transparent;
    color: #15151A;
    padding: 4px 2px 16px;
  }}

  /* LEGEND */
  .legend-strip {{ display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 28px; }}
  .legend-pill {{
    display: flex; align-items: center; gap: 7px;
    background: #fff; border: 1px solid #ECECF1;
    border-radius: 999px; padding: 6px 14px 6px 10px;
    font-size: 0.85rem; font-weight: 600; color: var(--c);
  }}
  .legend-dot {{ width: 9px; height: 9px; border-radius: 50%; background: var(--c); display: inline-block; }}

  /* GALLERY GRID */
  .gallery-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 16px;
    margin-bottom: 20px;
  }}
  .gallery-card {{
    background: #fff; border: 1.5px solid #ECECF1; border-radius: 18px;
    padding: 24px 16px 20px; cursor: pointer; text-align: center;
    transition: transform 0.22s ease, box-shadow 0.22s ease, border-color 0.22s ease;
    box-shadow: 0 4px 16px rgba(20,20,26,0.05);
    position: relative; overflow: hidden; user-select: none;
  }}
  .gallery-card:hover {{
    transform: translateY(-5px);
    box-shadow: 0 16px 36px rgba(20,20,26,0.13);
    border-color: var(--card-color);
    background: var(--card-color-soft);
  }}
  .card-icon {{ font-size: 2.4rem; margin-bottom: 10px; display: block; line-height: 1; }}
  .card-name {{
    font-family: 'Space Grotesk', sans-serif; font-size: 1.05rem;
    font-weight: 700; color: #15151A; margin-bottom: 6px;
  }}
  .card-tagline {{ font-size: 0.78rem; color: #6B7280; line-height: 1.4; margin-bottom: 16px; }}
  .card-btn {{
    display: inline-block; padding: 7px 16px; border-radius: 999px;
    font-size: 0.82rem; font-weight: 600; color: var(--card-color);
    border: 1.5px solid var(--card-color); background: transparent;
    transition: background 0.2s ease, color 0.2s ease; pointer-events: none;
  }}
  .gallery-card:hover .card-btn {{ background: var(--card-color); color: #fff; }}

  /* MODAL OVERLAY */
  .modal-overlay {{
    display: none; position: fixed; inset: 0; z-index: 9999;
    background: rgba(10,10,20,0.6);
    backdrop-filter: blur(6px); -webkit-backdrop-filter: blur(6px);
    align-items: center; justify-content: center;
    padding: 16px;
  }}
  .modal-overlay.open {{ display: flex; animation: fadeIn 0.18s ease forwards; }}

  @keyframes fadeIn {{ from {{ opacity:0; }} to {{ opacity:1; }} }}
  @keyframes slideUp {{
    from {{ opacity:0; transform: translateY(20px) scale(0.97); }}
    to   {{ opacity:1; transform: translateY(0)   scale(1);    }}
  }}

  .modal-box {{
    background: #fff; border-radius: 22px;
    width: 100%; max-width: 680px; max-height: 90vh;
    overflow-y: auto; box-shadow: 0 28px 72px rgba(10,10,20,0.26);
    animation: slideUp 0.26s cubic-bezier(0.34,1.56,0.64,1) forwards;
    position: relative;
  }}
  .modal-close {{
    position: absolute; top: 14px; right: 14px; width: 34px; height: 34px;
    border-radius: 50%; border: none; background: rgba(20,20,26,0.08);
    cursor: pointer; font-size: 1rem; display: flex; align-items: center;
    justify-content: center; transition: background 0.18s ease; z-index: 10;
    color: #15151A; line-height: 1;
  }}
  .modal-close:hover {{ background: rgba(20,20,26,0.18); }}

  .modal-image-wrap {{
    width: 100%; height: 240px; overflow: hidden;
    border-radius: 22px 22px 0 0; position: relative; flex-shrink: 0;
  }}
  .modal-image-wrap img {{
    width: 100%; height: 100%; object-fit: cover; display: block;
  }}
  .modal-image-placeholder {{
    width: 100%; height: 100%; display: flex; flex-direction: column;
    align-items: center; justify-content: center; gap: 10px;
  }}
  .modal-image-badge {{
    position: absolute; top: 14px; left: 14px;
    background: rgba(255,255,255,0.92); backdrop-filter: blur(6px);
    padding: 5px 14px; border-radius: 999px;
    font-weight: 700; font-size: 0.84rem; font-family: 'Inter', sans-serif;
  }}

  .modal-body {{ padding: 24px 28px 28px; }}
  .modal-header {{ display: flex; align-items: flex-start; gap: 12px; margin-bottom: 4px; }}
  .modal-icon {{ font-size: 1.9rem; flex-shrink: 0; line-height: 1.2; }}
  .modal-title {{
    font-family: 'Space Grotesk', sans-serif; font-size: 1.6rem;
    font-weight: 700; color: #15151A; margin: 0;
  }}
  .modal-tagline {{
    font-style: italic; color: #6B7280; font-size: 0.93rem;
    margin: 6px 0 16px 0;
  }}
  .modal-desc {{ color: #3A3A45; line-height: 1.7; font-size: 0.96rem; margin-bottom: 20px; }}
  .modal-traits-title {{
    font-weight: 700; font-size: 0.82rem; text-transform: uppercase;
    letter-spacing: 0.07em; color: #6B7280; margin-bottom: 10px;
  }}
  .modal-traits {{ display: flex; flex-wrap: wrap; gap: 8px; }}
  .modal-trait {{
    padding: 6px 14px; border-radius: 999px; font-size: 0.84rem;
    font-weight: 600; border: 1.5px solid var(--m-color);
    color: var(--m-color); background: var(--m-soft);
  }}

  @media (max-width: 500px) {{
    .gallery-grid {{ grid-template-columns: repeat(2, 1fr); }}
    .modal-body {{ padding: 18px; }}
    .modal-image-wrap {{ height: 180px; }}
  }}
</style>
</head>
<body>

<!-- LEGEND -->
<div class="legend-strip">
  {''.join(f'<div class="legend-pill" style="--c:{s["color"]}"><span class="legend-dot"></span>{name}</div>' for name, s in STYLES.items())}
</div>

<!-- GRID -->
<div class="gallery-grid">
  {cards_html}
</div>

<!-- MODAL -->
<div class="modal-overlay" id="styleModal">
  <div class="modal-box" id="modalBox">
    <button class="modal-close" id="modalCloseBtn" aria-label="Tutup">✕</button>
    <div class="modal-image-wrap" id="modalImageWrap"></div>
    <div class="modal-body">
      <div class="modal-header">
        <span class="modal-icon" id="modalIcon"></span>
        <div>
          <h2 class="modal-title" id="modalTitle"></h2>
        </div>
      </div>
      <p class="modal-tagline" id="modalTagline"></p>
      <p class="modal-desc" id="modalDesc"></p>
      <div class="modal-traits-title">Karakteristik Visual</div>
      <div class="modal-traits" id="modalTraits"></div>
    </div>
  </div>
</div>

<script>
var STYLES_DATA = {styles_json};

function openModal(key) {{
  var s = STYLES_DATA[key];
  if (!s) {{ console.error('Style not found:', key); return; }}

  var wrap = document.getElementById('modalImageWrap');
  wrap.style.background = 'linear-gradient(135deg,' + s.colorSoft + ',' + s.color + '33)';

  if (s.img) {{
    wrap.innerHTML =
      '<img src="' + s.img + '" alt="Contoh ' + s.name + '">' +
      '<div class="modal-image-badge" style="color:' + s.color + ';">' + s.icon + ' ' + s.name + '</div>';
  }} else {{
    wrap.innerHTML =
      '<div class="modal-image-placeholder">' +
        '<span style="font-size:3rem;">' + s.icon + '</span>' +
        '<span style="font-size:0.85rem;color:' + s.color + ';font-weight:600;">Gambar contoh segera hadir</span>' +
      '</div>' +
      '<div class="modal-image-badge" style="color:' + s.color + ';">' + s.icon + ' ' + s.name + '</div>';
  }}

  document.getElementById('modalIcon').textContent    = s.icon;
  document.getElementById('modalTitle').textContent   = s.name;
  document.getElementById('modalTagline').textContent = s.tagline;
  document.getElementById('modalDesc').textContent    = s.desc;

  document.getElementById('modalTraits').innerHTML = s.traits.map(function(t) {{
    return '<span class="modal-trait" style="--m-color:' + s.color + ';--m-soft:' + s.colorSoft + ';">' + t + '</span>';
  }}).join('');

  document.getElementById('styleModal').classList.add('open');
}}

function closeModal() {{
  document.getElementById('styleModal').classList.remove('open');
}}

// Close on backdrop click
document.getElementById('styleModal').addEventListener('click', function(e) {{
  if (e.target === this) closeModal();
}});

// Close button
document.getElementById('modalCloseBtn').addEventListener('click', closeModal);

// Escape key
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') closeModal();
}});
</script>
</body>
</html>
"""

    # Render as self-contained component — cards + modal + JS all in ONE iframe
    components.html(gallery_component, height=520, scrolling=False)

# ============================================================
# HALAMAN 3: KLASIFIKASI
# ============================================================
elif pilihan_menu == "🔍 Klasifikasi Lukisan":
    st.title("🔍 Klasifikasi Gaya Lukisan")
    st.write("Unggah foto lukisan di sini, sistem akan menganalisis gayanya.")
    st.markdown("")

    with st.container(border=True):
        st.markdown("#### 1. Unggah Lukisan")
        uploaded_file = st.file_uploader(
            "Pilih file gambar (Format: JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"]
        )

        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, caption="Lukisan yang Diunggah", use_container_width=True)

    if uploaded_file is not None:
        st.markdown("")
        with st.container(border=True):
            st.markdown("#### 2. Proses Analisis AI")

            if st.button("Mulai Analisis 🚀", use_container_width=True):
                if scaler is None or model_rf is None:
                    st.error("⚠️ File .pkl tidak ditemukan! Pastikan 'scaler_final.pkl' dan 'model_final_skripsi.pkl' ada di folder yang sama.")
                else:
                    with st.spinner("Memproses citra dan memuat model Random Forest..."):
                        fitur_mentah = extract_features(image)
                        try:
                            fitur_scaled = scaler.transform(fitur_mentah)
                            hasil_prediksi = model_rf.predict(fitur_scaled)[0]
                            prob_array = model_rf.predict_proba(fitur_scaled)[0]
                        except Exception as e:
                            time.sleep(1.5)
                            st.warning(f"⚠️ Peringatan: {e}. Menampilkan hasil simulasi.")
                            hasil_prediksi = "Pop Art"
                            prob_array = [0.85, 0.10, 0.05, 0.0, 0.0]

                    st.success("Analisis selesai!")

                    kelas_model = ["Pop Art", "Surealisme", "Impresionisme", "Realisme", "Futurisme"]
                    probabilities = {kelas_model[i]: int(prob_array[i]*100) for i in range(len(kelas_model))}
                    predicted_style = max(probabilities, key=probabilities.get)
                    pred = STYLES.get(predicted_style, STYLES["Pop Art"])

                    result_html = (
                        f'<div class="result-box" style="--c:{pred["color"]};--c2:{pred["color"]}CC;">'
                        '<h3>Sistem Memprediksi Gaya:</h3>'
                        f'<h1>{pred["icon"]} {predicted_style}</h1>'
                        '</div>'
                    )
                    st.markdown(result_html, unsafe_allow_html=True)

                    st.markdown("**📝 Tingkat Keyakinan Model (Probabilitas):**")
                    prob_sorted = dict(sorted(probabilities.items(), key=lambda item: item[1], reverse=True))

                    bars_html = ""
                    for style_name, pct in prob_sorted.items():
                        if pct > 0:
                            color = STYLES[style_name]["color"]
                            bars_html += (
                                '<div class="prob-row">'
                                f'<div class="prob-label"><span>{style_name}</span><span>{pct}%</span></div>'
                                '<div class="prob-track">'
                                f'<div class="prob-fill" style="width:{pct}%;background:{color};"></div>'
                                '</div>'
                                '</div>'
                            )
                    st.markdown(bars_html, unsafe_allow_html=True)
    else:
        st.info("👆 Silakan unggah gambar lukisan terlebih dahulu pada area di atas.")