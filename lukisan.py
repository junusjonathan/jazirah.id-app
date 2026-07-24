import streamlit as st
import streamlit.components.v1 as components
import joblib # Tambahin di jejeran import paling atas
from PIL import Image
import time
import pickle
import numpy as np
import cv2
import os
import base64
import json
from skimage.feature import graycomatrix, graycoprops, local_binary_pattern, hog

# ============================================================
# 0. PATH DASAR PROYEK
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ============================================================
# 1. LOAD MODEL MACHINE LEARNING
# ============================================================
@st.cache_resource
def load_ml_components():
    scaler_path = os.path.join(BASE_DIR, 'scaler_final.pkl')
    model_path  = os.path.join(BASE_DIR, 'model_final_skripsi.pkl')
    try:
        # Kita buka modelnya pakai joblib
        scaler = joblib.load(scaler_path)
        model = joblib.load(model_path)
        return scaler, model
    except Exception as e:
        st.error(f"Gagal memuat file .pkl! Detail error: {e}")
        return None, None

# scaler, model_rf = load_ml_components()

# ============================================================
# TUGAS LU SELANJUTNYA: GANTI ISI FUNGSI INI DENGAN KODE ASLI
# ============================================================
def extract_features(image_pil):
    # 1. Konversi gambar dari Streamlit (PIL) ke format array RGB murni, lalu ubah ke format OpenCV (BGR)
    img_array = np.array(image_pil.convert('RGB')) 
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    # 2. Resize standar persis seperti saat lu ngelatih model (128x128)
    img_resized = cv2.resize(img_cv, (128, 128))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    # --- A. TEKSTUR GLOBAL (GLCM) ---
    glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    entropy = -np.sum(glcm * np.log2(glcm + 1e-10))
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    correlation = graycoprops(glcm, 'correlation')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    
    # --- B. TEKSTUR MIKRO (LBP) ---
    radius = 1
    n_points = 8 * radius
    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
    lbp_mean = np.mean(lbp)
    lbp_std = np.std(lbp)
    
    # --- C. STRUKTUR & BENTUK (HOG) --- 
    fd = hog(img_resized, orientations=8, pixels_per_cell=(16, 16),
             cells_per_block=(1, 1), visualize=False, channel_axis=-1)
    hog_mean = np.mean(fd)
    hog_std = np.std(fd)
    
    # --- D. WARNA & RUANG (HSV) ---
    hsv = cv2.cvtColor(img_resized, cv2.COLOR_BGR2HSV)
    mean_v = np.mean(hsv[:,:,2]) 
    std_h = np.std(hsv[:,:,0])
    
    # --- E. STATISTIK RGB ---
    std_rgb = np.std(img_resized, axis=(0, 1))
    std_g = std_rgb[1] 
    std_r = std_rgb[2] 
    
    # --- F. STRUKTUR GARIS (Edge) & PERSEPSI WARNA ---
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.mean(edges) / 255.0 

    R, G, B = img_resized[:,:,2], img_resized[:,:,1], img_resized[:,:,0]
    rg = np.absolute(R - G)
    yb = np.absolute(0.5 * (R + G) - B)
    colorfulness = np.sqrt(np.std(rg)**2 + np.std(yb)**2) + (0.3 * np.mean(rg + yb))

    # Gabungkan 15 fitur dan DIBUNGKUS SIKU GANDA [[...]] agar model Random Forest lu mau ngebacanya
    features = [[
        float(contrast), float(entropy), float(homogeneity), float(correlation), float(energy),
        float(lbp_mean), float(lbp_std), float(hog_mean), float(hog_std), float(mean_v),
        float(std_h), float(std_g), float(std_r), float(edge_density), float(colorfulness)
    ]]
    
    return np.array(features)

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
    page_title="Jazirah.id",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 3. DATA GAYA LUKISAN
# ============================================================
STYLES = {
    "Pop Art": {
        "color": "#E8445A",
        "color_soft": "rgba(232,68,90,0.15)",
        "color_dim": "rgba(232,68,90,0.08)",
        "icon": "🎯",
        "tagline": "Berani, cerah, akrab budaya populer",
        "desc": "Berdasarkan kajian Banindro (2019), gaya lukisan Pop Art merupakan gerakan representasi realitas sosial yang mendemokratisasikan ranah visual dengan mengangkat citra barang konsumsi sehari-hari, kemasan produk komersial, dan budaya massa ke dalam status seni murni (high art). Melalui pendekatan yang mendekonstruksi makna semiotika periklanan, gaya ini secara radikal mendobrak elitisme institusi seni tradisional yang sebelumnya sangat eksklusif. Hasilnya, batasan kelas sosial dalam apresiasi estetika menjadi runtuh, sehingga karya seni tidak lagi sekadar menjadi dominasi kalangan elite, melainkan berubah menjadi bentuk visual yang inklusif dan dapat dinikmati secara luas oleh seluruh lapisan masyarakat.",
        "traits": ["Warna Cerah, Kontras, dan Blok Warna (Flat Color)", "Penggunaan Garis Tepi Hitam yang Tegas (Bold Outline)", "Teknik Repetisi dan Pola Titik (Ben-Day Dots)", "Adopsi Idiom Media Massa dan Komik", "Penggunaan Ikon Konsumerisme dan Budaya Populer"],
        "image": "pop art.jpg",
    },
    "Surealisme": {
        "color": "#9B6DFF",
        "color_soft": "rgba(155,109,255,0.15)",
        "color_dim": "rgba(155,109,255,0.08)",
        "icon": "🌙",
        "tagline": "Logika mimpi di atas kanvas",
        "desc": "Berdasarkan kajian Anna Sungkar (2021) dalam Jurnal Dekonstruksi (Surealisme Dalam Seni Lukis Indonesia) surealisme merupakan aliran seni rupa yang berakar pada eksplorasi alam bawah sadar dan dunia mimpi, yang memungkinkan seniman untuk membebaskan diri dari pikiran yang terkontrol, logika rasional, maupun prasangka moral. Sangat dipengaruhi oleh teori psikoanalisis Sigmund Freud, gaya lukisan ini kerap menggabungkan objek-objek nyata sehari-hari secara tidak koheren, ganjil, dan penuh teka-teki enigmatik. Dengan mengesampingkan aturan realitas tradisional, surealisme menampilkan simbol-simbol fantastis yang diambil langsung dari imajinasi liar dan ketidaksadaran manusia.",
        "traits": ["Suasana Seperti di Alam Mimpi", "Gabungan Benda yang Tidak Nyambung", "Bentuk Benda yang Berubah Aneh (Distorsi)", "Tekstur Permukaan yang Sangat Halus (Fotorealistik)", "Penggunaan Warna yang Tidak Logis dan Misterius", "Garis dan Bentuk yang Terdistorsi (Meleleh/Berubah)" ],
        "image": "surealis.jpg",
    },
    "Impresionisme": {
        "color": "#3ABFCF",
        "color_soft": "rgba(58,191,207,0.15)",
        "color_dim": "rgba(58,191,207,0.08)",
        "icon": "🖌️",
        "tagline": "Mengejar cahaya dalam satu kedipan",
        "desc": "Dalam tinjauan jurnal Seni Lukis Kontemporer Karya Andie Aradhea dalam Pendekatan Kritik Seni, Impresionisme dipahami sebagai sebuah pergeseran fundamental dalam sejarah seni rupa yang membebaskan seniman dari tuntutan untuk memproduksi karya yang menyerupai realitas secara objektif atau fotografis. Aliran ini menekankan pada subjektivitas penglihatan, di mana fokus utamanya adalah menangkap momen singkat atau impresi dari suatu objek yang diterpa cahaya matahari secara dinamis. Seniman Impresionis tidak lagi berusaha menggambarkan objek secara statis dan mendetail, melainkan lebih mengutamakan perekaman suasana atmosferik yang terus berubah.",
        "traits": ["Sapuan Kuas yang Kasar dan Terlihat", "Cahaya sebagai Pemeran Utama ", "Warna yang Bergetar (Pencampuran Optik)", "Bayangan yang Tidak Hitam", "Tanpa Garis Kontur (Outline)"],
        "image": "impresionis.png",
    },
    "Realisme": {
        "color": "#C8A96E",
        "color_soft": "rgba(200,169,110,0.15)",
        "color_dim": "rgba(200,169,110,0.08)",
        "icon": "🖼️",
        "tagline": "Dunia digambarkan sebagaimana adanya",
        "desc": "Menurut Wahdah dan Hasbi (2024), Realisme didefinisikan sebagai aliran seni lukis yang berkomitmen pada upaya representasi objek secara jujur dan objektif sesuai dengan kenyataan yang tampak di alam. Penjelasan ini selaras dengan pandangan sejarawan seni Linda Nochlin, yang menyatakan bahwa Realisme adalah sebuah komitmen terhadap kebenaran objektif terhadap subjek yang dipilih, tanpa ada upaya untuk mengidealkan atau memperindah realitas. Seniman Realis bekerja sebagai pengamat teliti yang memprioritaskan akurasi tekstur, anatomi, serta hukum pencahayaan alami untuk menciptakan ilusi kenyataan yang murni di atas bidang datar.",
        "traits": ["Ketepatan Proporsi dan Anatomi", "Penerapan Gelap-Terang (Chiaroscuro) yang Konsisten", "Tekstur yang Mendetail dan Realistis", "Detail Keseimbangan Komposisi yang Objektif", "Ketelitian dalam Penggunaan Warna yang Natural"],
        "image": "realism.jpeg",
    },
    "Futurisme": {
        "color": "#4ADE80",
        "color_soft": "rgba(74,222,128,0.15)",
        "color_dim": "rgba(74,222,128,0.08)",
        "icon": "⚡",
        "tagline": "Energi, kecepatan, dan mesin dalam gerak",
        "desc": "Oleh Kovaleva (2020) menjelaskan bahwa Futurisme bukan sekadar gaya seni, melainkan sebuah gerakan kultural radikal yang muncul sebagai bentuk pemujaan terhadap modernitas, kecepatan, teknologi, dan masa depan. Gerakan ini secara agresif menolak tradisi masa lalu (seperti museum dan seni klasik) yang dianggap sudah tidak relevan. Konsep utama Futurisme adalah upaya untuk merepresentasikan dinamisme universal atau energi pergerakan yang konstan. Seniman Futurisme berupaya menghadirkan sensasi kecepatan mesin, kebisingan kota industri, dan kekuatan teknologi yang mengubah cara manusia memandang ruang dan waktu. Bagi mereka, seni harus mampu menangkap sensasi gerak yang dirasakan oleh manusia modern yang hidup di tengah perubahan zaman yang cepat.",
        "traits": [ "Garis-Garis Gaya (Force Lines) tajam dan diagonal", "Warna yang Kontras dan Agresif", "Objek yang menyatu dengan sekitarnya", "Warna yang Ramai dan Bising"],
        "image": "futurism.png",
    },
}
STYLE_ORDER = list(STYLES.keys())

# ============================================================
# 4. NAVIGASI STATE
# ============================================================
MENU_META = [
    {"icon": "⬡", "label": "Beranda",             "slug": "beranda"},
    {"icon": "◈", "label": "Gaya Lukisan",        "slug": "informasi"},
    {"icon": "◎", "label": "Analisis Lukisan",    "slug": "klasifikasi"},
    {"icon": "❖", "label": "Tentang Aplikasi",    "slug": "tentang"},
    {"icon": "📖", "label": "Panduan Penggunaan", "slug": "panduan"}, # 👈 TAMBAHAN MENU PANDUAN
]
_slug = st.query_params.get("page", "beranda")
_slug_to_idx = {m["slug"]: i for i, m in enumerate(MENU_META)}
_active_idx = _slug_to_idx.get(_slug, 0)
pilihan_menu = MENU_META[_active_idx]["slug"]

# ============================================================
# 5. CSS GLOBAL
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');

/* ── RESET & VARS ─────────────────────────────────────── */
:root {
    --bg-light:      #d3d0d0; 
    --brand-navy:    #2b3d63; 
    --brand-dark:    #1e2b45; 
    --brand-light:   #3a5285; 
    
    --border-light:  rgba(43, 61, 99, 0.15); 
    --border-dark:   rgba(221, 221, 216, 0.15); 
    
    --gold:          #C8A96E; 
    --gold-dim:      rgba(200,169,110,0.18);
    
    --text-on-light: #1e2b45; 
    --text-on-light-muted: rgba(30, 43, 69, 0.65);
    
    --text-on-dark:  #ddddd8; 
    --text-on-dark-muted: rgba(221, 221, 216, 0.65);
    --text-on-dark-lo: rgba(221, 221, 216, 0.4);

    --radius-sm:  8px;
    --radius-md:  14px;
    --radius-lg:  20px;
    --radius-xl:  28px;
}

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
    color: var(--text-on-light) !important;
}

/* ── STREAMLIT WRAPPERS ───────────────────────────────── */
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stBottom"],
[data-testid="stVerticalBlock"],
.main,
.main .block-container {
    background: transparent !important;
}

[data-testid="stAppViewContainer"] {
    background: var(--bg-light) !important;
    background-image: radial-gradient(circle at top right, rgba(200,169,110,0.03), transparent 40%) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
}

[data-testid="stHeader"] {
    background: transparent !important;
    border-bottom: none !important;
}

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 5rem !important;
    max-width: 860px !important;
}

/* ── SIDEBAR ──────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--brand-navy) 0%, var(--brand-dark) 100%) !important;
    border-right: 1px solid var(--border-dark) !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.05) !important;
    padding: 0 !important;
    width: 260px !important;
    z-index: 999999 !important;
}
[data-testid="stSidebar"] > div:first-child,
[data-testid="stSidebarContent"] {
    padding: 0 !important;
    background: transparent !important;
}

/* Hide default Streamlit sidebar noise */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] .stMarkdown > p,
[data-testid="stSidebar"] .stRadio > label,
[data-testid="stSidebar"] .stInfo,
[data-testid="stSidebar"] hr { display: none !important; }
[data-testid="stSidebar"] [role="radiogroup"] {
    position: absolute !important; opacity: 0 !important;
    pointer-events: none !important; height: 1px !important; overflow: hidden !important;
}

/* ── SIDEBAR CUSTOM SHELL ─────────────────────────────── */
.sb {
    display: flex; flex-direction: column;
    min-height: 100vh; height: 100%;
    padding: 0;
}

/* Perbaikan penempatan struktur header agar rata tengah sempurna */
.sb-brand {
    padding: 32px 18px 24px;
    border-bottom: 1px solid var(--border-dark);
    display: flex;
    flex-direction: column;
    align-items: center;
    text-align: center;
}
.sb-brand-mark {
    display: flex; 
    flex-direction: column; 
    align-items: center; 
    gap: 14px;
    margin-bottom: 14px;
    width: 100%;
}
.sb-brand-gem {
    width: 52px; height: 52px;
    border-radius: 12px;
    border: 1px solid var(--gold-dim);
    background: rgba(200,169,110,0.1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.2rem; flex-shrink: 0;
    color: var(--gold);
}
.sb-brand-name {
    font-family: 'DM Serif Display', serif;
    font-size: 1.35rem; color: var(--text-on-dark);
    letter-spacing: 0.01em; line-height: 1.2;
    text-align: center;
}
.sb-brand-sub {
    font-size: 0.72rem; color: var(--text-on-dark-muted);
    font-weight: 400; margin-top: 4px;
    letter-spacing: 0.03em; line-height: 1.3;
    text-align: center;
}
.sb-tag {
    display: inline-flex; align-items: center; gap: 5px;
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase;
    color: var(--text-on-dark); background: rgba(221,221,216,0.1);
    border: 1px solid rgba(221,221,216,0.2);
    border-radius: 4px; padding: 3px 8px;
    margin-top: 2px;
}
.sb-tag-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--text-on-dark); flex-shrink: 0;
    animation: pulse 2.4s ease-in-out infinite;
}
@keyframes pulse { 0%,100%{opacity:1;} 50%{opacity:0.3;} }

.sb-nav-group {
    padding: 20px 14px 8px;
}
.sb-nav-label {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-on-dark-lo); padding: 0 8px;
    margin-bottom: 6px; display: block;
}
.sb-nav-item {
    display: flex; align-items: center; gap: 10px;
    padding: 10px 12px; border-radius: var(--radius-sm);
    color: var(--text-on-dark-muted); font-size: 0.88rem;
    font-weight: 400; text-decoration: none;
    border: 1px solid transparent;
    transition: color 0.15s ease, background 0.15s ease, border-color 0.15s ease;
    margin-bottom: 2px; cursor: pointer;
    background: transparent; width: 100%; text-align: left;
    position: relative;
}
.sb-nav-item:hover {
    color: var(--text-on-dark);
    background: rgba(255,255,255,0.06);
}
.sb-nav-item.active {
    color: var(--text-on-dark);
    background: rgba(255,255,255,0.1);
    border-color: rgba(255,255,255,0.15);
    font-weight: 500;
}
.sb-nav-item.active::before {
    content: '';
    position: absolute; left: 0; top: 50%;
    transform: translateY(-50%);
    width: 3px; height: 16px;
    background: var(--text-on-dark);
    border-radius: 0 2px 2px 0;
}
.sb-nav-icon {
    font-size: 0.75rem; width: 18px;
    text-align: center; flex-shrink: 0;
    opacity: 0.8;
}
.sb-nav-arrow { margin-left: auto; font-size: 0.7rem; opacity: 0.3; }
.sb-nav-item.active .sb-nav-arrow { opacity: 0.8; }

.sb-palette {
    padding: 16px 22px 0;
}
.sb-palette-label {
    font-size: 0.62rem; font-weight: 600;
    letter-spacing: 0.14em; text-transform: uppercase;
    color: var(--text-on-dark-lo); display: block; margin-bottom: 10px;
}
.sb-palette-dots {
    display: flex; flex-direction: column; gap: 7px;
}
.sb-palette-item {
    display: flex; align-items: center; gap: 9px;
    font-size: 0.76rem; color: var(--text-on-dark-muted);
}
.sb-palette-dot {
    width: 7px; height: 7px; border-radius: 50%; flex-shrink: 0;
}

.sb-footer {
    margin-top: auto;
    padding: 16px 14px 24px;
    border-top: 1px solid var(--border-dark);
}
.sb-model-chip {
    display: flex; align-items: center; gap: 8px;
    background: var(--brand-dark);
    border: 1px solid var(--border-dark);
    border-radius: var(--radius-sm);
    padding: 10px 14px;
}
.sb-model-icon { font-size: 0.85rem; color: var(--text-on-dark); }
.sb-model-name { font-size: 0.75rem; font-weight: 500; color: var(--text-on-dark); }
.sb-model-detail { font-size: 0.67rem; color: var(--text-on-dark-muted); }

/* ── CARDS & BOXES ────────────────────────────────────── */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: var(--brand-navy) !important;
    border: 1px solid #C8A96E !important;
div[data-testid="stVerticalBlockBorderWrapper"] > div,
div[data-testid="stVerticalBlockBorderWrapper"] > div > div {
    background-color: #2b3d63 !important;
}

    border-radius: var(--radius-lg) !important;
    box-shadow: 0 8px 30px rgba(43,61,99,0.15) !important;
    padding: 0.2rem !important;
    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 36px rgba(43,61,99,0.2) !important;
}
div[data-testid="stVerticalBlockBorderWrapper"] p,
div[data-testid="stVerticalBlockBorderWrapper"] div,
div[data-testid="stVerticalBlockBorderWrapper"] span {
    color: var(--text-on-dark) !important;
}

/* ── BUTTONS ──────────────────────────────────────────── */
div.stButton > button {
    background: var(--gold) !important;
    color: #1e2b45 !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 600 !important; font-size: 0.88rem !important;
    padding: 0.65rem 1.5rem !important;
    letter-spacing: 0.01em !important;
    transition: all 0.18s ease !important;
    box-shadow: 0 4px 14px rgba(200,169,110,0.3) !important;
}
div.stButton > button:hover {
    opacity: 0.95 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 18px rgba(200,169,110,0.4) !important;
}

/* ── FILE UPLOADER ────────────────────────────────────── */
[data-testid="stFileUploaderDropzone"],
section[data-testid="stFileUploadDropzone"] {
    background: rgba(255,255,255,0.05) !important;
    border: 1px dashed var(--border-dark) !important;
    border-radius: var(--radius-md) !important;
    transition: all 0.2s ease !important;
}
[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--gold) !important;
    background: rgba(255,255,255,0.08) !important;
}

/* ── HERO ─────────────────────────────────────────────── */
.hero {
    position: relative;
    border-radius: var(--radius-xl);
    overflow: hidden;
    padding: 60px 48px 52px;
    margin-bottom: 32px;
    background: linear-gradient(135deg, var(--brand-navy) 0%, var(--brand-dark) 100%);
    box-shadow: 0 10px 40px rgba(43,61,99,0.15);
}
.hero-canvas-bg {
    position: absolute; inset: 0;
    background-image:
        repeating-linear-gradient(0deg, transparent, transparent 24px, rgba(221,221,216,0.03) 24px, rgba(221,221,216,0.03) 25px),
        repeating-linear-gradient(90deg, transparent, transparent 24px, rgba(221,221,216,0.03) 24px, rgba(221,221,216,0.03) 25px);
    pointer-events: none;
}
.hero-glow {
    position: absolute; top: -80px; right: -60px;
    width: 360px; height: 360px; border-radius: 50%;
    background: radial-gradient(circle, rgba(200,169,110,0.12) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--gold); margin-bottom: 18px;
    display: flex; align-items: center; gap: 8px;
}
.hero-eyebrow::before {
    content: ''; display: inline-block;
    width: 24px; height: 1px; background: var(--gold);
}
.hero h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 3.2rem !important; font-weight: 400 !important;
    color: var(--text-on-dark) !important;
    line-height: 1.1 !important; margin: 0 0 18px 0 !important;
    letter-spacing: -0.01em !important;
}
.hero h1 em { font-style: italic; color: var(--gold); }
.hero p {
    font-size: 1rem; color: var(--text-on-dark-muted);
    max-width: 480px; line-height: 1.65; margin: 0 0 28px 0;
}
.hero-cta {
    display: inline-flex; align-items: center; gap: 8px;
    background: var(--gold); color: #1e2b45;
    font-size: 0.88rem; font-weight: 600;
    padding: 11px 24px; border-radius: var(--radius-sm);
    text-decoration: none; transition: all 0.2s ease;
    letter-spacing: 0.01em; box-shadow: 0 4px 14px rgba(200,169,110,0.3);
}
.hero-cta:hover { transform: translateY(-2px); box-shadow: 0 6px 18px rgba(200,169,110,0.4); }
.hero-cta-arrow { font-size: 1rem; }

/* ── STAT CHIPS ───────────────────────────────────────── */
.stat-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin: 0 0 36px 0; }
.stat-chip {
    background: var(--brand-navy);
    border: 1px solid var(--border-dark);
    border-radius: var(--radius-md); padding: 20px 22px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(43,61,99,0.08);
}
.stat-chip:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(43,61,99,0.15); }
.stat-chip-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; font-weight: 400;
    color: var(--gold); line-height: 1; margin-bottom: 6px; display: block;
}
.stat-chip-label { font-size: 0.78rem; color: var(--text-on-dark-muted); font-weight: 400; line-height: 1.4; display: block; }

/* ── SECTION HEADING ────────────────────── */
.section-head { margin: 0 0 20px 0; display: flex; align-items: baseline; gap: 12px; }
.section-head h2 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 1.5rem !important; font-weight: 400 !important;
    color: var(--text-on-light) !important; margin: 0 !important;
}
.section-divider { flex: 1; height: 1px; background: var(--border-light); }

/* ── HOW IT WORKS CARDS ───────────────────────────────── */
.how-card {
    background: var(--brand-navy);
    border: 1px solid var(--border-dark);
    border-radius: var(--radius-md); padding: 24px; height: 100%;
    box-shadow: 0 4px 15px rgba(43,61,99,0.08);
}
.how-num {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem; color: rgba(221,221,216,0.15);
    line-height: 1; margin-bottom: 10px; display: block;
}
.how-title { font-size: 0.9rem; font-weight: 600; color: var(--text-on-dark); margin-bottom: 8px; }
.how-desc { font-size: 0.82rem; color: var(--text-on-dark-muted); line-height: 1.6; }

/* ── INFO PAGE ────────────────────────────────────────── */
.info-header { margin-bottom: 28px; }
.info-header .hero-eyebrow { color: var(--brand-navy); }
.info-header .hero-eyebrow::before { background: var(--brand-navy); }
.info-header h1 {
    font-family: 'DM Serif Display', serif !important;
    font-size: 2.4rem !important; font-weight: 400 !important;
    color: var(--text-on-light) !important; margin: 0 0 8px 0 !important;
}
.info-header p { font-size: 0.95rem; color: var(--text-on-light-muted); line-height: 1.6; margin: 0; }

/* ── KLASIFIKASI PAGE ─────────────────────────────────── */
.page-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem; font-weight: 400;
    color: var(--text-on-light); margin: 0 0 6px 0;
}
.page-sub { font-size: 0.9rem; color: var(--text-on-light-muted); margin: 0 0 28px 0; }
.klas-eyebrow {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.16em; text-transform: uppercase;
    color: var(--brand-navy); margin-bottom: 10px;
    display: flex; align-items: center; gap: 8px;
}
.klas-eyebrow::before { content: ''; display: inline-block; width: 24px; height: 1px; background: var(--brand-navy); }

.upload-label {
    font-size: 0.75rem; 
    font-weight: 600;
    letter-spacing: 0.1em; 
    text-transform: uppercase;
    color: #1e2b45 !important; /* Warna berubah jadi navy pekat */
    display: block; 
    margin-bottom: 8px;
}

/* ── RESULT BOX ───────────────────────────────────────── */
.result-wrap {
    background: linear-gradient(135deg, rgba(255,255,255,0.03) 0%, rgba(255,255,255,0.0) 100%);
    border: 1px solid var(--border-dark);
    border-radius: var(--radius-lg); padding: 28px; margin: 20px 0;
}
.result-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--text-on-dark-lo); margin-bottom: 10px; display: block;
}
.result-style-name {
    font-family: 'DM Serif Display', serif;
    font-size: 2.4rem; font-weight: 400;
    color: var(--text-on-dark); margin: 0 0 4px 0; line-height: 1.1;
}
.result-icon { font-size: 1.5rem; }
.result-accent-bar { height: 2px; width: 40px; border-radius: 1px; margin: 12px 0 0 0; display: block; }

/* ── PROB BARS (Teks Dibuat Kontras) ────────────────────────── */
.prob-section { margin-top: 24px; }
.prob-section-label {
    font-size: 0.72rem; font-weight: 600;
    letter-spacing: 0.12em; text-transform: uppercase;
    color: var(--gold) !important; margin-bottom: 16px; display: block;
}
.prob-row { margin-bottom: 12px; }
.prob-header { display: flex; justify-content: space-between; font-size: 0.82rem; margin-bottom: 6px; }
.prob-name { 
    color: #ffffff !important; /* Diubah menjadi putih terang agar sangat jelas */
    font-weight: 500 !important; 
}
.prob-pct { 
    color: var(--gold) !important; /* Diubah menjadi warna emas agar kontras dengan latar belakang */
    font-weight: 600 !important; font-size: 0.82rem; 
}
.prob-track { height: 6px; background: rgba(255,255,255,0.15); border-radius: 4px; overflow: hidden; }
.prob-fill { height: 100%; border-radius: 4px; }

/* ── SUCCESS / ERROR ALERTS ───────────────────────────── */
[data-testid="stAlert"] {
    background: var(--brand-dark) !important;
    border: 1px solid var(--border-dark) !important;
    border-radius: var(--radius-md) !important;
    color: var(--text-on-dark) !important;
}

@media (prefers-reduced-motion: reduce) { * { transition: none !important; animation: none !important; } }
@media (max-width: 720px) {
    .hero { padding: 36px 24px; }
    .hero h1 { font-size: 2.2rem !important; }
    .stat-row { grid-template-columns: 1fr; }
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 6. SIDEBAR
# ============================================================
dot_items = "".join(
    f'<div class="sb-palette-item">'
    f'<span class="sb-palette-dot" style="background:{v["color"]};"></span>'
    f'<span>{name}</span>'
    f'</div>'
    for name, v in STYLES.items()
)

nav_items = ""
for i, item in enumerate(MENU_META):
    active_cls = " active" if i == _active_idx else ""
    nav_items += (
        # ⚠️ UBAH target="_top" MENJADI target="_self" DI BARIS BAWAH INI
        f'<a class="sb-nav-item{active_cls}" href="?page={item["slug"]}" target="_self">'
        f'<span class="sb-nav-icon">{item["icon"]}</span>'
        f'<span>{item["label"]}</span>'
        f'<span class="sb-nav-arrow">›</span>'
        f'</a>'
    )

# --- LOGIC UNTUK MENAMPILKAN LOGO ---
logo_b64 = get_image_b64("logo.jpg")

if logo_b64:
    # Logo berukuran lebih besar (76px), membulat sempurna, dan ditaruh di tengah atas
    logo_html = f'<img src="{logo_b64}" style="width: 200px; height: 100px; border-radius: 12px; object-fit: cover; border: 1px solid rgba(200,169,110,0.25); box-shadow: 0 4px 14px rgba(0,0,0,0.3); flex-shrink: 0; margin-bottom: 4px;">'
else:
    # Fallback jika gambar logo.jpg belum dimasukkan ke direktori
    logo_html = '<div class="sb-brand-gem" style="width: 56px; height: 56px; font-size: 1.3rem; margin-bottom: 4px;">◈</div>'

st.sidebar.markdown(f"""
<div class="sb">
  <div class="sb-brand">
    <div class="sb-brand-mark">
      {logo_html}
      <div>
        <div class="sb-brand-name">Jazirah.id</div>
        <div class="sb-brand-sub">Timur Labuhan Kata</div>
      </div>
    </div>
    <span class="sb-tag">
      <span class="sb-tag-dot"></span>
      Since 2024
    </span>
  </div>

  <div class="sb-nav-group">
    <span class="sb-nav-label">Menu</span>
    {nav_items}
  </div>

  <div class="sb-footer">
    <div class="sb-model-chip">
      <span class="sb-model-icon">◎</span>
      <div>
        <div class="sb-model-name">Random Forest</div>
        <div class="sb-model-detail">Computer Vision · 5 kelas</div>
      </div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# ============================================================
# HALAMAN: BERANDA
# ============================================================
if pilihan_menu == "beranda":
    # 1. LOAD 4 GAMBAR LOKAL DARI FOLDER LAPTOP UNTUK CAROUSEL
    img_car_1 = get_image_b64("rupa1.jpg") or "https://via.placeholder.com/1200x600/2b3d63/ddddd8?text=Dokumentasi+Kegiatan+1"
    img_car_2 = get_image_b64("rupa2.jpg") or "https://via.placeholder.com/1200x600/2b3d63/ddddd8?text=Dokumentasi+Kegiatan+2"
    img_car_3 = get_image_b64("diskusi1.jpg") or "https://via.placeholder.com/1200x600/2b3d63/ddddd8?text=Dokumentasi+Kegiatan+3"
    img_car_4 = get_image_b64("diskusi2.jpg") or "https://via.placeholder.com/1200x600/2b3d63/ddddd8?text=Dokumentasi+Kegiatan+4"

    # 2. RENDER CAROUSEL DENGAN 4 SLIDE DAN AUTO-SCROLL JAVASCRIPT
    st.markdown(f"""
    <style>
    /* CSS Khusus Carousel Horizontal di Atas */
    .top-carousel-container {{
        display: flex;
        gap: 16px;
        overflow-x: auto;
        scroll-snap-type: x mandatory;
        padding: 4px 4px 20px 4px;
        margin-bottom: 20px;
        scrollbar-width: none;
        scroll-behavior: smooth; /* Transisi pergeseran menjadi halus */
    }}
    .top-carousel-container::-webkit-scrollbar {{ display: none; }}
    .carousel-slide {{
        flex: 0 0 85%;
        scroll-snap-align: center;
        border-radius: var(--radius-lg);
        overflow: hidden;
        height: 280px;
        border: 4px solid var(--brand-navy); 
        box-shadow: 0 12px 24px rgba(43,61,99,0.2);
        position: relative;
    }}
    .carousel-slide img {{
        width: 100%; height: 100%; object-fit: cover; display: block;
    }}
    .carousel-hint {{
        text-align: center; font-size: 0.75rem; color: var(--text-on-light-muted);
        margin-top: -15px; margin-bottom: 25px; font-weight: 500; letter-spacing: 0.05em;
    }}
    @media (max-width: 720px) {{
        .carousel-slide {{ flex: 0 0 90%; height: 180px; }}
    }}
    </style>

    <div class="top-carousel-container" id="autoCarousel">
        <div class="carousel-slide"><img src="{img_car_1}" alt="Dokumentasi Rupa 1"></div>
        <div class="carousel-slide"><img src="{img_car_2}" alt="Dokumentasi Rupa 2"></div>
        <div class="carousel-slide"><img src="{img_car_3}" alt="Dokumentasi Diskusi 1"></div>
        <div class="carousel-slide"><img src="{img_car_4}" alt="Dokumentasi Diskusi 2"></div>
    </div>
    <div class="carousel-hint">← Geser atau tunggu beberapa detik untuk melihat galeri →</div>

    <script>
    // Logika Auto-Scroll Carousel
    document.addEventListener("DOMContentLoaded", function() {{
        const carousel = document.getElementById("autoCarousel");
        let scrollInterval = setInterval(autoScroll, 3000); // 3000ms = 3 detik per geseran

        function autoScroll() {{
            if (!carousel) return;
            
            // Hitung lebar satu slide ditambah jarak gap (16px)
            const slideWidth = carousel.querySelector(".carousel-slide").offsetWidth + 16;
            const maxScrollLeft = carousel.scrollWidth - carousel.clientWidth;
            
            // Jika sudah sampai ujung kanan, kembali ke awal (kiri = 0)
            if (carousel.scrollLeft >= maxScrollLeft) {{
                carousel.scrollTo({{ left: 0, behavior: "smooth" }});
            }} else {{
                carousel.scrollBy({{ left: slideWidth, behavior: "smooth" }});
            }}
        }}

        // Opsional: Jeda auto-scroll saat kursor mouse diarahkan ke carousel
        carousel.addEventListener("mouseenter", function() {{
            clearInterval(scrollInterval);
        }});
        
        // Lanjutkan auto-scroll saat kursor mouse menjauh
        carousel.addEventListener("mouseleave", function() {{
            scrollInterval = setInterval(autoScroll, 3000);
        }});
    }});
    </script>
    """, unsafe_allow_html=True)

    # 3. HERO (JAZIRAH TIMUR LABUHAN KATA)
    st.markdown("""
    <div class="hero">
      <div class="hero-canvas-bg"></div>
      <div class="hero-glow"></div>
      <h1><em>Jazirah :</em><br>Timur Labuhan Kata</h1>
      <p>Jazirah: Timur Labuhan Kata adalah perayaan literasi pertama di Kota Ambon yang
membuka ruang temu bagi pertukaran gagasan dan eksplorasi kreatif, berakar pada sejarah
dan kebudayaan Maluku. Inisiatif ini merupakan bagian dari program kebudayaan jangka
panjang untuk pengembangan sastra, musik, dan seni di Maluku, Indonesia.</p>
        """, unsafe_allow_html=True)

   # 4. VISI MISI
    st.markdown("""
    <div style="display: flex; justify-content: center; gap: 20px; flex-wrap: wrap; margin: 0 !important;">
      <div class="stat-chip" style="flex: 1; min-width: 280px; max-width: 450px;">
        <span class="stat-chip-num" style="color: white; font-size: 1.7rem;">Visi Jazirah</span>
        <div class="stat-chip-label">
          <ul style="margin-top: 4px; margin-bottom: 0; padding-left: 18px;">
            <li>Menjadikan Maluku sebagai loka yang bergairah bagi pengembangan sastra, musik, dan seni.</li>
          </ul>
        </div>
      </div>
      <div class="stat-chip" style="flex: 1; min-width: 280px; max-width: 450px;">
        <span class="stat-chip-num" style="color: white; font-size: 1.7rem;">Misi Jazirah</span>
        <div class="stat-chip-label">
          <ul style="margin-top: 4px; margin-bottom: 0; padding-left: 18px;">
            <li>Mendorong ekspresi artistik</li>
            <li>Menumbuhkan komunitas kreatif</li>
            <li>Mendukung generasi kreatif baru</li>
            <li>Melestarikan warisan budaya</li>
          </ul>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
# 5. PROGRAM JAZIRAH (GRID DENGAN RENDER HTML/MARKDOWN LANGSUNG & POPUP)
    st.markdown("""
    <style>
    .prog-section-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.2rem !important;
        color: var(--text-on-light) !important;
        margin: 48px 0 12px 0 !important;
        font-weight: 400 !important;
    }
    .prog-section-subtitle {
        font-size: 0.95rem !important;
        color: var(--brand-dark) !important;
        margin-bottom: 32px !important;
        font-weight: 500 !important;
        letter-spacing: 0.01em;
    }
    .prog-title-wrap {
        display: flex;
        flex-wrap: wrap;
        justify-content: space-between;
        align-items: center;
        margin: 48px 0 32px 0;
        gap: 20px;
    }
    .prog-main-title {
        font-family: 'DM Serif Display', serif !important;
        font-size: 2.2rem !important;
        color: var(--text-on-light) !important;
        margin: 0 !important;
        font-weight: 400 !important;
    }
    .prog-badge {
        display: flex;
        align-items: center;
        background: rgba(74,222,128,0.18);
        border: 1px solid rgba(74,222,128,0.3);
        border-radius: 999px;
        padding: 8px 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: #1e6b3b;
        letter-spacing: 0.02em;
    }
    .prog-badge-num {
        font-family: 'DM Serif Display', serif;
        font-size: 1.6rem;
        margin-right: 8px;
        line-height: 1;
    }
    .prog-grid-custom {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 24px;
        margin-bottom: 48px;
    }
    .pop-desc {
        padding: 8px 4px;
        font-size: 0.95rem;
        color: var(--brand-dark);
        line-height: 1.5;
        letter-spacing: 0.01em;
    }
    @media (max-width: 720px) {
        .prog-grid-custom { grid-template-columns: 1fr; gap: 16px; }
        .prog-title-wrap { margin: 32px 0 20px 0; flex-direction: column; align-items: flex-start; gap: 10px; }
    }
    </style>

    <h2 class="prog-section-title">PROGRAM JAZIRAH</h2>
    """, unsafe_allow_html=True)

    programs_data = [
        ("01", "JAZIRAH DISKUSI", 
         "Ruang diskusi yang akan membahas secara mendalam perihal seni, sastra, sejarah, dan kebudayaan Maluku serta topik-topik lain yang relevan.", 
         "diskusi2.jpg"),
        ("08", "JAZIRAH RUPA", 
        "Ruang pameran untuk menampilkan karya-karya terpilih dari seniman dan fotografer yang telah melaui proses kurasi",
         "rupa1.jpg"),
        ("02", "JAZIRAH OBROLAN", 
        "Ruang obrolan yang lebih santai seputar aktivitas seni sastra yang dilakukan individu dan komunitas seni sastra di Maluku",
         "obrolan1.jpg"),
        ("09", "JAZIRAH TUTUR", 
        "Ruang percakapan dengan para sesepuh dan budayawan membagikan pengalaman dan wawasan mereka tentang nilai-nilai yang menjadi kekayaan budaya Maluku.",
         "tutur1.jpg"),
        ("04", "JAZIRAH LOKAKARYA", 
        "Ruang pelatihan dasar menulis, melukis, mengarang lagu, menyusun naskah film, dan lainnya yang relevan dengan konteks seni, sastra, dan budaya.",
         "lokakarya1.jpg"),
        ("10", "HASA-HASA JAZIRAH", 
        "Kegiatan belajar bersama dengan berjalan kaki menelusuri jejak sejarah kota. Melalui pengalaman langsung di ruang-ruang kota, peserta diajak mengenali lapisan waktu dan perubahan yang membentuknya.",
         "hasa-hasa1.jpg"),
        ("05", "JAZIRAH DONGENG", 
        "Ruang khusus untuk anak-anak bermain dan berimajinasi dengan jalan mendengarkan dongeng dan cerita rakyat di Maluku. Jazirah Dongeng adalah program kolaboratif dengan komunitas literasi Timur Menulis.",
         "dongeng1.jpg"),
        ("11", "WAKTU INDONESIA BAGIAN KOPI", 
        "Mengawali hari dengan percakapan hangat bersama penulis, seniman, pegiat literasi—menyusuri lembar-lembar buku, berbagi cerita, dan menyesap gagasan dalam suasana yang akrab.",
         "wibkopi1.jpg"),
        ("06", "JAZIRAH REMPAH", 
        "Pertunjukan musik, tari, dan teater yang menyuguhkan karya asli para seniman di Maluku. Jazirah Rempah merupakan kolaborasi antara Jazirah: Timur Labuhan Kata dan Rempah Gunung—aroma dendang sahaja.",
         "rempah1.jpg"),
        ("12", "PASAR GOTONG ROYONG", 
        "Pasar Gotong Royong adalah pasar kuliner yang melibatkan UMKM Kota Ambon sebagai bagian dari usaha membangun literasi makanan.",
         "pasargotong1.jpg"),
        ("07", "PASAR BUKU MARDIKA", 
        "Pertunjukan puisi yang ditampilkan setiap petang di area Jazirah: Timur Labuhan Kata yang berkolaborasi dengan Bengkel Sastra Maluku.",
         "buku mardika .png"),
        ("13", "PESTA PUISI", 
         "Selebrasi puitika, pembacaan sajak, dan pertunjukan musikalisasi puisi yang merayakan keindahan bahasa.", 
         "puisi1.jpg")
    ]

    kiri, kanan = st.columns(2)
    
    for index, (num, title, desc, img_file) in enumerate(programs_data):
        target_col = kiri if index % 2 == 0 else kanan
        
        with target_col:
            with st.expander(f"**{num} &nbsp;|&nbsp; {title}**"):
                st.markdown(f"<div class='pop-desc'>{desc}</div>", unsafe_allow_html=True)
                
                try:
                    # Menggunakan width dalam pixel atau use_container_width=True
                    st.image(img_file, width=450)
                except Exception as e:
                    st.caption(f"💡 (Catatan: Gambar '{img_file}' belum tersedia)")


# ============================================================
# HALAMAN: INFORMASI GAYA
# ============================================================
elif pilihan_menu == "informasi":
    st.markdown("""
    <div class="info-header">
      <div class="klas-eyebrow">Panduan Visual</div>
      <h1>Lima Gaya Lukisan</h1>
      <p>Klik salah satu kartu untuk membaca penjelasan lengkap, karakteristik visual, dan contoh karya dari setiap gaya yang dikenali sistem ini.</p>
    </div>
    """, unsafe_allow_html=True)

    styles_js = {}
    for name, s in STYLES.items():
        img_b64 = get_image_b64(s["image"])
        key = name.replace(" ", "_")
        styles_js[key] = {
            "name": name, "icon": s["icon"],
            "color": s["color"], "colorSoft": s["color_soft"],
            "tagline": s["tagline"], "desc": s["desc"],
            "traits": s["traits"], "img": img_b64 or "",
        }
    styles_json = json.dumps(styles_js, ensure_ascii=False)

    cards_html = ""
    for name, s in STYLES.items():
        key = name.replace(" ", "_")
        cards_html += (
            f'<div class="g-card" onclick="openModal(\'{key}\')" '
            f'role="button" tabindex="0" '
            f'onkeydown="if(event.key===\'Enter\')openModal(\'{key}\')">'
            f'<div class="g-card-top" style="background:radial-gradient(circle at 70% 30%, {s["color"]}22 0%, transparent 65%);">'
            f'<span class="g-card-icon">{s["icon"]}</span>'
            f'<span class="g-card-dot" style="background:{s["color"]};"></span>'
            f'</div>'
            f'<div class="g-card-body">'
            f'<div class="g-card-name">{name}</div>'
            f'<div class="g-card-tagline">{s["tagline"]}</div>'
            f'<div class="g-card-cta" style="color:{s["color"]};">Lihat detail →</div>'
            f'</div>'
            f'</div>'
        )

    gallery_component = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=Inter:wght@300;400;500;600&display=swap');
  *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Inter', sans-serif;
    background: #ddddd8;
    color: #1e2b45;
    padding: 4px 2px 16px;
  }}
  .g-grid {{
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(170px, 1fr));
    gap: 12px;
    margin-bottom: 8px;
  }}
  .g-card {{
    background: #2b3d63;
    border: 1px solid rgba(221,221,216,0.15);
    border-radius: 16px;
    overflow: hidden;
    cursor: pointer;
    transition: border-color 0.2s ease, transform 0.2s ease, box-shadow 0.2s ease;
    user-select: none;
    box-shadow: 0 4px 15px rgba(43,61,99,0.08);
  }}
  .g-card:hover {{
    border-color: rgba(221,221,216,0.3);
    transform: translateY(-3px);
    box-shadow: 0 8px 25px rgba(43,61,99,0.15);
  }}
  .g-card-top {{
    height: 80px;
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 16px;
    position: relative;
  }}
  .g-card-icon {{ font-size: 2rem; line-height: 1; }}
  .g-card-dot {{
    width: 8px; height: 8px;
    border-radius: 50%; flex-shrink: 0;
  }}
  .g-card-body {{ padding: 14px 16px 18px; }}
  .g-card-name {{
    font-family: 'DM Serif Display', serif;
    font-size: 1.05rem; color: #ddddd8;
    margin-bottom: 5px; line-height: 1.1;
  }}
  .g-card-tagline {{
    font-size: 0.75rem; color: rgba(221,221,216,0.65);
    line-height: 1.45; margin-bottom: 12px;
  }}
  .g-card-cta {{
    font-size: 0.75rem; font-weight: 500;
    transition: opacity 0.15s;
  }}
  .g-card:hover .g-card-cta {{ opacity: 0.8; }}

  /* MODAL */
  .mo {{
    display: none; position: fixed; inset: 0; z-index: 999;
    background: rgba(30,43,69,0.6);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    align-items: center; justify-content: center;
    padding: 16px;
  }}
  .mo.open {{ display: flex; animation: fadeIn 0.15s ease; }}
  @keyframes fadeIn {{ from {{opacity:0;}} to {{opacity:1;}} }}
  @keyframes slideUp {{
    from {{opacity:0; transform:translateY(18px) scale(0.98);}}
    to   {{opacity:1; transform:translateY(0)   scale(1);   }}
  }}
  .mo-box {{
    background: #2b3d63;
    border: 1px solid rgba(221,221,216,0.2);
    border-radius: 22px;
    width: 100%; max-width: 640px;
    max-height: 88vh; overflow-y: auto;
    position: relative;
    box-shadow: 0 20px 60px rgba(30,43,69,0.3);
    animation: slideUp 0.22s cubic-bezier(0.34,1.36,0.64,1);
  }}
  .mo-close {{
    position: absolute; top: 14px; right: 14px;
    width: 32px; height: 32px; border-radius: 50%;
    border: 1px solid rgba(255,255,255,0.15);
    background: rgba(255,255,255,0.05);
    color: rgba(221,221,216,0.8);
    cursor: pointer; font-size: 0.85rem;
    display: flex; align-items: center; justify-content: center;
    transition: background 0.15s; z-index: 10;
  }}
  .mo-close:hover {{ background: rgba(255,255,255,0.15); }}
  .mo-img-wrap {{
    height: 320px; overflow: hidden;
    border-radius: 22px 22px 0 0; position: relative;
    display: flex; align-items: center; justify-content: center;
  }}
  .mo-img-wrap img {{ width:100%; height:100%; object-fit:contain; display:block; padding: 20px 20px 40px 20px;}}
  .mo-img-placeholder {{
    width:100%; height:100%;
    display:flex; flex-direction:column;
    align-items:center; justify-content:center; gap:8px;
  }}
  .mo-img-badge {{
    position:absolute; top:14px; left:14px;
    background:rgba(221,221,216,0.9); backdrop-filter:blur(8px);
    padding:5px 14px; border-radius:999px;
    font-size:0.76rem; font-weight:600; color: #1e2b45;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
  }}
  .mo-body {{ padding:24px 28px 28px; }}
  .mo-header {{ display:flex; align-items:center; gap:12px; margin-bottom:6px; }}
  .mo-icon {{ font-size:1.6rem; line-height:1; flex-shrink:0; }}
  .mo-title {{
    font-family:'DM Serif Display', serif;
    font-size:1.5rem; font-weight:400; color:#ddddd8;
    margin:0; line-height:1.15;
  }}
  .mo-tagline {{
    font-style:italic; color:rgba(221,221,216,0.65);
    font-size:0.88rem; margin:6px 0 16px;
  }}
  .mo-accent {{ height:2px; margin:0 0 18px; width:40px; border-radius:1px; }}
  .mo-desc {{
    color:rgba(221,221,216,0.85); line-height:1.7;
    font-size:0.9rem; margin-bottom:20px;
  }}
  .mo-traits-label {{
    font-size:0.65rem; font-weight:600;
    letter-spacing:0.12em; text-transform:uppercase;
    color:rgba(221,221,216,0.5); margin-bottom:10px;
  }}
  .mo-traits {{ display:flex; flex-wrap:wrap; gap:7px; }}
  .mo-trait {{
    font-size:0.78rem; font-weight:500;
    padding:5px 13px; border-radius:999px;
    border:1px solid var(--tc);
    color:var(--tc); background:var(--ts);
  }}
  @media(max-width:480px) {{
    .g-grid {{ grid-template-columns: repeat(2,1fr); }}
    .mo-body {{ padding:18px; }}
  }}
</style>
</head>
<body>
<div class="g-grid">{cards_html}</div>
<div class="mo" id="mo">
  <div class="mo-box" id="moBox">
    <button class="mo-close" id="moClose">✕</button>
    <div class="mo-img-wrap" id="moImg"></div>
    <div class="mo-body">
      <div class="mo-header">
        <span class="mo-icon" id="moIcon"></span>
        <h2 class="mo-title" id="moTitle"></h2>
      </div>
      <p class="mo-tagline" id="moTagline"></p>
      <div class="mo-accent" id="moAccent"></div>
      <p class="mo-desc" id="moDesc"></p>
      <div class="mo-traits-label">Karakteristik Visual</div>
      <div class="mo-traits" id="moTraits"></div>
    </div>
  </div>
</div>
<script>
var D={styles_json};
function openModal(k){{
  var s=D[k]; if(!s)return;
  var iw=document.getElementById('moImg');
  iw.style.background='radial-gradient(circle at 60% 40%,'+s.colorSoft+',#1e2b45 70%)';
  if(s.img){{
    iw.innerHTML='<img src="'+s.img+'" alt="'+s.name+'">'
      +'<div class="mo-img-badge">'+s.icon+' '+s.name+'</div>';
  }}else{{
    iw.innerHTML='<div class="mo-img-placeholder">'
      +'<span style="font-size:2.8rem;">'+s.icon+'</span>'
      +'<span style="font-size:0.78rem;color:'+s.color+';font-weight:600;">Gambar contoh belum tersedia</span>'
      +'</div>'
      +'<div class="mo-img-badge">'+s.icon+' '+s.name+'</div>';
  }}
  document.getElementById('moIcon').textContent=s.icon;
  document.getElementById('moTitle').textContent=s.name;
  document.getElementById('moTagline').textContent=s.tagline;
  document.getElementById('moAccent').style.background=s.color;
  document.getElementById('moDesc').textContent=s.desc;
  document.getElementById('moTraits').innerHTML=s.traits.map(function(t){{
    return '<span class="mo-trait" style="--tc:'+s.color+';--ts:'+s.colorSoft+';">'+t+'</span>';
  }}).join('');
  document.getElementById('mo').classList.add('open');
}}
function closeModal(){{ document.getElementById('mo').classList.remove('open'); }}
document.getElementById('mo').addEventListener('click',function(e){{ if(e.target===this)closeModal(); }});
document.getElementById('moClose').addEventListener('click',closeModal);
document.addEventListener('keydown',function(e){{ if(e.key==='Escape')closeModal(); }});
</script>
</body>
</html>"""

    components.html(gallery_component, height=680, scrolling=False)

# ============================================================
# HALAMAN: KLASIFIKASI
# ============================================================
elif pilihan_menu == "klasifikasi":
    st.markdown("""
    <div style="margin-bottom:28px;">
      <div class="klas-eyebrow">Analisis Citra</div>
      <div class="page-title">Analisis Lukisan</div>
      <div class="page-sub">Unggah foto lukisan, dan model akan mengidentifikasi gayanya beserta tingkat keyakinannya.</div>
    </div>
    """, unsafe_allow_html=True)

    # ─── INJECT CSS UNTUK CARD NAVY ───────────────────────────
    st.markdown("""
    <style>
    /* Background navy untuk semua container border di halaman ini */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #2b3d63 !important;
        border: 1px solid #C8A96E !important;
        border-radius: 16px !important;
        box-shadow: 0 8px 24px rgba(0,0,0,0.20) !important;
        padding: 4px 0 !important;
    }
    /* Teks dalam container jadi terang */
    [data-testid="stVerticalBlockBorderWrapper"] p,
    [data-testid="stVerticalBlockBorderWrapper"] span,
    [data-testid="stVerticalBlockBorderWrapper"] small,
    [data-testid="stVerticalBlockBorderWrapper"] div {
        color: #ddddd8 !important;
    }
    /* Label caption gambar */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="caption"] {
        color: rgba(221,221,216,0.6) !important;
    }
    /* Dropzone uploader */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderDropzone"] {
        background-color: rgba(255,255,255,0.04) !important;
        border: 1.5px dashed rgba(200,169,110,0.55) !important;
        border-radius: 10px !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderDropzone"]:hover {
        background-color: rgba(200,169,110,0.07) !important;
        border-color: #C8A96E !important;
    }
    /* Teks "Browse files" & drag-drop hint */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderDropzone"] span,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderDropzone"] p,
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderDropzone"] small {
        color: rgba(221,221,216,0.75) !important;
    }
    /* File chip yang sudah diupload */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderFileName"] {
        color: #ddddd8 !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stFileUploaderFileData"] {
        background-color: rgba(255,255,255,0.08) !important;
        border-radius: 8px !important;
    }
    /* Tombol Analisis */
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stBaseButton-secondary"] {
        background-color: rgba(200,169,110,0.15) !important;
        border: 1px solid #C8A96E !important;
        color: #C8A96E !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        transition: all 0.2s ease !important;
    }
    [data-testid="stVerticalBlockBorderWrapper"] [data-testid="stBaseButton-secondary"]:hover {
        background-color: #C8A96E !important;
        color: #1a2744 !important;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<span class="upload-label">Pilih gambar</span>', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Format: JPG, JPEG, PNG",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            col_l, col_m, col_r = st.columns([1, 3, 1])
            with col_m:
                st.image(image, caption=uploaded_file.name, width=450)

    if uploaded_file is not None:
        st.markdown("<div style='height:8px;'></div>", unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown('<span class="upload-label">Jalankan model</span>', unsafe_allow_html=True)
            if st.button("Analisis sekarang →", use_container_width=True):
                if scaler is None or model_rf is None:
                    st.error("File model tidak ditemukan. Pastikan 'scaler_final.pkl' dan 'model_final_skripsi.pkl' ada di direktori yang sama.")
                else:
                    with st.spinner("Mengekstrak fitur dan menjalankan prediksi..."):
                        fitur_mentah = extract_features(image)
                        try:
                            fitur_scaled   = scaler.transform(fitur_mentah)
                            hasil_prediksi = model_rf.predict(fitur_scaled)[0]
                            prob_array     = model_rf.predict_proba(fitur_scaled)[0]
                        except Exception as e:
                            st.error(f"Gagal melakukan prediksi. Error sistem: {e}")
                            st.stop()

                    kelas_model   = ["Futurisme", "Impresionisme", "Pop Art", "Realisme", "Surealisme"]
                    probabilities = {kelas_model[i]: int(prob_array[i] * 100) for i in range(len(kelas_model))}
                    predicted     = max(probabilities, key=probabilities.get)
                    pred_style    = STYLES.get(predicted, STYLES["Pop Art"])

                    # 💥 KONTENER KUSTOM UNTUK HASIL PREDIKSI UTAMA
                    st.markdown(f"""
                    <div style="
                        background-color: #2b3d63; 
                        padding: 28px; 
                        border-radius: 20px; 
                        border: 2px solid #C8A96E; 
                        box-shadow: 0 10px 30px rgba(43,61,99,0.2);
                        margin-bottom: 24px;
                    ">
                      <span style="font-size: 0.72rem; font-weight: 600; letter-spacing: 0.12em; text-transform: uppercase; color: #ddddd8; margin-bottom: 10px; display: block;">Prediksi gaya</span>
                      <div style="display:flex;align-items:baseline;gap:12px;">
                        <div style="font-family: 'DM Serif Display', serif; font-size: 2.6rem; font-weight: 400; color: #ddddd8; margin: 0 0 4px 0; line-height: 1.1;">{predicted}</div>
                        <div style="font-size: 1.5rem;">{pred_style['icon']}</div>
                      </div>
                      <span style="height: 3px; width: 50px; border-radius: 1.5px; margin: 16px 0 0 0; display: block; background:{pred_style['color']};"></span>
                    </div>
                    """, unsafe_allow_html=True)

                    prob_sorted = dict(sorted(probabilities.items(), key=lambda x: x[1], reverse=True))

                    bars = '<div class="prob-section"><span class="prob-section-label">Probabilitas per kelas</span>'
                    for sname, pct in prob_sorted.items():
                        if pct > 0:
                            c = STYLES[sname]["color"]
                            bars += (
                                f'<div class="prob-row">'
                                f'<div class="prob-header">'
                                f'<span class="prob-name" style="color: #ffffff !important; font-weight: 600;">{sname}</span>'
                                f'<span class="prob-pct" style="color: #C8A96E !important; font-weight: 600;">{pct}%</span>'
                                f'</div>'
                                f'<div class="prob-track" style="height: 6px; background: rgba(255,255,255,0.15); border-radius: 4px; overflow: hidden;">'
                                f'<div class="prob-fill" style="width:{pct}%;background:{c};height: 100%; border-radius: 4px;"></div>'
                                f'</div></div>'
                            )
                    bars += '</div>'

                    st.markdown(f"""
                    <div style="
                        background-color: #2b3d63; 
                        padding: 24px; 
                        border-radius: 16px; 
                        border: 1px solid #C8A96E;
                        box-shadow: 0 8px 24px rgba(0,0,0,0.15);
                        margin-top: 20px;
                    ">
                        {bars}
                    </div>
                    """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="
          background:rgba(255,255,255,0.02);
          border:1px dashed rgba(255,255,255,0.15);
          border-radius:14px; padding:28px 24px; text-align:center;
          margin-top:8px;
        ">
          <div style="font-size:1.6rem;margin-bottom:10px;color:rgba(221,221,216,0.6);">◎</div>
          <div style="font-size:0.85rem;color:rgba(221,221,216,0.5);line-height:1.6;">
            Unggah gambar lukisan di atas untuk memulai analisis.
          </div>
        </div>
        """, unsafe_allow_html=True)
        # ============================================================
# ============================================================
# HALAMAN: TENTANG APLIKASI
# ============================================================
elif pilihan_menu == "tentang":
    st.markdown("""
<div style="margin-bottom:28px;">
  <div class="klas-eyebrow" style="color: #C8A96E;">Informasi Sistem</div>
  <div class="page-title" style="color: #1e2b45;">Tentang Aplikasi</div>
  <div class="page-sub" style="color: rgba(30,43,69,0.75);">Mengenal lebih dekat teknologi di balik Jazirah.id.</div>
</div>
    """, unsafe_allow_html=True)

    # (Lanjut dengan kodingan st.markdown kotak dark blue yang bawahnya...)

    st.markdown("""
<div style="background-color: #2b3d63; padding: 36px; border-radius: 18px; border: 1px solid rgba(200,169,110,0.3); box-shadow: 0 8px 24px rgba(0,0,0,0.15); margin-bottom: 24px; color: #ddddd8; line-height: 1.7;">
<h3 style="font-family: 'DM Serif Display', serif; color: #C8A96E; margin-top: 0; font-size: 1.8rem; font-weight: 400;">Jazirah: Timur Labuhan Kata</h3>
<p style="font-size: 0.95rem;">Aplikasi <strong>Jazirah.id</strong> hadir sebagai platform interaktif yang memadukan teknologi <em>Computer Vision</em> dan <em>Machine Learning</em> untuk memfasilitasi pengenalan gaya lukisan secara otomatis. Lebih dari sekadar alat teknis, platform ini hadir sebagai jembatan inklusif bagi pengguna awam untuk mengenal seni sekaligus bergabung dalam komunitas Jazirah.id guna merayakan dan melestarikan literasi kebudayaan seni.</p>

<h4 style="color: #C8A96E; margin-top: 32px; font-size: 1.1rem; letter-spacing: 0.05em; text-transform: uppercase;">⚙️ Teknologi Klasifikasi</h4>
<p style="font-size: 0.95rem;">Sistem ini menggunakan algoritma pemelajaran mesin <strong>Random Forest Classifier</strong> untuk mengkategorikan lukisan ke dalam 5 gaya utama. Proses pengenalan dilakukan dengan mengekstrak 15 nilai komputasi fitur unik dari setiap gambar yang diunggah, meliputi:</p>

<ul style="font-size: 0.9rem; color: rgba(221,221,216,0.85); background: rgba(0,0,0,0.15); padding: 16px 16px 16px 36px; border-radius: 12px; margin-top: 12px;">
<li style="margin-bottom: 6px;"><strong>Tekstur Global (GLCM):</strong> Contrast, Entropy, Homogeneity, Correlation, Energy.</li>
<li style="margin-bottom: 6px;"><strong>Tekstur Mikro (LBP):</strong> LBP Mean, LBP Std.</li>
<li style="margin-bottom: 6px;"><strong>Struktur Bentuk (HOG):</strong> HOG Mean, HOG Std.</li>
<li style="margin-bottom: 6px;"><strong>Analisis Warna (HSV & RGB):</strong> Mean V, Std Hue, Std Green, Std Red, Colorfulness.</li>
<li><strong>Kepadatan Tepi (Canny):</strong> Edge Density.</li>
</ul>

<div style="margin-top: 36px; padding-top: 24px; border-top: 1px solid rgba(221,221,216,0.1);">
<h4 style="color: #C8A96E; margin-top: 0; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase;">👨‍💻 Tujuan Pengembangan</h4>
<p style="font-size: 0.95rem; margin-bottom: 0;">Dikembangkan untuk mendukung komunitas perayaan literasi, seni, dan kebudayaan Maluku, Indonesia.</p>
</div>
</div>
""", unsafe_allow_html=True)
    
    # ============================================================
# HALAMAN: PANDUAN PENGGUNAAN
# ============================================================
elif pilihan_menu == "panduan":
    st.markdown("""
<div style="margin-bottom:28px;">
  <div class="klas-eyebrow" style="color: #C8A96E;">Instruksi Operasional</div>
  <div class="page-title" style="color: #1e2b45;">Panduan Penggunaan Aplikasi</div>
  <div class="page-sub" style="color: rgba(30,43,69,0.75);">Langkah-langkah mudah untuk menggunakan fitur pendeteksi gaya lukisan otomatis di aplikasi.</div>
</div>
    """, unsafe_allow_html=True)

    st.markdown("""
<div style="background-color: #2b3d63; padding: 36px; border-radius: 18px; border: 1px solid rgba(200,169,110,0.3); box-shadow: 0 8px 24px rgba(0,0,0,0.15); margin-bottom: 24px; color: #ddddd8; line-height: 1.7;">
<h3 style="font-family: 'DM Serif Display', serif; color: #C8A96E; margin-top: 0; font-size: 1.8rem; font-weight: 400;">Langkah-langkah Analisis Lukisan</h3>
<p style="font-size: 0.95rem;">Ikuti petunjuk di bawah ini untuk memulai proses identifikasi gaya lukisan menggunakan model <em>Machine Learning</em>:</p>

<ol style="font-size: 0.95rem; color: rgba(221,221,216,0.85); background: rgba(0,0,0,0.15); padding: 20px 20px 20px 36px; border-radius: 14px; margin-top: 16px; line-height: 1.8;">
<li style="margin-bottom: 10px;">Buka menu <strong>Analisis Lukisan</strong> melalui (<em>sidebar</em>) di sebelah kiri layar Anda.</li>
<li style="margin-bottom: 10px;">Unggah (<em>upload</em>) foto/gambar lukisan yang ingin diuji ke dalam kotak area yang disediakan (sistem mendukung format <strong>JPG, JPEG dan PNG</strong>).</li>
<li style="margin-bottom: 10px;">Pastikan pratinjau (<em>preview</em>) gambar lukisan Anda telah termuat dengan jelas di bagian tengah halaman.</li>
<li style="margin-bottom: 10px;">Klik tombol <strong>Analisis sekarang →</strong> Tunggu beberapa saat karena sistem sedang melakukan analisis lukisan.</li>
<li>Hasil prediksi akan langsung ditampilkan.</li>
</ol>

<div style="margin-top: 32px; padding-top: 20px; border-top: 1px solid rgba(221,221,216,0.15);">
<h4 style="color: #C8A96E; margin-top: 0; font-size: 0.85rem; letter-spacing: 0.1em; text-transform: uppercase;">💡 CATATAN SAMPINGAN </h4>
<p style="font-size: 0.90rem; margin-bottom: 0; color: rgba(221,221,216,0.75);">Pastikan gambar lukisan yang diuji memiliki kualitas pencahayaan yang cukup dan merupakan salah satu dari 5 kelas yang dilatih oleh sistem (Pop Art, Surealisme, Impresionisme, Realisme, atau Futurisme) agar akurasi tebakan optimal.</p>
</div>
</div>
    """, unsafe_allow_html=True)