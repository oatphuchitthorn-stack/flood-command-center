import html
import random
from datetime import datetime

import altair as alt
import folium
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import MakePipeline

# ================== ตั้งค่าหน้าเว็บ ==================
st.set_page_config(
    page_title="DisasterOps ML | ศูนย์ปฏิบัติการภัยพิบัติอัจฉริยะ",
    page_icon="🛡️",
    layout="wide",
)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Sarabun:wght@400;600;700&display=swap');
html, body, .stApp, [class*="css"] {font-family:'Sarabun',sans-serif;}
#MainMenu, footer {visibility:hidden;}
.block-container {padding-top:1.2rem;}
.hero {background:linear-gradient(135deg,#0b3d91 0%,#1565c0 60%,#1e88e5 100%);color:#fff;padding:26px 32px;border-radius:16px;box-shadow:0 8px 24px rgba(11,61,145,.25);margin-bottom:18px;}
.hero h1 {color:#fff;font-size:30px;margin:8px 0 0 0;font-weight:700;}
.hero p {color:#dbe7ff;margin:4px 0 0 0;font-size:15px;}
.badge {display:inline-block;background:rgba(255,255,255,.18);padding:3px 12px;border-radius:20px;font-size:13px;margin-right:6px;}
.kpi {background:#fff;border-radius:14px;padding:16px 20px;box-shadow:0 2px 12px rgba(16,24,40,.06);border:1px solid #e6eaf0;}
.kpi .label {color:#667085;font-size:14px;font-weight:600;}
.kpi .value {font-size:34px;font-weight:700;line-height:1.2;}
.kpi .sub {font-size:13px;color:#98a2b3;}
.alert-bar {background:#fff4f4;border:1px solid #fecdca;border-left:6px solid #d92d20;color:#7a271a;padding:12px 16px;border-radius:10px;margin:14px 0;}
.hotspot-bar {background:#fffbeb;border:1px solid #fedf89;border-left:6px solid #f79009;color:#b54708;padding:12px 16px;border-radius:10px;margin:14px 0;}
.ml-box {background:#f0f4f8;border:1px solid #d0d7de;border-left:6px solid #1565c0;padding:12px 16px;border-radius:10px;margin:14px 0;font-size:14px;color:#1e293b;}
.feed {background:#fff;border:1px solid #e6eaf0;border-radius:12px;padding:14px;margin-bottom:12px;color:#101828;}
.feed .meta {color:#98a2b3;font-size:12px;}
.pill {display:inline-block;padding:2px 10px;border-radius:12px;font-size:12px;font-weight:600;color:#fff;margin-right:4px;}
.pill-gray {background:#eaecf0;color:#344054;}
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {background:#1565c0;color:#fff;border:none;border-radius:10px;padding:8px 18px;font-weight:600;}
.stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {background:#0b3d91;color:#fff;}
.footer {text-align:center;color:#98a2b3;font-size:13px;padding:24px 0 6px 0;}
</style>""", unsafe_allow_html=True)

# ================== ข้อมูลพิกัด 77 จังหวัด ==================
PROVINCES = {
    'กรุงเทพมหานคร': (13.7563, 100.5018), 'กระบี่': (8.0863, 98.9063), 'กาญจนบุรี': (14.0228, 99.5328),
    'กาฬสินธุ์': (16.4322, 103.5068), 'กำแพงเพชร': (16.4828, 99.5226), 'ขอนแก่น': (16.4419, 102.8360),
    'จันทบุรี': (12.6113, 102.1039), 'ฉะเชิงเทรา': (13.6904, 101.0779), 'ชลบุรี': (13.3611, 100.9847),
    'ชัยนาท': (15.1852, 100.1251), 'ชัยภูมิ': (15.8065, 102.0216), 'ชุมพร': (10.4930, 99.1799),
    'เชียงราย': (19.9105, 99.8406), 'เชียงใหม่': (18.7883, 98.9853), 'ตรัง': (7.5563, 99.6111),
    'ตราด': (12.2457, 102.5159), 'ตาก': (16.8840, 99.1259), 'นครนายก': (14.2069, 101.2131),
    'นครปฐม': (13.8197, 100.0620), 'นครพนม': (17.4085, 104.7797), 'นครราชสีมา': (14.9799, 102.0977),
    'นครศรีธรรมราช': (8.4304, 99.9631), 'นครสวรรค์': (15.7001, 100.1237), 'นนทบุรี': (13.8591, 100.5217),
    'นราธิวาส': (6.4255, 101.8253), 'น่าน': (18.7756, 100.7730), 'บึงกาฬ': (18.3646, 103.6523),
    'บุรีรัมย์': (14.9930, 103.1029), 'ปทุมธานี': (14.0207, 100.5250), 'ประจวบคีรีขันธ์': (11.8126, 99.7974),
    'ปราจีนบุรี': (14.0549, 101.3622), 'ปัตตานี': (6.8643, 101.2539), 'พระนครศรีอยุธยา': (14.3532, 100.5689),
    'พะเยา': (19.1966, 99.9008), 'พังงา': (8.4449, 98.5251), 'พัทลุง': (7.6167, 100.0743),
    'พิจิตร': (16.4426, 100.3501), 'พิษณุโลก': (16.8211, 100.2659), 'เพชรบุรี': (13.1122, 99.9397),
    'เพชรบูรณ์': (16.4190, 101.1537), 'แพร่': (18.1446, 100.1402), 'ภูเก็ต': (7.8804, 98.3923),
    'มหาสารคาม': (16.1853, 103.3007), 'มุกดาหาร': (16.5434, 104.7235), 'แม่ฮ่องสอน': (19.3020, 97.9654),
    'ยโสธร': (15.7926, 104.1453), 'ยะลา': (6.5411, 101.2804), 'ร้อยเอ็ด': (16.0538, 103.6520),
    'ระนอง': (9.9658, 98.6340), 'ระยอง': (12.6814, 101.2781), 'ราชบุรี': (13.5283, 99.8134),
    'ลพบุรี': (14.7995, 100.6534), 'ลำปาง': (18.2888, 99.5086), 'ลำพูน': (18.5743, 99.0087),
    'เลย': (17.4860, 101.7223), 'ศรีสะเกษ': (15.1186, 104.3220), 'สกลนคร': (17.1640, 104.1486),
    'สงขลา': (7.1756, 100.6146), 'สตูล': (6.6111, 100.0673), 'สมุทรปราการ': (13.5991, 100.5998),
    'สมุทรสงคราม': (13.4098, 99.9989), 'สมุทรสาคร': (13.5475, 100.2744), 'สระแก้ว': (13.8010, 102.0732),
    'สระบุรี': (14.5289, 100.9101), 'สิงห์บุรี': (14.8936, 100.4025), 'สุโขทัย': (17.0056, 99.8264),
    'สุพรรณบุรี': (14.4745, 100.1177), 'สุราษฎร์ธานี': (9.1382, 99.3217), 'สุรินทร์': (14.8818, 103.4936),
    'หนองคาย': (17.8783, 102.7414), 'หนองบัวลำภู': (17.2045, 102.4430), 'อ่างทอง': (14.5896, 100.4550),
    'อำนาจเจริญ': (15.8573, 104.6258), 'อุดรธานี': (17.4157, 102.7859), 'อุตรดิตถ์': (17.6200, 100.0990),
    'อุทัยธานี': (15.3835, 100.0246), 'อุบลราชธานี': (15.2289, 104.8563),
}
ALIASES = {"กทม": "กรุงเทพมหานคร", "กรุงเทพ": "กรุงเทพมหานคร", "โคราช": "นครราชสีมา",
           "อยุธยา": "พระนครศรีอยุธยา", "หาดใหญ่": "สงขลา", "พัทยา": "ชลบุรี",
           "เกาะสมุย": "สุราษฎร์ธานี", "หัวหิน": "ประจวบคีรีขันธ์"}

PRIORITY_HEX = {"วิกฤต": "#d92d20", "เร่งด่วน": "#f79009", "เฝ้าระวัง": "#1570ef"}
STATUSES = ["รอรับเรื่อง", "กำลังช่วยเหลือ", "ดำเนินการเสร็จสิ้น"]
COLUMNS = ["ID", "Time", "Province", "Type", "Priority", "Needs", "Status", "Reporter", "Text", "lat", "lon", "Image"]
THAI_COLS = {"ID": "รหัสเหตุ", "Time": "เวลา", "Province": "จังหวัด", "Type": "ประเภทภัย",
             "Priority": "ระดับ", "Needs": "ความต้องการ", "Status": "สถานะ",
             "Reporter": "ผู้แจ้ง", "Text": "รายละเอียด"}


# ================== เทรน Machine Learning Model (Text Classification) ==================
@st.cache_resource
def train_ml_models():
    # ชุดข้อมูลตัวอย่างสำหรับสอนโมเดล (Training Data)
    train_texts = [
        "ช่วยด้วย น้ำท่วมสูงถึงชั้น 2 ติดอยู่บนหลังคา ต้องการเรืออพยพด่วน",
        "น้ำป่าไหลหลากเข้าหมู่บ้าน คนติดค้าง ผู้ป่วยติดเตียง",
        "พายุลมแรง พัดหลังคาบ้านพังเสียหาย ต้นไม้ล้มทับเส้นทาง",
        "วาตภัยพายุฤดูร้อน เสาไฟฟ้าล้ม",
        "ไฟไหม้บ้านเรือนประชาชน เพลิงลุกไหม้รุนแรง ลามเข้าป่า",
        "ไฟป่าลุกลามใกล้ชุมชน ควันหนาแน่น",
        "ดินถล่มปิดทับถนนดินสไลด์ลงมาทับบ้านเรือน",
        "โคลนถล่มจากภูเขา หลังฝนตกหนักต่อเนื่อง",
        "ฝนตกหนักน้ำท่วมขังรอการระบาย การจราจรติดขัด",
        "เฝ้าระวังระดับน้ำในแม่น้ำเพิ่มสูงขึ้น",
        "ไม่มีอาหารและน้ำดื่ม ขาดแคลนเสบียง",
        "ต้องการทีมแพทย์และยาช่วยเหลือผู้ป่วยฉุกเฉิน",
    ]
    train_types = [
        "อุทกภัย / น้ำท่วม", "อุทกภัย / น้ำท่วม",
        "วาตภัย / พายุ", "วาตภัย / พายุ",
        "อัคคีภัย / ไฟป่า", "อัคคีภัย / ไฟป่า",
        "ดินถล่ม", "ดินถล่ม",
        "อุทกภัย / น้ำท่วม", "อุทกภัย / น้ำท่วม",
        "อื่น ๆ / ทั่วไป", "อื่น ๆ / ทั่วไป"
    ]
    train_priorities = [
        "วิกฤต", "วิกฤต",
        "เร่งด่วน", "เร่งด่วน",
        "วิกฤต", "เร่งด่วน",
        "วิกฤต", "เร่งด่วน",
        "เฝ้าระวัง", "เฝ้าระวัง",
        "เร่งด่วน", "วิกฤต"
    ]

    # สร้าง Pipeline แยกประเภทภัย (Type Classifier)
    type_model = MakePipeline(TfidfVectorizer(), LogisticRegression())
    type_model.fit(train_texts, train_types)

    # สร้าง Pipeline จัดระดับความรุนแรง (Priority Classifier)
    prio_model = MakePipeline(TfidfVectorizer(), LogisticRegression())
    prio_model.fit(train_texts, train_priorities)

    return type_model, prio_model

ml_type_clf, ml_prio_clf = train_ml_models()


# ================== ฟังก์ชันวิเคราะห์ด้วย ML ==================
def detect_province(text):
    for name in sorted(PROVINCES, key=len, reverse=True):
        if name in text:
            return name
    for alias, prov in ALIASES.items():
        if alias in text:
            return prov
    return None

def analyze_with_ml(text):
    # ให้ Machine Learning ทำนายประเภทภัยและความรุนแรง
    pred_type = ml_type_clf.predict([text])[0]
    pred_prio = ml_prio_clf.predict([text])[0]

    # วิเคราะห์ความต้องการ (Needs) จาก Keyword พื้นฐานประกอบ
    t = text.lower()
    needs_rules = {
        "เรืออพยพ": ["เรือ", "อพยพ", "ย้ายออก", "หลังคา"],
        "อาหารและน้ำดื่ม": ["อาหาร", "ข้าว", "น้ำดื่ม", "หิว", "เสบียง"],
        "แพทย์และยา": ["ป่วย", "ยา", "หมอ", "ติดเตียง", "บาดเจ็บ"],
        "ถุงยังชีพ": ["ถุงยังชีพ", "เดือดร้อน", "เสียหาย"],
        "ที่พักพิง": ["ที่พัก", "ศูนย์พักพิง", "บ้านพัง"],
    }
    needs = [n for n, kws in needs_rules.items() if any(k in t for k in kws)]
    
    return pred_type, pred_prio, (", ".join(needs) if needs else "เฝ้าระวังทั่วไป")

def new_record(text, reporter="ประชาชนทั่วไป", province=None, status="รอรับเรื่อง", image=None):
    dtype, priority, needs = analyze_with_ml(text)
    detected = province or detect_province(text)
    prov = detected or "กรุงเทพมหานคร"
    lat, lon = PROVINCES[prov]
    st.session_state.counter += 1
    return {
        "ID": f"INC-{st.session_state.counter:04d}",
        "Time": datetime.now().strftime("%d/%m/%Y %H:%M"),
        "Province": prov, "Type": dtype, "Priority": priority, "Needs": needs,
        "Status": status, "Reporter": reporter or "ประชาชนทั่วไป", "Text": text,
        "lat": lat + random.uniform(-0.04, 0.04), "lon": lon + random.uniform(-0.04, 0.04),
        "Image": image,
        "_detected": detected is not None,
    }


# ================== ฟังก์ชันสร้างแผนที่ ==================
def build_map(data):
    m = folium.Map(location=[13.0, 101.0], zoom_start=6, tiles="OpenStreetMap")
    for _, r in data.iterrows():
        color = PRIORITY_HEX[r["Priority"]]
        popup = (
            f"<div style='font-family:sans-serif;width:250px'>"
            f"<b style='color:{color}'>{r['ID']} • {r['Priority']} (ML Predicted)</b><br>"
            f"<b>จังหวัด:</b> {r['Province']}<br>"
            f"<b>ประเภท:</b> {r['Type']}<br>"
            f"<b>ความต้องการ:</b> {r['Needs']}<br>"
            f"<b>สถานะ:</b> {r['Status']}<br>"
            f"<b>รายละเอียด:</b> {html.escape(r['Text'])}<br>"
            f"<small style='color:gray'>{r['Time']} • {html.escape(r['Reporter'])}</small></div>"
        )
        folium.CircleMarker(
            location=[r["lat"], r["lon"]],
            radius=11, color="#ffffff", weight=2, fill=True,
            fill_color=color, fill_opacity=0.9,
            popup=folium.Popup(popup, max_width=320),
            tooltip=f"{r['ID']} • {r['Province']} ({r['Priority']})",
        ).add_to(m)
    return m


# ================== ข้อมูลตัวอย่าง ==================
if "incidents" not in st.session_state:
    st.session_state.counter = 0
    st.session_state.incidents = []
    samples = [
        ("ช่วยด้วย น้ำท่วมสูงถึงชั้น 2 อำเภอเมือง จังหวัดเชียงใหม่ มีผู้ป่วยติดเตียง ต้องการเรืออพยพ", "อปพร. เชียงใหม่", "กำลังช่วยเหลือ", None),
        ("น้ำป่าไหลหลากเข้าหมู่บ้าน อำเภอหาดใหญ่ จังหวัดสงขลา ชาวบ้านต้องการอาหารและน้ำดื่มด่วน", "ประชาชนทั่วไป", "รอรับเรื่อง", None),
        ("พายุลมแรง บ้านเรือนเสียหายหลายหลัง จังหวัดอุบลราชธานี", "อบต. ในพื้นที่", "ดำเนินการเสร็จสิ้น", None),
        ("เฝ้าระวังดินสไลด์หลังฝนตกต่อเนื่อง จังหวัดน่าน", "สำนักงาน ปภ. น่าน", "รอรับเรื่อง", None),
        ("ฝนตกหนัก น้ำท่วมขังถนนหลายสายในกรุงเทพมหานคร", "ศูนย์ข่าว กทม.", "กำลังช่วยเหลือ", None),
    ]
    for s_text, s_reporter, s_status, s_img in samples:
        st.session_state.incidents.append(new_record(s_text, s_reporter, status=s_status, image=s_img))

df = pd.DataFrame(st.session_state.incidents, columns=COLUMNS)

# ================== Sidebar ==================
st.sidebar.markdown("## 🛡️ DisasterOps ML")
st.sidebar.caption("ระบบจำแนกภัยพิบัติด้วย Machine Learning")
st.sidebar.divider()
f_priority = st.sidebar.multiselect("ระดับความรุนแรง", list(PRIORITY_HEX), default=list(PRIORITY_HEX))
f_status = st.sidebar.multiselect("สถานะการดำเนินงาน", STATUSES, default=STATUSES)
f_province = st.sidebar.selectbox("จังหวัด", ["ทุกจังหวัด"] + list(PROVINCES))
f_keyword = st.sidebar.text_input("ค้นหาคำสำคัญ")
st.sidebar.divider()
if st.sidebar.button("ล้างข้อมูลทั้งหมด"):
    st.session_state.incidents = []
    st.session_state.counter = 0
    st.session_state.pop("last", None)
    st.rerun()

fdf = df[df["Priority"].isin(f_priority) & df["Status"].isin(f_status)]
if f_province != "ทุกจังหวัด":
    fdf = fdf[fdf["Province"] == f_province]
if f_keyword:
    fdf = fdf[fdf["Text"].str.contains(f_keyword, case=False, na=False)]

# ================== Header ==================
now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
st.markdown(
    f'<div class="hero"><span class="badge">🤖 ML ENGINE ACTIVE</span><span class="badge">อัปเดตล่าสุด {now}</span>'
    f'<h1>ศูนย์ปฏิบัติการและเตือนภัยพิบัติแห่งชาติ (AI / ML)</h1>'
    f'<p>National Disaster Operations Center — ขับเคลื่อนด้วยโมเดล Machine Learning (TF-IDF & Logistic Regression)</p></div>',
    unsafe_allow_html=True,
)

# ================== KPI ==================
total = len(df)
crit = int((df["Priority"] == "วิกฤต").sum())
active = int((df["Status"] == "กำลังช่วยเหลือ").sum())
done = int((df["Status"] == "ดำเนินการเสร็จสิ้น").sum())
pct = (done / total * 100) if total else 0

def kpi(col, label, value, sub, color):
    col.markdown(
        f'<div class="kpi" style="border-top:4px solid {color}"><div class="label">{label}</div>'
        f'<div class="value" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>',
        unsafe_allow_html=True,
    )

k1, k2, k3, k4 = st.columns(4)
kpi(k1, "เหตุการณ์ทั้งหมด", total, f"ใน {df['Province'].nunique()} จังหวัด", "#1565c0")
kpi(k2, "ระดับวิกฤต (ML)", crit, "ต้องช่วยเหลือทันที", "#d92d20")
kpi(k3, "กำลังช่วยเหลือ", active, "เจ้าหน้าที่ลงพื้นที่แล้ว", "#f79009")
kpi(k4, "ดำเนินการเสร็จสิ้น", done, f"คิดเป็น {pct:.0f}% ของทั้งหมด", "#12b76a")

pending_crit = df[(df["Priority"] == "วิกฤต") & (df["Status"] != "ดำเนินการเสร็จสิ้น")]
if not pending_crit.empty:
    last = pending_crit.iloc[-1]
    st.markdown(
        f'<div class="alert-bar"><b>⚠ แจ้งเตือนเหตุวิกฤต {len(pending_crit)} รายการที่ยังไม่เสร็จสิ้น</b> — '
        f'ล่าสุด {last["ID"]} จังหวัด{last["Province"]}: {html.escape(last["Text"])}</div>',
        unsafe_allow_html=True,
    )

hotspots = df[df["Priority"] == "วิกฤต"]["Province"].value_counts()
if not hotspots.empty and hotspots.iloc[0] >= 1:
    top_hotspot = hotspots.index[0]
    st.markdown(
        f'<div class="hotspot-bar"><b>🚨 ระบบ Machine Learning วิเคราะห์พบพื้นที่เสี่ยงสูงสุด (Hotspot):</b> จังหวัด<b>{top_hotspot}</b> มีรายงานเคสวิกฤตสะสมสูงสุด</div>',
        unsafe_allow_html=True,
    )

# ================== Tabs ==================
tab1, tab2, tab3, tab4 = st.tabs(["🗺️ ภาพรวมสถานการณ์", "📝 แจ้งเหตุ (ML Predict)", "📋 ฐานข้อมูลเหตุการณ์", "📈 สถิติและวิเคราะห์"])

# ---------- Tab 1: แผนที่ + ฟีด ----------
with tab1:
    c_map, c_feed = st.columns([2, 1])

    with c_map:
        map_obj = build_map(fdf)
        components.html(map_obj.get_root().render(), height=530)
        legend = " ".join(f'<span class="pill" style="background:{c}">{p}</span>' for p, c in PRIORITY_HEX.items())
        st.markdown(f"คำอธิบายสี: {legend} &nbsp; <small>คลิกจุดบนแผนที่เพื่อดูรายละเอียด</small>", unsafe_allow_html=True)

    with c_feed:
        st.markdown("#### เหตุการณ์ล่าสุด")
        if fdf.empty:
            st.info("ไม่พบเหตุการณ์ตามเงื่อนไขที่เลือก")
        for _, r in fdf.iloc[::-1].head(5).iterrows():
            img_tag = f'<br><img src="data:image/jpeg;base64,{r["Image"]}" style="width:100%;max-height:120px;object-fit:cover;border-radius:8px;margin-top:6px;">' if r["Image"] else ""
            st.markdown(
                f'<div class="feed"><span class="pill" style="background:{PRIORITY_HEX[r["Priority"]]}">{r["Priority"]}</span>'
                f'<span class="pill pill-gray">{r["Status"]}</span><br>'
                f'<b>{r["ID"]} • {r["Province"]}</b> — {r["Type"]}<br>'
                f'<span style="font-size:14px">{html.escape(r["Text"][:80])}</span>{img_tag}<br>'
                f'<span class="meta">{r["Time"]} • {html.escape(r["Reporter"])}</span></div>',
                unsafe_allow_html=True,
            )

# ---------- Tab 2: แจ้งเหตุด้วย ML ----------
with tab2:
    st.markdown("#### แบบฟอร์มรับแจ้งเหตุอัจฉริยะ (ขับเคลื่อนด้วย Machine Learning)")
    st.markdown(
        '<div class="ml-box"><b>💡 การทำงานของระบบ Machine Learning:</b> ข้อความที่คุณพิมพ์จะถูกส่งเข้าโมเดล <b>TF-IDF + Logistic Regression</b> เพื่อทำนายประเภทภัยและความรุนแรงโดยอัตโนมัติทันที</div>',
        unsafe_allow_html=True,
    )
    
    with st.form("report_form", clear_on_submit=True):
        text = st.text_area("รายละเอียดเหตุการณ์", height=130,
                            placeholder="เช่น ด่วน! น้ำท่วมสูง อำเภอเมือง จังหวัดขอนแก่น มีผู้สูงอายุติดอยู่ในบ้าน ต้องการเรืออพยพ")
        a, b = st.columns(2)
        prov_choice = a.selectbox("จังหวัด (เลือกเพื่อให้ระบบล็อกตำแหน่ง)", ["ตรวจจับอัตโนมัติจากข้อความ"] + list(PROVINCES))
        reporter = b.text_input("หน่วยงาน / ผู้แจ้ง (ไม่บังคับ)")
        uploaded_file = st.file_uploader("แนบรูปภาพหลักฐานความเสียหาย (ถ้ามี)", type=["jpg", "jpeg", "png"])
        submitted = st.form_submit_button("🚀 ส่งรายงานให้ ML วิเคราะห์")

    if submitted:
        if not text.strip():
            st.warning("กรุณากรอกรายละเอียดเหตุการณ์")
        else:
            import base64
            img_b64 = None
            if uploaded_file is not None:
                img_bytes = uploaded_file.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode("utf-8")

            override = None if prov_choice.startswith("ตรวจจับ") else prov_choice
            rec = new_record(text.strip(), reporter.strip(), province=override, image=img_b64)
            st.session_state.incidents.append(rec)
            st.session_state.last = rec
            st.rerun()

    if "last" in st.session_state:
        r = st.session_state.last
        st.success(f"บันทึกเหตุ {r['ID']} สำเร็จ! (ผ่านการประมวลผลด้วย Machine Learning)")
        
        if r["Priority"] == "วิกฤต":
            if st.button("📲 จำลองส่ง SMS และ LINE Alert ไปยังทีมกู้ภัยด่วน"):
                st.toast(f"ส่งการแจ้งเตือนเคส {r['ID']} ไปยังเครือข่ายกู้ภัยและผู้ว่าราชการจังหวัด {r['Province']} สำเร็จ!", icon="🚨")
        
        x1, x2, x3, x4 = st.columns(4)
        x1.metric("จังหวัด", r["Province"])
        x2.metric("ประเภทภัย (ML Predicted)", r["Type"])
        x3.metric("ระดับความรุนแรง (ML Predicted)", r["Priority"])
        x4.metric("ความต้องการ", r["Needs"].split(",")[0])
        if not r["_detected"]:
            st.warning("ไม่พบชื่อจังหวัดในข้อความ ระบบจึงปักหมุดที่กรุงเทพมหานครแทน")

# ---------- Tab 3: ฐานข้อมูล ----------
with tab3:
    h1, h2 = st.columns([4, 1])
    h1.markdown(f"#### ฐานข้อมูลเหตุการณ์ ({len(fdf)} รายการ)")
    h2.download_button(
        "ดาวน์โหลด CSV",
        fdf[list(THAI_COLS)].rename(columns=THAI_COLS).to_csv(index=False).encode("utf-8-sig"),
        file_name=f"disaster_ml_report_{datetime.now():%Y%m%d_%H%M}.csv",
        mime="text/csv",
    )
    st.dataframe(fdf.drop(columns=["lat", "lon", "Image"], errors="ignore").iloc[::-1][list(THAI_COLS)].rename(columns=THAI_COLS), hide_index=True)

    if not df.empty:
        st.markdown("#### อัปเดตสถานะการช่วยเหลือ")
        s1, s2, s3 = st.columns([2, 2, 1])
        sel_id = s1.selectbox("รหัสเหตุ", df["ID"].iloc[::-1].tolist())
        new_status = s2.selectbox("เปลี่ยนสถานะเป็น", STATUSES)
        s3.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if s3.button("บันทึกสถานะ"):
            for inc in st.session_state.incidents:
                if inc["ID"] == sel_id:
                    inc["Status"] = new_status
            st.rerun()

# ---------- Tab 4: สถิติและวิเคราะห์ ----------
STATUS_HEX = {"รอรับเรื่อง": "#98a2b3", "กำลังช่วยเหลือ": "#f79009", "ดำเนินการเสร็จสิ้น": "#12b76a"}
TYPE_HEX = {"อุทกภัย / น้ำท่วม": "#1570ef", "วาตภัย / พายุ": "#7a5af8", "ดินถล่ม": "#b54708",
            "อัคคีภัย / ไฟป่า": "#d92d20", "อื่น ๆ / ทั่วไป": "#667085"}
PRIO_SCALE = alt.Scale(domain=list(PRIORITY_HEX), range=list(PRIORITY_HEX.values()))
STATUS_SCALE = alt.Scale(domain=list(STATUS_HEX), range=list(STATUS_HEX.values()))
TYPE_SCALE = alt.Scale(domain=list(TYPE_HEX), range=list(TYPE_HEX.values()))

def style_chart(chart, height=300):
    return (
        chart.properties(width="container", height=height)
        .configure(font="Sarabun")
        .configure_view(strokeWidth=0)
        .configure_axis(grid=False, labelFontSize=13, titleFontSize=13,
                        labelColor="#475467", titleColor="#475467", domainColor="#e4e7ec")
        .configure_legend(labelFontSize=13, orient="bottom")
    )

def donut_chart(data, field, scale):
    counts = data[field].value_counts().reset_index()
    counts.columns = ["Label", "Count"]
    total_n = int(counts["Count"].sum())
    arc = alt.Chart(counts).mark_arc(innerRadius=75, outerRadius=115, cornerRadius=6, padAngle=0.02).encode(
        theta=alt.Theta("Count:Q", stack=True),
        color=alt.Color("Label:N", scale=scale, legend=alt.Legend(title=None)),
        tooltip=[alt.Tooltip("Label:N", title="หมวด"), alt.Tooltip("Count:Q", title="จำนวน")],
    )
    return style_chart(arc, 320)

def hbar_chart(counts, scale=None, color="#1565c0"):
    order = counts["Label"].tolist()
    max_v = max(int(counts["Count"].max()), 1)
    color_enc = alt.Color("Label:N", scale=scale, legend=None) if scale is not None else alt.value(color)
    base = alt.Chart(counts).encode(
        y=alt.Y("Label:N", sort=order, title=None),
        x=alt.X("Count:Q", title=None, axis=None, scale=alt.Scale(domain=[0, max_v * 1.2])),
    )
    bars = base.mark_bar(cornerRadiusEnd=6, height=22).encode(
        color=color_enc,
        tooltip=[alt.Tooltip("Label:N", title="หมวด"), alt.Tooltip("Count:Q", title="จำนวน")],
    )
    labels = base.mark_text(align="left", dx=6, fontSize=13, fontWeight="bold", color="#344054").encode(text="Count:Q")
    return style_chart(bars + labels, max(160, 44 * len(counts)))

with tab4:
    if fdf.empty:
        st.info("ยังไม่มีข้อมูลสำหรับแสดงสถิติ")
    else:
        st.markdown("#### ภาพรวมเชิงวิเคราะห์ด้วย Machine Learning Insights")
        st.caption("สถิติประมวลผลจากผลลัพธ์การจำแนกประเภทของโมเดล ML")

        f_total = len(fdf)
        f_done = int((fdf["Status"] == "ดำเนินการเสร็จสิ้น").sum())
        f_rate = f_done / f_total * 100 if f_total else 0
        f_crit_pending = int(((fdf["Priority"] == "วิกฤต") & (fdf["Status"] != "ดำเนินการเสร็จสิ้น")).sum())
        top_prov = fdf["Province"].value_counts()
        top_type = fdf["Type"].value_counts()

        s1, s2, s3, s4 = st.columns(4)
        kpi(s1, "อัตราดำเนินการสำเร็จ", f"{f_rate:.0f}%", f"เสร็จแล้ว {f_done} จาก {f_total} เหตุ", "#12b76a")
        kpi(s2, "วิกฤตที่ยังค้างอยู่", f_crit_pending, "ต้องเร่งจัดสรรทีมช่วยเหลือ", "#d92d20")
        kpi(s3, "จังหวัดที่มีเหตุมากสุด", top_prov.index[0], f"{top_prov.iloc[0]} เหตุ", "#1565c0")
        kpi(s4, "ภัยที่ ML ทำนายบ่อยสุด", top_type.index[0].split(" /")[0], f"{top_type.iloc[0]} เหตุ", "#7a5af8")

        st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
        st.progress(f_rate / 100, text=f"ความคืบหน้าการช่วยเหลือโดยรวม {f_rate:.0f}%")

        r1a, r1b = st.columns(2)
        with r1a:
            with st.container(border=True):
                st.markdown("**สัดส่วนระดับความรุนแรง (ML Classification)**")
                st.altair_chart(donut_chart(fdf, "Priority", PRIO_SCALE))
        with r1b:
            with st.container(border=True):
                st.markdown("**สถานะการดำเนินงาน**")
                st.altair_chart(donut_chart(fdf, "Status", STATUS_SCALE))

        r2a, r2b = st.columns([3, 2])
        with r2a:
            with st.container(border=True):
                st.markdown("**10 จังหวัดที่มีเหตุมากที่สุด**")
                top10 = top_prov.head(10).index.tolist()
                pc = (fdf[fdf["Province"].isin(top10)]
                      .groupby(["Province", "Priority"]).size().reset_index(name="Count"))
                prov_chart = alt.Chart(pc).mark_bar(height=22).encode(
                    y=alt.Y("Province:N", sort=top10, title=None),
                    x=alt.X("Count:Q", title="จำนวนเหตุ", axis=alt.Axis(tickMinStep=1, grid=True, gridColor="#f2f4f7")),
                    color=alt.Color("Priority:N", scale=PRIO_SCALE, legend=alt.Legend(title=None)),
                    tooltip=[alt.Tooltip("Province:N", title="จังหวัด"),
                             alt.Tooltip("Priority:N", title="ระดับ"),
                             alt.Tooltip("Count:Q", title="จำนวน")],
                )
                st.altair_chart(style_chart(prov_chart, max(200, 44 * len(top10))))
        with r2b:
            with st.container(border=True):
                st.markdown("**ประเภทภัยพิบัติ (ML Predicted)**")
                tc = top_type.reset_index()
                tc.columns = ["Label", "Count"]
                st.altair_chart(hbar_chart(tc, scale=TYPE_SCALE))

        r3a, r3b = st.columns([2, 3])
        with r3a:
            with st.container(border=True):
                st.markdown("**ความต้องการที่ถูกร้องขอ**")
                needs_s = fdf["Needs"].str.split(", ").explode()
                needs_s = needs_s[needs_s != "เฝ้าระวังทั่วไป"]
                if needs_s.empty:
                    st.info("ยังไม่มีการร้องขอความช่วยเหลือเฉพาะด้าน")
                else:
                    nc = needs_s.value_counts().reset_index()
                    nc.columns = ["Label", "Count"]
                    st.altair_chart(hbar_chart(nc, color="#0b3d91"))
        with r3b:
            with st.container(border=True):
                st.markdown("**สรุปผลการดำเนินงานรายจังหวัด**")
                summary = fdf.groupby("Province").agg(
                    Total=("ID", "count"),
                    Critical=("Priority", lambda s: int((s == "วิกฤต").sum())),
                    Done=("Status", lambda s: int((s == "ดำเนินการเสร็จสิ้น").sum())),
                ).reset_index()
                summary["Rate"] = summary["Done"] / summary["Total"] * 100
                summary = summary.sort_values(["Total", "Critical"], ascending=False)
                st.dataframe(
                    summary,
                    hide_index=True,
                    column_config={
                        "Province": "จังหวัด",
                        "Total": st.column_config.NumberColumn("เหตุทั้งหมด"),
                        "Critical": st.column_config.NumberColumn("วิกฤต"),
                        "Done": st.column_config.NumberColumn("เสร็จสิ้น"),
                        "Rate": st.column_config.ProgressColumn(
                            "อัตราสำเร็จ", format="%.0f%%", min_value=0, max_value=100),
                    },
                )

st.markdown('<div class="footer">© 2026 DisasterOps ML Thailand • ระบบปฏิบัติการอัจฉริยะขับเคลื่อนด้วย Machine Learning</div>',
            unsafe_allow_html=True)
