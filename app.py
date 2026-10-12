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
from sklearn.pipeline import make_pipeline

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

# ================== เทรน Machine Learning Model ==================
@st.cache_resource
def train_ml_models():
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

    type_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    type_model.fit(train_texts, train_types)

    prio_model = make_pipeline(TfidfVectorizer(), LogisticRegression())
    prio_model.fit(train_texts, train_priorities)

    return type_model, prio_model

ml_type_clf, ml_prio_clf = train_ml_models()

def detect_province(text):
    for name in sorted(PROVINCES, key=len, reverse=True):
        if name in text:
            return name
    for alias, prov in ALIASES.items():
        if alias in text:
            return prov
    return None

def analyze_with_ml(text):
    pred_type = ml_type_clf.predict([text])[0]
    pred_prio = ml_prio_clf.predict([text])[0]
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

now = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
st.markdown(
    f'<div class="hero"><span class="badge">🤖 ML ENGINE ACTIVE</span><span class="badge">อัปเดตล่าสุด {now}</span>'
    f'<h1>ศูนย์ปฏิบัติการและเตือนภัยพิบัติแห่งชาติ (AI / ML)</h1>'
    f'<p>National Disaster Operations Center — ขับเคลื่อนด้วยโมเดล Machine Learning (TF-IDF & Logistic Regression)</p></div>',
    unsafe_allow_html=True,
)

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

tab1, tab2, tab3, tab4 = st.tabs(["🗺️ ภาพรวมสถานการณ์", "📝 แจ้งเหตุ (ML Predict)", "📋 ฐานข้อมูลเหตุการณ์", "📈 สถิติและวิเคราะห์"])

with tab1:
    c_map, c_feed = st.columns([2, 1])
    with c_map:
        map_obj = build_map(fdf)
        components.html(map_obj.get_root().render(), height=530)
    with c_feed:
        st.markdown("#### เหตุการณ์ล่าสุด")
        for _, r in fdf.iloc[::-1].head(5).iterrows():
            st.markdown(f'<div class="feed"><b>{r["ID"]} • {r["Province"]}</b> — {r["Type"]}<br>{html.escape(r["Text"][:80])}</div>', unsafe_allow_html=True)

with tab2:
    st.markdown("#### แบบฟอร์มรับแจ้งเหตุอัจฉริยะ (Machine Learning)")
    with st.form("report_form", clear_on_submit=True):
        text = st.text_area("รายละเอียดเหตุการณ์", height=130)
        a, b = st.columns(2)
        prov_choice = a.selectbox("จังหวัด", ["ตรวจจับอัตโนมัติจากข้อความ"] + list(PROVINCES))
        reporter = b.text_input("ผู้แจ้ง")
        submitted = st.form_submit_button("🚀 ส่งรายงานให้ ML วิเคราะห์")
    if submitted and text.strip():
        override = None if prov_choice.startswith("ตรวจจับ") else prov_choice
        rec = new_record(text.strip(), reporter.strip(), province=override)
        st.session_state.incidents.append(rec)
        st.session_state.last = rec
        st.rerun()

with tab3:
    st.dataframe(fdf.drop(columns=["lat", "lon", "Image"], errors="ignore").iloc[::-1][list(THAI_COLS)].rename(columns=THAI_COLS), hide_index=True)

with tab4:
    st.markdown("#### สถิติและวิเคราะห์")
    st.metric("เหตุการณ์รวม", len(fdf))
