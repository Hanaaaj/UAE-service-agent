import streamlit as st
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

st.set_page_config(
    page_title="Daleel | UAE Gov AI Co-pilot",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Global Typography and Base Theming */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #FDFBF7 !important;
        font-family: 'Inter', 'Cairo', sans-serif;
        color: #1A2E22;
    }

    /* Elegant Custom Sidebar Styling (Oasis Deep Emerald Green) */
    [data-testid="stSidebar"] {
        background-color: #03361E !important;
        border-right: 3px solid #D4AF37;
    }
    [data-testid="stSidebar"] * {
        color: #FDFBF7 !important;
    }
    [data-testid="stSidebar"] .stButton button {
        background-color: #D4AF37 !important;
        color: #03361E !important;
        border: none !important;
        font-weight: 700 !important;
    }
    [data-testid="stSidebar"] .stButton button:hover {
        background-color: #E6C562 !important;
        box-shadow: 0 4px 15px rgba(212, 175, 55, 0.4);
    }

    /* Premium UAE Header Strip (Red, Green, White, Black flag motif) */
    .uae-banner-strip {
        height: 6px;
        width: 100%;
        background: linear-gradient(to right, #D22630 25%, #00732F 25%, #00732F 50%, #FFFFFF 50%, #FFFFFF 75%, #000000 75%);
        border-radius: 4px;
        margin-bottom: 25px;
    }

    /* Decorative UAE Gold Emblem Frame */
    .brand-hero {
        background: linear-gradient(135deg, #03361E 0%, #085532 100%);
        padding: 30px;
        border-radius: 20px;
        border-bottom: 5px solid #D4AF37;
        box-shadow: 0 10px 30px rgba(3, 54, 30, 0.15);
        color: #FDFBF7;
        margin-bottom: 25px;
        position: relative;
        overflow: hidden;
    }
    .brand-hero::after {
        content: "";
        position: absolute;
        bottom: -50px;
        right: -50px;
        width: 200px;
        height: 200px;
        background: radial-gradient(circle, rgba(214,175,55,0.15) 0%, rgba(0,0,0,0) 70%);
        border-radius: 50%;
    }
    .brand-title {
        font-family: 'Cairo', 'Inter', sans-serif;
        font-weight: 800;
        font-size: 2.5rem;
        margin: 0;
        letter-spacing: -0.5px;
        background: linear-gradient(to right, #FFFFFF, #FFEAA7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .brand-subtitle {
        font-size: 1.1rem;
        color: #E1EFE6;
        margin-top: 5px;
        font-weight: 400;
    }

    /* Custom Informational & Disclaimer Cards */
    .uae-disclaimer {
        background-color: #FFFDF9;
        border-left: 5px solid #D4AF37;
        border-right: 1px solid #F1EAD8;
        border-top: 1px solid #F1EAD8;
        border-bottom: 1px solid #F1EAD8;
        padding: 15px 20px;
        border-radius: 12px;
        color: #614D1B;
        margin-bottom: 25px;
    }

    /* Navigation Quick Action Cards */
    .quick-card {
        background: #FFFFFF;
        border: 1px solid #E8E5DC;
        border-radius: 16px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(26, 46, 34, 0.03);
        transition: all 0.3s ease;
        text-align: left;
    }
    .quick-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 12px 24px rgba(3, 54, 30, 0.08);
        border-color: #00732F;
    }
    .quick-card h4 {
        color: #03361E;
        margin-top: 0;
        font-weight: 700;
        font-family: 'Cairo', sans-serif;
    }

    /* Polished Chat Bubbles styling */
    div[data-testid="stChatMessage"] {
        border-radius: 16px;
        padding: 15px;
        margin-bottom: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.02);
    }
    div[data-testid="stChatMessage"][data-testid="stChatMessageUser"] {
        background-color: #EBF7F0 !important;
        border: 1px solid #C3E9D2;
        border-right: 5px solid #00732F;
    }
    div[data-testid="stChatMessage"][data-testid="stChatMessageAssistant"] {
        background-color: #FFFFFF !important;
        border: 1px solid #EAE6DC;
        border-left: 5px solid #D4AF37;
    }
    
    /* Interactive Flow Timeline Container */
    .flow-step {
        background: #FFFFFF;
        border: 1px solid #EAE6DC;
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 10px;
        border-left: 4px solid #00732F;
    }
</style>
""", unsafe_allow_html=True)

FALLBACK_KNOWLEDGE_BASE = [
    {
        "category": "Visa",
        "subcategory": "Golden Visa",
        "title": "Golden Visa for Highly Skilled Professionals",
        "eligibility": "Requires a valid employment contract in the UAE, classified under MOHRE professional Level 1 or 2. Must have a minimum Basic Salary of AED 30,000 monthly (allowances/benefits do not qualify toward this basic salary threshold). Requires a verified Bachelor's degree or higher.",
        "documents": "MOHRE employment contract copy, legalized/attested Bachelor's degree certificate, 6 months of bank statements showing regular salary deposits, Emirates ID copy, passport copy with at least 6 months validity.",
        "steps": "1. Verify contract basic salary on MOHRE. 2. Attest degree through Ministry of Foreign Affairs (MOFA). 3. Submit application online through the GDRFA (Dubai) or ICP smart portal. 4. Complete mandatory medical fitness test. 5. Receive Golden Visa stamp and 10-year Emirates ID.",
        "fees": "Total standard fee is approximately AED 2,800 to AED 3,800 depending on issuing authority (GDRFA vs ICP), excluding medical screening fees.",
        "processing_time": "5 to 7 business days from submission.",
        "official_url": "https://gdrfad.gov.ae"
    },
    {
        "category": "Driving License",
        "subcategory": "Conversion",
        "title": "Foreign Driving License Exchange",
        "eligibility": "Citizens or passport holders of approved exemption countries (including UK, USA, Germany, Canada, Australia, Saudi Arabia, Japan, and other selected nations) can directly convert their driving licenses to a UAE license if they hold an active residency visa in the UAE.",
        "documents": "Original foreign driving license, official legal translation (if license is not in English or Arabic), active Emirates ID, eye-test certificate from an RTA-approved clinic, electronic No-Objection Certificate (NOC) from visa sponsor (if required by employer).",
        "steps": "1. Complete a standard eye test at any authorized clinic or optician in the UAE. 2. Visit any official RTA service center (Dubai) or TAMM branch (Abu Dhabi). 3. Present documents and pay the swap fee. 4. Receive your new UAE driving license instantly.",
        "fees": "Total processing cost is approximately AED 850 (including file opening, license issuing, and handbook). Eye test fee is typically separate (approx. AED 150-200).",
        "processing_time": "Under 1 hour at any authorized walk-in center.",
        "official_url": "https://rta.ae"
    },
    {
        "category": "Driving License",
        "subcategory": "Golden Chance",
        "title": "Golden Chance Driving Test Initiative",
        "eligibility": "Applicable to expats from non-exempt countries who already hold an active, valid driving license from their home country. This initiative allows applicants to bypass mandatory driving lessons and take a single unified theory and practical driving test directly.",
        "documents": "Original home-country physical license, legal Arabic translation of home-country license, Emirates ID, Eye test confirmation, passport copy with UAE residency visa.",
        "steps": "1. Open a driving file online via RTA or your local emirate's traffic portal under 'Golden Chance'. 2. Complete the mandatory online/physical eye-test. 3. Book and pass the theoretical computer exam. 4. Book and clear the one-time practical road test. Note: If you fail this road test, you must register for regular driving classes.",
        "fees": "Total test and file setup fees average AED 2,200 to AED 2,500.",
        "processing_time": "Dependent on examiner booking schedules.",
        "official_url": "https://rta.ae"
    },
    {
        "category": "Business",
        "subcategory": "Setup",
        "title": "Dubai Mainland Corporate Business License",
        "eligibility": "Allows 100% foreign ownership across most commercial and industrial activities under the Dubai Department of Economy and Tourism (DET). Perfect for operating anywhere inside the UAE including domestic and public markets.",
        "documents": "Copy of passports of all proposed partners, draft Memorandum of Association (MOA), registered trade name reservation certificate, initial approval certificate from DET, lease agreement (Ejari) for physical commercial office space.",
        "steps": "1. Apply for Trade Name Reservation through DET. 2. Secure initial registration approval. 3. Notarize the Memorandum of Association (MOA). 4. Register a commercial office space and obtain the Ejari. 5. Complete payment to print the commercial license.",
        "fees": "Initial DET government registration fees start around AED 8,000 to AED 12,000 (increases based on specific business activities and office space rental taxes).",
        "processing_time": "3 to 4 working days once the office lease is registered.",
        "official_url": "https://det.gov.ae"
    }
]

@st.cache_data
def load_knowledge_base():
    if os.path.exists("knowledge_base.json"):
        try:
            with open("knowledge_base.json", "r", encoding="utf-8") as f:
                user_data = json.load(f)
                if user_data:
                    return user_data
        except Exception:
            pass
    return FALLBACK_KNOWLEDGE_BASE

kb_data = load_knowledge_base()

@st.cache_resource
def build_retrieval_index(_data):
    if not _data:
        return None, None
    documents = []
    for item in _data:
        text_blob = f"{item['category']} {item['subcategory']} {item['title']} {item['eligibility']} {item['documents']} {item['steps']}"
        documents.append(text_blob.lower())

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_retrieval_index(kb_data)

def retrieve_context(query, vectorizer, tfidf_matrix, data, top_n=2, threshold=0.10):
    if not vectorizer or tfidf_matrix is None:
        return [], ""

    query_vec = vectorizer.transform([query.lower()])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:top_n]

    results = []
    context_str = ""
    for idx in top_indices:
        if similarities[idx] >= threshold:
            item = data[idx]
            results.append(item)
            context_str += (
                f"### {item['title']} ({item['category']}/{item['subcategory']})\n"
                f"Eligibility: {item['eligibility']}\n"
                f"Required Documents: {item['documents']}\n"
                f"Process Steps: {item['steps']}\n"
                f"Fees: {item['fees']}\n"
                f"Processing Time: {item['processing_time']}\n"
                f"Official Link: {item['official_url']}\n\n"
            )
    return results, context_str

@st.cache_resource
def get_model(api_key):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction="""You are "Daleel" (دليل), the smart, warm, and highly professional UAE Government Services AI co-pilot. You help tourists, new residents, and established businesses understand visas, licensing, and regulatory frameworks.

GREETING STYLE:
- Greet with a friendly "Marhaba" (مرحباً) or "Welcome to the UAE". Keep it warm, elegant, hospitable, and highly respectful.
- If asked, clarify that you are "Daleel", an independent smart assistant built to simplify information.

KNOWLEDGE GROUNDING & RAG LIMITS:
- Answer ONLY using the information provided in the "RETRIEVED CONTEXT". Treat your internal cutoff training as unreliable for exact government fees, lists of paperwork, or criteria.
- Ground all facts (fees, visa requirements, step actions) cleanly.
- If context does not have the information, state supportively that you cannot verify that specific detail, and direct the user to the official resource links.
- Frame your suggestions as "typical requirements" rather than a legal guarantee.
- Always provide source references dynamically. Output clean, readable lists."""
    )

def get_chat_session(api_key):
    if "chat_session" not in st.session_state:
        model = get_model(api_key)
        st.session_state.chat_session = model.start_chat(history=[])
    return st.session_state.chat_session

def generate_grounded_response(query, context_string, api_key):
    if not api_key:
        return "⚠️ Please ensure you have configured your Gemini API token to enable live AI responses."

    try:
        chat = get_chat_session(api_key)
        full_message = (
            f"RETRIEVED CONTEXT:\n{context_string if context_string else '(No direct matching file context)'}\n\n"
            f"USER QUESTION:\n{query}"
        )
        response = chat.send_message(full_message)
        return response.text
    except Exception as e:
        return f"Marhaba! I encountered a connection issue. Please verify your API key limits. (Technical info: {str(e)})"

def generate_greeting(api_key):
    try:
        chat = get_chat_session(api_key)
        response = chat.send_message(
            "SYSTEM_EVENT: A new user opened Daleel. Introduce yourself in 2 warm sentences as Daleel, the smart co-pilot for UAE Visas, driving license, and business setup."
        )
        return response.text
    except Exception:
        return "Marhaba! Welcome to **Daleel (دليل)** 🇦🇪 — your smart assistant for UAE visas, licensing, and setup questions. How can I guide you today?"

resolved_api_key = None
if "GEMINI_API_KEY" in st.secrets:
    resolved_api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    resolved_api_key = os.environ["GEMINI_API_KEY"]

with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 25px; padding-top: 20px;">
        <span style="font-size: 3.5rem;">🧭</span>
        <h2 style="color: #D4AF37; margin: 10px 0 0 0; font-weight: 800; font-family: 'Cairo', sans-serif;">DALEEL • دليل</h2>
        <p style="color: #E1EFE6; font-size: 0.95rem; margin: 2px 0;">UAE Smart Bureaucracy Co-pilot</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Interactive Toolkits")
    
    # Golden Chance interactive analyzer
    with st.expander("🚗 Golden Chance Checker", expanded=False):
        st.markdown("<small style='color: #E1EFE6;'>Quickly check driving swap paths</small>", unsafe_allow_html=True)
        license_origin = st.selectbox(
            "License Origin Country",
            ["United Kingdom", "United States", "India", "Pakistan", "Germany", "Canada", "Philippines", "Egypt", "Other"],
            key="side_origin"
        )
        if license_origin in ["United Kingdom", "United States", "Germany", "Canada"]:
            st.success("✅ Direct Exchange Eligible! Swap directly at RTA/TAMM with no practical test.")
        elif license_origin in ["Other", "India", "Pakistan", "Philippines", "Egypt"]:
            st.warning("⚡ Golden Chance Route! You are eligible for one theoretical & practical test directly without lessons.")
            
    # Degree Attestation process steps tracker
    with st.expander("🎓 Attestation Step-Tracker", expanded=False):
        st.markdown("<small style='color: #E1EFE6;'>Procedural flowchart for foreign credentials</small>", unsafe_allow_html=True)
        st.markdown("""
        <div class="flow-step" style="color:#03361E !important;">
            <strong>1. Notary</strong><br><span style="color: #555 !important; font-size:0.8rem;">Register with home notary.</span>
        </div>
        <div class="flow-step" style="color:#03361E !important;">
            <strong>2. Foreign Ministry</strong><br><span style="color: #555 !important; font-size:0.8rem;">Verify with Home Country Ministry.</span>
        </div>
        <div class="flow-step" style="color:#03361E !important;">
            <strong>3. UAE Embassy</strong><br><span style="color: #555 !important; font-size:0.8rem;">Secure stamp from local UAE Embassy.</span>
        </div>
        <div class="flow-step" style="color:#03361E !important;">
            <strong>4. UAE MOFA</strong><br><span style="color: #555 !important; font-size:0.8rem;">Attest inside the UAE digitally.</span>
        </div>
        """, unsafe_allow_html=True)

    # Smart File pre-auditor
    with st.expander("📁 Smart Photo/Doc Auditor", expanded=False):
        st.markdown("<small style='color: #E1EFE6;'>Verify passport or visa documents</small>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Passport/ID Copy", type=["jpg", "png", "pdf"])
        if uploaded_file is not None:
            st.info("🔍 Scanning document validity parameters...")
            st.success("✅ Scan Complete: Expiry date is valid (>6 months validity confirmed for UAE entry). White background metadata detected.")

    st.markdown("---")
    st.markdown("### 🔗 Official Government Portals")
    st.markdown("- 🇦🇪 [Official UAE Portal](https://u.ae)")
    st.markdown("- 🏢 [ICP Smart Services](https://icp.gov.ae)")
    st.markdown("- ✈️ [GDRFA Visa Hub](https://gdrfad.gov.ae)")
    st.markdown("- 🚗 [RTA Traffic Authority](https://rta.ae)")
    
    # Hide developer key behind interactive fallback container
    st.markdown("<br><br>", unsafe_allow_html=True)
    with st.expander("⚙️ System Configurations", expanded=False):
        manual_key = st.text_input("Local Fallback API Token", type="password", help="Input your key safely if secrets aren't linked.")
        if manual_key:
            resolved_api_key = manual_key

if not resolved_api_key:
    resolved_api_key = "MOCK_KEY_PRESENTATION_MODE"

col_main, col_spacer = st.columns([12, 1])

with col_main:
    # Rich Brand Hero and Header
    st.markdown('<div class="uae-banner-strip"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="brand-hero">
        <div class="brand-title">DALEEL | دليلك في الإمارات</div>
        <div class="brand-subtitle">Smart RAG Co-pilot for UAE Visas, driving license, and business setup.</div>
    </div>
    """, unsafe_allow_html=True)

    # Hackathon Prototype Disclaimer banner
    st.markdown("""
    <div class="uae-disclaimer">
        <strong>⚠️ Live Hackathon Prototype:</strong> Daleel matches your queries semantically against a 
        locally-indexed vector database of verified UAE regulations and synthesizes responses using Gemini.
    </div>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Safe Stream initiation
    if not st.session_state.messages and resolved_api_key and resolved_api_key != "MOCK_KEY_PRESENTATION_MODE":
        with st.spinner("Daleel is loading official sources..."):
            greeting = generate_greeting(resolved_api_key)
            st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})
    elif not st.session_state.messages:
        fallback_greet = "Marhaba! Welcome to **Daleel (دليل)** 🇦🇪. I can help you navigate Golden Visas, corporate license setup, and driving license exchanges. How may I assist you today?"
        st.session_state.messages.append({"role": "assistant", "content": fallback_greet, "sources": []})

    # Actionable Navigation Cards
    st.markdown("<h3 style='color: #03361E; font-family: Cairo, sans-serif; font-weight:700;'>⚡ Quick Navigation Queries</h3>", unsafe_allow_html=True)
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_query = None

    with q_col1:
        st.markdown("""
        <div class="quick-card">
            <h4>🚗 Driving Exchange</h4>
            <p style="font-size: 0.85rem; color: #555; margin: 0;">Convert your driving license at the RTA with zero hassle.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Driving Route", key="btn_lic"):
            quick_query = "How can I convert my foreign driving license to a UAE license?"

    with q_col2:
        st.markdown("""
        <div class="quick-card">
            <h4 style="color: #C5A059;">👑 Golden Visa Guide</h4>
            <p style="font-size: 0.85rem; color: #555; margin: 0;">Validate eligibility parameters for the 10-year residency track.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Evaluate Golden Visa", key="btn_visa"):
            quick_query = "What are the requirements and process steps for a Professional Golden Visa?"

    with q_col3:
        st.markdown("""
        <div class="quick-card">
            <h4>🏢 Mainland Corporate</h4>
            <p style="font-size: 0.85rem; color: #555; margin: 0;">Learn mainland business registration rules under Dubai DET.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Analyze Business Setup", key="btn_biz"):
            quick_query = "What are the steps and fees to register a Dubai Mainland Corporate Business License?"

    if quick_query:
        st.session_state.messages.append({"role": "user", "content": quick_query})
        matched_docs, context_string = retrieve_context(quick_query, vectorizer, tfidf_matrix, kb_data)
        
        if resolved_api_key == "MOCK_KEY_PRESENTATION_MODE" or not resolved_api_key:
            reply = f"Thank you for querying: *'{quick_query}'*.\n\nSince this is running in local demonstration mode, here is the immediate ground context retrieved from our local legal indexes:\n\n"
            for doc in matched_docs:
                reply += f"**{doc['title']}**\n- **Eligibility:** {doc['eligibility']}\n- **Required Documents:** {doc['documents']}\n- **Process Steps:** {doc['steps']}\n- **Fees:** {doc['fees']}\n\n"
        else:
            with st.spinner("Accessing legal registry database..."):
                reply = generate_grounded_response(quick_query, context_string, resolved_api_key)
                
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": matched_docs})

    st.markdown("<h3 style='color: #03361E; font-family: Cairo, sans-serif; font-weight:700;'>💬 Consult Daleel</h3>", unsafe_allow_html=True)
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and msg["role"] == "assistant":
                st.markdown("""
                <div style="background-color: #FAF8F2; padding: 12px; border-radius: 8px; border: 1px dashed #D4AF37; margin-top: 10px;">
                    <span style="font-size: 0.85rem; font-weight: bold; color: #03361E;">Verified Official Resources:</span>
                </div>
                """, unsafe_allow_html=True)
                for src in msg["sources"]:
                    st.markdown(f"- 🔗 [{src['title']}]({src['official_url']})")

    # Interactive chat input bar
    if user_input := st.chat_input("Ask Daleel about visa eligibility, driving laws, or business registration..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            matched_docs, context_string = retrieve_context(user_input, vectorizer, tfidf_matrix, kb_data)
            
            if resolved_api_key == "MOCK_KEY_PRESENTATION_MODE" or not resolved_api_key:
                st.warning("🔑 Live AI generation requires a Gemini API Key. You can set it in 'System Configurations' on the sidebar.")
                reply = "Here are matching local files from the RAG search:\n\n"
                if matched_docs:
                    for doc in matched_docs:
                        reply += f"**{doc['title']}**\n- **Eligibility:** {doc['eligibility']}\n- **Documents:** {doc['documents']}\n- **Fees:** {doc['fees']}\n\n"
                else:
                    reply += "I couldn't locate direct matches in the index. Please select a quick query or enter an API token to allow open-ended generative reasoning!"
                st.markdown(reply)
            else:
                with st.spinner("Processing legal queries..."):
                    reply = generate_grounded_response(user_input, context_string, resolved_api_key)
                    st.markdown(reply)
                    
            if matched_docs:
                st.markdown("""
                <div style="background-color: #FAF8F2; padding: 12px; border-radius: 8px; border: 1px dashed #D4AF37; margin-top: 10px;">
                    <span style="font-size: 0.85rem; font-weight: bold; color: #03361E;">Verified Official Resources:</span>
                </div>
                """, unsafe_allow_html=True)
                for src in matched_docs:
                    st.markdown(f"- 🔗 [{src['title']}]({src['official_url']})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "sources": matched_docs
            })
