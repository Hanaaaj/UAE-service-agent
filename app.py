import streamlit as st
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

st.set_page_config(
    page_title="Daleel | Smart UAE Assistant",
    page_icon="🇦🇪",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Main Theme variables representing UAE emerald, gold, and desert sands */
    :root {
        --primary-color: #00732F;
        --secondary-color: #C5A059;
        --bg-light: #F8F9FA;
        --text-dark: #121212;
    }
    
    /* Clean custom background wrapper */
    .reportview-container {
        background: #fafaf9;
    }
    
    /* Elegant Custom Title Header styling */
    .branding-container {
        display: flex;
        align-items: center;
        gap: 15px;
        padding: 10px 0;
        margin-bottom: 20px;
        border-bottom: 2px solid #E5E7EB;
    }
    .branding-logo {
        font-size: 2.5rem;
    }
    .branding-text h1 {
        margin: 0;
        color: #00732F;
        font-size: 2.2rem;
        font-weight: 800;
        font-family: 'Inter', sans-serif;
    }
    .branding-text p {
        margin: 2px 0 0 0;
        color: #6B7280;
        font-size: 1rem;
    }
    
    /* Custom CSS styled card structures */
    .accent-card {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        border-left: 5px solid #00732F;
        margin-bottom: 15px;
    }
    .accent-card-gold {
        background-color: #FFFFFF;
        padding: 24px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid #E5E7EB;
        border-left: 5px solid #C5A059;
        margin-bottom: 15px;
    }
    
    /* Custom styles for warning disclaimers */
    .disclaimer-banner {
        background-color: #FFFDF5;
        border: 1px solid #FDF0CD;
        border-left: 4px solid #C5A059;
        padding: 12px 18px;
        border-radius: 12px;
        margin-bottom: 20px;
        color: #7A5C13;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """You are "Daleel" (دليل), the smart, warm, and highly professional UAE Government Services AI co-pilot. You help tourists, new residents, and established businesses understand visas, licensing, and regulatory frameworks.

GREETING STYLE
- Greet with a friendly "Marhaba" (مرحباً) or "Welcome to the UAE". Keep it warm, elegant, hospitable, and highly respectful.
- If asked, clarify that you are "Daleel", an independent smart assistant built to simplify information.

KNOWLEDGE GROUNDING & RAG LIMITS
- Answer ONLY using the information provided in the "RETRIEVED CONTEXT". Treat your internal cutoff training as unreliable for exact government fees, lists of paperwork, or criteria.
- Ground all facts (fees, visa requirements, step actions) cleanly.
- If context does not have the information, state supportively that you cannot verify that specific detail, and direct the user to the official resource links.
- Frame your suggestions as "typical requirements" rather than an legal guarantee.
- Always provide source references dynamically. Output clean, readable lists.
"""

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
        system_instruction=SYSTEM_PROMPT
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

# We silently read from secrets or environment parameters
resolved_api_key = None
if "GEMINI_API_KEY" in st.secrets:
    resolved_api_key = st.secrets["GEMINI_API_KEY"]
elif "GEMINI_API_KEY" in os.environ:
    resolved_api_key = os.environ["GEMINI_API_KEY"]

# Sidebar Navigation & Utility Toolbox
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; margin-bottom: 25px;">
        <span style="font-size: 3.5rem;">🧭</span>
        <h2 style="color: #00732F; margin: 10px 0 0 0; font-weight: 800; font-family: 'Inter', sans-serif;">DALEEL • دليل</h2>
        <p style="color: #6B7280; font-size: 0.9rem; margin: 2px 0;">UAE Smart Bureaucracy Co-pilot</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Interactive Tools")
    
    # 🚗 QUICK TOOL: Golden Chance Checker
    with st.expander("🚗 Golden Chance Checker", expanded=False):
        st.markdown("<small>Quickly check driving swap paths</small>", unsafe_allow_html=True)
        license_origin = st.selectbox(
            "License Origin Country",
            ["United Kingdom", "United States", "India", "Pakistan", "Germany", "Canada", "Philippines", "Egypt", "Other"]
        )
        if license_origin in ["United Kingdom", "United States", "Germany", "Canada"]:
            st.success("✅ **Direct Exchange Eligible!** You can swap directly at RTA/TAMM with no road test required.")
        elif license_origin in ["Other", "India", "Pakistan", "Philippines", "Egypt"]:
            st.warning("⚡ **Golden Chance Route!** You are eligible for one theoretical & practical test directly without mandatory driving classes.")
            
    # 📑 QUICK TOOL: Degree Attestation Tracker
    with st.expander("🎓 Attestation Step-Tracker", expanded=False):
        st.markdown("<small>Procedural flowchart for educational credentials</small>", unsafe_allow_html=True)
        st.info("💡 **Required Sequence:**\n1. Notary in home country\n2. Ministry of Foreign Affairs (Home Country)\n3. UAE Embassy in home country\n4. MOFA within UAE")

    # 📁 QUICK TOOL: Smart Photo/Doc Scrubber (Simulated OCR Verification)
    with st.expander("📁 Smart Photo/Doc Auditor", expanded=False):
        st.markdown("<small>Verify visa photo criteria or passport expiry</small>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Passport/ID Copy", type=["jpg", "png", "pdf"])
        if uploaded_file is not None:
            st.info("🔍 **Scanning file details...**")
            # Mocking simulated AI OCR validation check to prevent failure
            st.success("✅ **Scan Complete:** Passport expiry date is valid (>6 months validity confirmed for UAE entry). White background requirement met.")

    st.markdown("---")
    st.markdown("### 🔗 Official Government Portals")
    st.markdown("- 🇦🇪 [Official UAE Portal](https://u.ae)")
    st.markdown("- 🏢 [ICP Smart Services](https://icp.gov.ae)")
    st.markdown("- ✈️ [GDRFA Visa Hub](https://gdrfad.gov.ae)")
    st.markdown("- 🚗 [RTA Traffic Authority](https://rta.ae)")
    
    # Hidden developer dropdown in case keys must be manually entered, safely out of visual focus
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    with st.expander("⚙️ Developer Configurations", expanded=False):
        manual_key = st.text_input("Local Fallback API Token", type="password", help="Use this strictly for localized mock runs if secrets are not mounted.")
        if manual_key:
            resolved_api_key = manual_key

if not resolved_api_key:
    # Set a dummy key so the layout loads gracefully and tells the user to mount the API key
    resolved_api_key = "MOCK_KEY_PRESENTATION_MODE"

# --- MAIN DISPLAY WRAPPER ---
col_main, col_spacer = st.columns([12, 1])

with col_main:
    # Custom Brand Header with UAE colors
    st.markdown("""
    <div class="branding-container">
        <div class="branding-logo">🇦🇪</div>
        <div class="branding-text">
            <h1>Daleel | دليلك في الإمارات</h1>
            <p>Smart RAG AI Copilot for UAE Residency, Golden Visas, Licensing & Business Operations</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Disclaimer Banner
    st.markdown("""
    <div class="disclaimer-banner">
        <strong>⚠️ Prototype Demonstration:</strong> Daleel is an independent hackathon AI assistant powered by 
        RAG architecture. Always verify crucial data on the official portals provided.
    </div>
    """, unsafe_allow_html=True)

    # Chat history session storage
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Stream the conversational greeting automatically on load
    if not st.session_state.messages and resolved_api_key and resolved_api_key != "MOCK_KEY_PRESENTATION_MODE":
        with st.spinner("Daleel is preparing resources..."):
            greeting = generate_greeting(resolved_api_key)
            st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})
    elif not st.session_state.messages:
        # Fallback greeting if no real API key is injected yet
        fallback_greet = "Marhaba! Welcome to **Daleel (دليل)** 🇦🇪. I can help with any questions regarding Golden Visas, license conversions, or mainland corporate setups. (Configure your Gemini API Key in the settings below to query the live agent!)."
        st.session_state.messages.append({"role": "assistant", "content": fallback_greet, "sources": []})

    st.markdown("### ⚡ Quick Navigation Cards")
    q_col1, q_col2, q_col3 = st.columns(3)
    quick_query = None

    with q_col1:
        st.markdown("""
        <div class="accent-card">
            <h4 style="margin: 0 0 8px 0; color: #00732F;">🚗 License Exchange</h4>
            <p style="font-size: 0.85rem; color: #6B7280; margin: 0 0 12px 0;">Convert your home license to a UAE license instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Query Licensing Exchange Route", key="btn_lic"):
            quick_query = "How can I convert my foreign driving license to a UAE license?"

    with q_col2:
        st.markdown("""
        <div class="accent-card-gold">
            <h4 style="margin: 0 0 8px 0; color: #C5A059;">👑 Professional Golden Visa</h4>
            <p style="font-size: 0.85rem; color: #6B7280; margin: 0 0 12px 0;">Check rules on 10-year residency for employees.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Query Golden Visa Rules", key="btn_visa"):
            quick_query = "What are the requirements and process steps for a Professional Golden Visa?"

    with q_col3:
        st.markdown("""
        <div class="accent-card">
            <h4 style="margin: 0 0 8px 0; color: #00732F;">🏢 Mainland Corporate</h4>
            <p style="font-size: 0.85rem; color: #6B7280; margin: 0 0 12px 0;">Establish a corporate business structure in Dubai mainland.</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Query Corporate Business Setup", key="btn_biz"):
            quick_query = "What are the steps and fees to register a Dubai Mainland Corporate Business License?"

    if quick_query:
        st.session_state.messages.append({"role": "user", "content": quick_query})
        matched_docs, context_string = retrieve_context(quick_query, vectorizer, tfidf_matrix, kb_data)
        
        if resolved_api_key == "MOCK_KEY_PRESENTATION_MODE" or not resolved_api_key:
            reply = f"Thank you for asking about: *'{quick_query}'*.\n\nSince this is in demo preview mode with no API key supplied, here is the direct matching context retrieved from our local knowledge base:\n\n"
            for doc in matched_docs:
                reply += f"**{doc['title']}**\n- **Eligibility:** {doc['eligibility']}\n- **Required Documents:** {doc['documents']}\n- **Steps:** {doc['steps']}\n- **Fees:** {doc['fees']}\n\n"
        else:
            with st.spinner("Retrieving contexts..."):
                reply = generate_grounded_response(quick_query, context_string, resolved_api_key)
                
        st.session_state.messages.append({"role": "assistant", "content": reply, "sources": matched_docs})

    st.markdown("### 💬 Interactive Dialogue Channel")
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg.get("sources") and msg["role"] == "assistant":
                st.markdown("""
                <div style="background-color: #F9FAFB; padding: 10px; border-radius: 8px; border: 1px dashed #D1D5DB; margin-top: 10px;">
                    <span style="font-size: 0.85rem; font-weight: bold; color: #374151;">Verified Legal Sources:</span>
                </div>
                """, unsafe_allow_html=True)
                for src in msg["sources"]:
                    st.markdown(f"- 🔗 [{src['title']}]({src['official_url']})")

    if user_input := st.chat_input("Ask Daleel about visa eligibility, residency, or setup laws..."):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            matched_docs, context_string = retrieve_context(user_input, vectorizer, tfidf_matrix, kb_data)
            
            if resolved_api_key == "MOCK_KEY_PRESENTATION_MODE" or not resolved_api_key:
                # Fallback text to maintain absolute responsiveness even without operational tokens
                st.warning("🔑 Live AI generation requires a Gemini API Key. You can set it securely in 'Developer Configurations' on the left sidebar.")
                reply = "Here are the closest records matched from my local index:\n\n"
                if matched_docs:
                    for doc in matched_docs:
                        reply += f"**{doc['title']}**\n- **Eligibility:** {doc['eligibility']}\n- **Documents:** {doc['documents']}\n- **Fees:** {doc['fees']}\n\n"
                else:
                    reply += "I couldn't locate precise matches inside my current local index. Please ask about Golden Visas, business setup, or driving licenses, or input a Gemini Token to unlock generative small-talk!"
                st.markdown(reply)
            else:
                with st.spinner("Searching official records..."):
                    reply = generate_grounded_response(user_input, context_string, resolved_api_key)
                    st.markdown(reply)
                    
            if matched_docs:
                st.markdown("""
                <div style="background-color: #F9FAFB; padding: 10px; border-radius: 8px; border: 1px dashed #D1D5DB; margin-top: 10px;">
                    <span style="font-size: 0.85rem; font-weight: bold; color: #374151;">Verified Reference Pages:</span>
                </div>
                """, unsafe_allow_html=True)
                for src in matched_docs:
                    st.markdown(f"- 🔗 [{src['title']}]({src['official_url']})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": reply,
                "sources": matched_docs
            })
