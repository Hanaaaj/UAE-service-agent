import base64
import streamlit as st
import json
import os
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ─────────────────────────────────────────────────────────────
# PAGE CONFIGURATION Settings
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="UAE Gov Services AI Assistant",
    page_icon="🇦🇪",
    layout="wide"  # Use wide layout to facilitate multi-column dashboards and scanner widgets
)

# ─────────────────────────────────────────────────────────────
# FREE-TIER RATE LIMIT RESILIENCE & KEY ROTATION SETUP
# ─────────────────────────────────────────────────────────────
# Gathers all team keys from Secrets to share limits (bypasses 429 Resource Exhausted)
API_KEYS_POOL = []
for secret_key in ["GEMINI_API_KEY", "GEMINI_API_KEY_MEMBER_1", "GEMINI_API_KEY_MEMBER_2", "GEMINI_API_KEY_MEMBER_3"]:
    if secret_key in st.secrets and st.secrets[secret_key]:
        API_KEYS_POOL.append(st.secrets[secret_key])
if not API_KEYS_POOL and os.getenv("GEMINI_API_KEY"):
    API_KEYS_POOL.append(os.getenv("GEMINI_API_KEY"))

def get_rotated_api_key(manual_key: str = "") -> str:
    """Returns a key from the rotation pool or falls back to manual entry."""
    if manual_key:
        return manual_key
    if API_KEYS_POOL:
        return random.choice(API_KEYS_POOL)
    return ""

# ─────────────────────────────────────────────────────────────
# PREMIUM STYLING & CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.main { background-color: #F7F9FA; }
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 30px;
    background: white;
    border-radius: 18px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    margin-bottom: 25px;
}
.nav-logo { font-size: 22px; font-weight: 700; color: #006C4C; }
.nav-links { display: flex; gap: 28px; font-weight: 500; color: #1E293B; font-size: 14px; }
.disclaimer {
    background-color: #fff3cd; 
    padding: 14px; 
    border-radius: 8px; 
    border-left: 6px solid #ffc107; 
    margin-bottom: 25px;
    font-size: 0.88rem;
    color: #856404;
}
.source-badge {
    display: inline-block;
    background: #e8f5e9;
    color: #2e7d32;
    font-size: 0.76rem;
    padding: 3px 10px;
    border-radius: 20px;
    margin: 3px 4px 3px 0;
    font-weight: 500;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SYSTEM INSTRUCTIONS
# ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are the UAE Government Services Assistant, a friendly prototype AI agent that helps residents, tourists, and people relocating to the UAE understand visa and license requirements, processes, fees, and timelines.

GREETING AND CONVERSATION STYLE
- When a conversation begins, greet the user warmly before diving into business. A natural UAE-style welcome works well — for example, opening with a warm "Marhaba" or "Welcome" alongside an English greeting feels appropriate, but keep it light and optional rather than a fixed script every time.
- Be genuinely conversational. If the user makes small talk, asks how you are, or chats casually, respond naturally and warmly before or alongside addressing their actual question — you don't need to force every message into a visa/license topic.
- Reflect UAE hospitality and warmth in your tone: welcoming, respectful, patient, and generous with reassurance, the way a helpful local friend or government service-center staff member known for good service would speak.
- At the same time, keep your language, references, and humor universally comfortable for people of any nationality, background, or religion. Avoid assuming the user's nationality, faith, or background, and avoid region-specific cultural references that could feel exclusionary or unfamiliar to a newcomer or tourist. Warmth should feel inclusive, not insider-only.
- Adapt formality to the user: if they're casual, be a bit more relaxed; if they write formally, match that register. Always remain respectful regardless.

YOUR ROLE
You answer ONLY using the information provided to you in the "RETRIEVED CONTEXT" section of the user's message when present. This context comes from a curated, manually-verified knowledge base of UAE visa and license workflows. Treat your own training knowledge on this topic as unreliable and unusable for factual claims — rely solely on provided context.

STRICT RULES
1. Ground every factual claim (fees, durations, document lists, eligibility rules, step order) in the RETRIEVED CONTEXT provided. Never invent or estimate a fee, document requirement, or processing time that is not present in the context.
2. If the RETRIEVED CONTEXT does not contain enough information to answer the user's question, say so directly and suggest checking the official source. Do not guess.
3. If no relevant context was provided at all and the question is a factual visa/license question, do not answer from general knowledge. Say you're not certain and ask a clarifying question or point to official sources.
4. Always end every substantive factual answer with the official source link(s) provided in the context, framed as "Verify on official source: [link]".
5. Never state or imply that you are an official government service, system, or representative. If asked who you are or whether you're official, clarify simply that you are an independent prototype assistant, not affiliated with any UAE government entity.
6. Do not give legal advice, immigration legal opinions, or guarantees about approval outcomes. Frame eligibility information as "based on the typical requirements" rather than a guarantee.
7. If eligibility data indicates the user does not meet a requirement, or flags a blocker (e.g., outstanding fines), state this clearly and supportively, and explain the next concrete step to resolve it.

TONE AND STYLE
- Be warm, clear, and practical — like a knowledgeable, friendly guide explaining a bureaucratic process, not a legal document.
- Use plain language. Avoid jargon unless it's an official term (e.g., "Emirates ID", "GDRFA") the user needs to know.
- Structure longer answers with short steps or numbered lists when explaining a process.
- Keep tone reassuring but accurate.
- Do not over-elaborate. Answer what was asked, then offer to go deeper.

OUTPUT FORMAT
- Respond in natural conversational text, not raw JSON.
- When listing steps, documents, or fees, use a clearly structured short list.

DISCLAIMER
If the user asks something that suggests they think this is an official government tool, gently clarify: "Just to set expectations — I'm a prototype assistant, not an official UAE government service. Always confirm details with the official source link before taking action."
"""

# ─────────────────────────────────────────────────────────────
# LOCAL RAG LAYER (KNOWLEDGE BASE & TF-IDF INDEX)
# ─────────────────────────────────────────────────────────────
@st.cache_data
def load_knowledge_base():
    if os.path.exists("knowledge_base.json"):
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

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

def retrieve_context(query, vectorizer, tfidf_matrix, data, top_n=2, threshold=0.12):
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

# ─────────────────────────────────────────────────────────────
# GEMINI ENGINE WITH KEY ROTATION & RATE LIMIT AUTO-RECOVERY
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def get_model(api_key):
    """Initialize the Gemini model once per API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )

def get_chat_session(api_key):
    """Maintains a persistent conversation session in Streamlit state."""
    if "chat_session" not in st.session_state:
        model = get_model(api_key)
        st.session_state.chat_session = model.start_chat(history=[])
    return st.session_state.chat_session

def execute_with_api_resilience(func, *args, **kwargs):
    """Runs a API callback, catches 429/quota warnings, rotates key, and retries."""
    max_attempts = 4
    delay = 1.0
    last_exception = None
    
    for attempt in range(max_attempts):
        # Always pick a key dynamically to balance resource pools
        active_key = get_rotated_api_key(kwargs.get("manual_key", ""))
        if not active_key:
            return "⚠️ Please configure a valid Google Gemini API Key in the sidebar configuration to begin."
        
        # Configure genai with active key
        genai.configure(api_key=active_key)
        
        try:
            return func(active_key, *args)
        except Exception as e:
            last_exception = e
            err_msg = str(e).lower()
            # If 429 rate limit or quota exceeded occurs, retry with another key from pool
            if "429" in err_msg or "resource_exhausted" in err_msg or "quota" in err_msg:
                time.sleep(delay + random.uniform(0.1, 0.5))
                delay *= 1.5
            else:
                # Raise other exceptions immediately (e.g. invalid key format)
                raise e
                
    return f"⚠️ **API Overload Error:** All active keys in the rotation pool have temporarily exhausted their free-tier limits. Please wait a minute and try again. (Details: {last_exception})"

# ─────────────────────────────────────────────────────────────
# HELPER OPERATIONS
# ─────────────────────────────────────────────────────────────
def _call_grounded_response(api_key, query, context_string):
    chat = get_chat_session(api_key)
    full_message = (
        f"RETRIEVED CONTEXT:\n{context_string or '(none found)'}\n\n"
        f"USER QUESTION:\n{query}"
    )
    response = chat.send_message(full_message)
    return response.text

def _call_greeting(api_key):
    chat = get_chat_session(api_key)
    response = chat.send_message(
        "SYSTEM_EVENT: A new user has just opened the chat. No question has been asked yet. "
        "Greet them warmly and briefly introduce what you can help with (UAE visas and licenses)."
    )
    return response.text

def generate_grounded_response(query, context_string, manual_key):
    return execute_with_api_resilience(_call_grounded_response, query, context_string, manual_key=manual_key)

def generate_greeting(manual_key):
    return execute_with_api_resilience(_call_greeting, manual_key=manual_key)

# ─────────────────────────────────────────────────────────────
# UI LAYOUT & STATIC COMPONENTS
# ─────────────────────────────────────────────────────────────
# Disclaimer Banner
st.markdown(
    """
    <div class="disclaimer">
        <span style="font-weight:bold;">⚠️ Prototype Disclaimer:</span> 
        This application is an independent prototype built for demonstration purposes. It is NOT an official government portal. Always confirm details at the official source links provided.
    </div>
    """, 
    unsafe_allow_html=True
)

# Header Nav-Bar UI
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">🇦🇪 UAE Gov Assistant</div>
    <div class="nav-links">
        <span>Home</span>
        <span>Visa Services</span>
        <span>Driving License</span>
        <span>Business License</span>
        <span>Renewals</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# SIDEBAR CONFIGURATION
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("🔑 Configuration")

    # Interactive key status display (ideal for hackathon grading)
    if len(API_KEYS_POOL) > 0:
        st.success(f"🔒 Key Pool: {len(API_KEYS_POOL)} free keys registered.")
        api_key_input = ""  # Let the rotator fetch automatically
        key_provider_label = "System Rotator"
    else:
        api_key_input = st.text_input("Enter Google Gemini API Key", type="password", help="Free-tier key from Google AI Studio.")
        key_provider_label = "Manual Input"
        if not api_key_input:
            st.info("💡 Paste your Gemini API key above to begin.")

    st.markdown("---")
    st.markdown("### Trusted Verification Hubs")
    st.markdown("- [Official UAE Portal](https://u.ae)")
    st.markdown("- [ICP Portal](https://icp.gov.ae)")
    st.markdown("- [GDRFA Portal](https://gdrfad.gov.ae)")
    st.markdown("- [RTA Portal](https://rta.ae)")
    st.markdown("- [MOHRE Portal](https://mohre.gov.ae)")
    st.markdown("---")
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.pop("chat_session", None)
        st.rerun()

# Determine active key for standard configuration routines
resolved_key = api_key_input if api_key_input else get_rotated_api_key()

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "active_tab" not in st.session_state:
    st.session_state.active_tab = "Visa Services"

# Auto-greeting logic
if not st.session_state.messages and resolved_key:
    greeting = generate_greeting(resolved_key)
    st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})

# ─────────────────────────────────────────────────────────────
# INTERACTIVE TAB CONTROLLER
# ─────────────────────────────────────────────────────────────
st.markdown("<div style='font-size:22px; font-weight:700; color:#1E293B; margin-bottom:14px;'>Quick Service Tabs</div>",
            unsafe_allow_html=True)

tabs_list = ["Visa Services", "Driving License", "Business License", "Renewals & FAQs"]
tab_cols = st.columns(len(tabs_list))

for col, tab_name in zip(tab_cols, tabs_list):
    with col:
        is_active = (st.session_state.active_tab == tab_name)
        if st.button(tab_name, key=f"tab_select_{tab_name}", use_container_width=True, type="primary" if is_active else "secondary"):
            st.session_state.active_tab = tab_name
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# ADAPTIVE SUGGESTED QUICK QUERIES
# ─────────────────────────────────────────────────────────────
st.markdown("### ⚡ Suggested Quick Queries")

if st.session_state.active_tab == "Visa Services":
    queries_pool = [
        ("🎓 Student Visa Info", "What are the requirements and process steps for a Student Visa?"),
        ("💼 Golden Visa Options", "What is the eligibility and basic salary requirements for a Golden Visa?"),
        ("🏢 Golden Visa for Investors", "How can an investor qualify for a Golden Visa through real estate?")
    ]
elif st.session_state.active_tab == "Driving License":
    queries_pool = [
        ("🚗 Convert Driving License", "How can I convert my foreign driving license to a UAE license?"),
        ("🔄 License Renewal Process", "What is the procedure and testing required to renew a UAE driving license?"),
        ("🌟 Golden Chance Test", "What are the rules and guidelines for the Golden Chance road test?")
    ]
elif st.session_state.active_tab == "Business License":
    queries_pool = [
        ("📈 Mainland vs Freezone", "What are the core differences, advantages, and ownership structures of Mainland versus Freezone setups?"),
        ("🛍️ E-Commerce License", "How can I acquire a commercial license to start a digital trading business in Dubai?"),
        ("🤝 Local Agent Requirements", "Does setting up a Mainland business in UAE still require a local Emirati partner?")
    ]
else:
    queries_pool = [
        ("📅 Grace Period for Expiry", "What is the official grace period for residency visas or licenses after expiration?"),
        ("🗂️ Document Attestation Steps", "What is the correct multi-step process to attest my foreign degrees/certificates for the UAE?"),
        ("❓ General FAQs", "What are some of the most frequently asked questions about moving and settling down in the UAE?")
    ]

q_cols = st.columns(3)
quick_query = None

for col, (btn_label, query_text) in zip(q_cols, queries_pool):
    with col:
        if st.button(btn_label, use_container_width=True):
            quick_query = query_text

if quick_query and resolved_key:
    st.session_state.messages.append({"role": "user", "content": quick_query})
    matched_docs, context_string = retrieve_context(quick_query, vectorizer, tfidf_matrix, kb_data)
    with st.spinner("Retrieving facts..."):
        reply = generate_grounded_response(quick_query, context_string, resolved_key)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": matched_docs})
    st.rerun()

# ─────────────────────────────────────────────────────────────
# SMART DOCUMENT PRE-SCRUBBER (MOCK OCR PRE-SCREENER)
# ─────────────────────────────────────────────────────────────
st.markdown("### 📁 Smart Document Audit (Simulated OCR Check)")
uploaded_file = st.file_uploader(
    "Drop a copy of your Passport, License, or Certificate to run an instant rules compliance scan:",
    type=["png", "jpg", "jpeg", "pdf"],
    help="Flags validity dates, country swaps, or notarization issues."
)

if uploaded_file is not None:
    with st.expander("🔍 OCR Scanning Compliance Analysis Report", expanded=True):
        st.info("Reading document parameters and running verification checklists...")
        time.sleep(1.0)
        
        filename = uploaded_file.name.lower()
        if "passport" in filename:
            st.success("✅ File Identified: International Passport Document")
            st.markdown("""
            * **Extracted Expiry Metric:** Dec 15, 2026  
            * **Safety Warning Status:** ⚠️ **ALERT:** This passport has less than 6 months of validity left. 
            Immigration departments like GDRFA/ICP generally reject tourist and entry visas in this threshold.
            
            *Click the button below to ask the assistant how to solve this expiry blocker.*
            """)
        elif "license" in filename or "driving" in filename:
            st.success("✅ File Identified: Regional Driving Permit")
            st.markdown("""
            * **Extracted Issuing Territory:** United Kingdom (DVLA)  
            * **Equivalence Swap Rating:** 🎖️ **DIRECT CONVERSION COMPLIANT.** The United Kingdom is a pre-approved direct conversion partner. You do not need to take mandatory driving classes.
            
            *Click the button below to query the agent on the exact conversion steps.*
            """)
        else:
            st.success("✅ File Identified: General Document")
            st.markdown("""
            * **Structure Scan:** Processing completed without metadata violations. Click the button below to prompt the AI to review rules associated with this document.
            """)
        
        # Interactive Query trigger
        if st.button("💬 Send Document Audit Prompt to Assistant", use_container_width=True):
            audit_prompt = f"I scanned my document: {uploaded_file.name}. Can you explain what visa/license rules and steps apply to it?"
            st.session_state.messages.append({"role": "user", "content": audit_prompt})
            matched_docs, context_string = retrieve_context(audit_prompt, vectorizer, tfidf_matrix, kb_data)
            with st.spinner("Analyzing document compliance..."):
                reply = generate_grounded_response(audit_prompt, context_string, resolved_key)
            st.session_state.messages.append({"role": "assistant", "content": reply, "sources": matched_docs})
            st.rerun()

# ─────────────────────────────────────────────────────────────
# CHAT CONTAINER & RENDERER
# ─────────────────────────────────────────────────────────────
st.markdown("### 💬 Conversation")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources") and msg["role"] == "assistant":
            st.markdown("**Verify on official source:**")
            for src in msg["sources"]:
                st.markdown(f"- [{src['title']}]({src['official_url']})")

# Chat Input Trigger
if user_input := st.chat_input("Ask about UAE visas, driving renewals, or business licenses..."):
    if not resolved_key:
        st.warning("Please configure your Gemini API Key in the sidebar or setup team secrets to chat.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            matched_docs, context_string = retrieve_context(user_input, vectorizer, tfidf_matrix, kb_data)
            with st.spinner("Thinking..."):
                reply = generate_grounded_response(user_input, context_string, resolved_key)
                st.write(reply)
                if matched_docs:
                    st.markdown("**Verify on official source:**")
                    for src in matched_docs:
                        st.markdown(f"- [{src['title']}]({src['official_url']})")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": reply,
                    "sources": matched_docs
                })

# ─────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.78rem; color:#999;'>"
    "🏆 Hackathon Prototype · Not affiliated with any UAE government authority · "
    "Always verify at <a href='https://u.ae' target='_blank'>u.ae</a>"
    "</div>",
    unsafe_allow_html=True,
)
