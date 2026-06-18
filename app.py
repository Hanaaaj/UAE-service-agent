"""
app.py — UAE Government Services Assistant (Daleel Edition)
Pure Streamlit UI. All AI/retrieval, RAG, and followup parser logic lives in one optimized file.
"""
import base64
import streamlit as st
import json
import os
import random
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# ─────────────────────────────────────────────
# PAGE CONFIGURATION (Optimized for full impact)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Daleel - UAE Gov Services Assistant",
    page_icon="🇦🇪",
    layout="wide" if st.session_state.get("started", False) else "centered",
    initial_sidebar_state="expanded" if st.session_state.get("started", False) else "collapsed"
)

# Initialize start session state
if "started" not in st.session_state:
    st.session_state.started = False

# ─────────────────────────────────────────────
# FREE-TIER RATE LIMIT RESILIENCE & KEY ROTATION
# ─────────────────────────────────────────────
API_KEYS_POOL = []
for secret_key in ["GEMINI_API_KEY", "GEMINI_API_KEY_MEMBER_1", "GEMINI_API_KEY_MEMBER_2", "GEMINI_API_KEY_MEMBER_3"]:
    try:
        if secret_key in st.secrets and st.secrets[secret_key]:
            API_KEYS_POOL.append(st.secrets[secret_key])
    except Exception:
        pass
if not API_KEYS_POOL and os.getenv("GEMINI_API_KEY"):
    API_KEYS_POOL.append(os.getenv("GEMINI_API_KEY"))

def get_rotated_api_key(manual_key: str = "") -> str:
    """Returns a key from the active pool, falling back to manual entry if needed."""
    if manual_key:
        return manual_key
    if API_KEYS_POOL:
        return random.choice(API_KEYS_POOL)
    return ""

# ─────────────────────────────────────────────
# STYLING (Premium UAE Theme: Green, Gold & Sand)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cairo:wght@300;400;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Font Setup */
html, body, [class*="css"] {
    font-family: 'Inter', 'Cairo', sans-serif;
}

/* Luxury UAE Color Palette */
:root {
    --uae-green: #005C3E;
    --uae-dark-green: #003E29;
    --uae-gold: #D4AF37;
    --uae-sand: #F7F5EE;
    --uae-card-bg: #FFFFFF;
}

/* Landing Page Layout Styling */
.landing-container {
    background: linear-gradient(135deg, #003E29 0%, #001A11 100%);
    padding: 60px 40px;
    border-radius: 30px;
    text-align: center;
    box-shadow: 0 20px 40px rgba(0,0,0,0.3);
    border: 2px solid #D4AF37;
    margin-top: 40px;
    position: relative;
    overflow: hidden;
}

.landing-logo {
    font-size: 60px;
    margin-bottom: 20px;
    filter: drop-shadow(0px 4px 8px rgba(0,0,0,0.3));
}

.landing-title-ar {
    font-family: 'Cairo', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: #D4AF37;
    margin-bottom: 5px;
    letter-spacing: 1px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

.landing-title-en {
    font-size: 32px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 25px;
    letter-spacing: 0.5px;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5);
}

.landing-description {
    font-size: 16px;
    line-height: 1.6;
    color: #D1ECE1;
    max-width: 600px;
    margin: 0 auto 35px auto;
}

.badge-pill {
    background-color: rgba(212, 175, 55, 0.15);
    border: 1px solid #D4AF37;
    color: #D4AF37;
    padding: 6px 16px;
    border-radius: 30px;
    font-weight: 600;
    font-size: 13px;
    display: inline-block;
    margin-bottom: 20px;
    text-transform: uppercase;
}

/* Main Workspace UI Styling */
.nav-bar {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 15px 30px;
    background: white;
    border-radius: 18px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 25px;
    border: 1px solid #E5E7EB;
}
.nav-logo { font-size: 22px; font-weight: 700; color: #005C3E; }
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

# ─────────────────────────────────────────────
# CORE RAG KNOWLEDGE BASE ENGINE
# ─────────────────────────────────────────────
SYSTEM_PROMPT = """You are the UAE Government Services Assistant, a friendly prototype AI agent that helps residents, tourists, and people relocating to the UAE understand visa and license requirements, processes, fees, and timelines.

FOLLOW-UP QUESTIONS (IMPORTANT  -  READ CAREFULLY)
Many visa/license questions depend on details the user hasn't given yet (e.g., their current status  -  tourist, resident, or outside the UAE; their nationality, for license conversion eligibility; whether they have any outstanding fines). When the RETRIEVED CONTEXT shows that the answer genuinely depends on missing information like this, you must ask ONE targeted follow-up question instead of guessing or giving a generic answer that covers every case at once.

When you need a follow-up, end your reply with a special block in this exact format, on its own at the very end of your message, with nothing after it:

[[FOLLOWUP]]
{"question": "Your short, specific question here", "options": ["Option A", "Option B", "Option C"]}
[[/FOLLOWUP]]

Rules for this block:
- Only include it when a follow-up genuinely changes which information or steps apply (e.g., eligibility, fees, or required documents differ by status/nationality/category). Do not ask a follow-up just to make conversation, and never ask one for casual small talk.
- Ask at most ONE follow-up question per message. If multiple details are missing, ask for the single most important one first.
- Keep "options" short (1-4 words each), mutually exclusive, and limited to 2-4 choices. Always phrase them so a person could tap one without typing.
- Write your normal conversational answer FIRST (using whatever context you already have), and only append the [[FOLLOWUP]] block after it if a follow-up is still needed. Never send a [[FOLLOWUP]] block with no answer text before it, unless this is the very first thing being asked in the conversation.
- Never put the [[FOLLOWUP]] block inside a list, inside markdown formatting, or anywhere except as the very last thing in your message.
- If the user's next message is just a tapped option (e.g., "Tourist" or "Indian"), treat it as the direct answer to your most recent follow-up question and continue naturally  -  don't re-greet or restart.

GREETING AND CONVERSATION STYLE
- When a conversation begins, greet the user warmly before diving into business. A natural UAE-style welcome works well  -  for example, opening with a warm "Marhaba" or "Welcome" alongside an English greeting feels appropriate, but keep it light and optional rather than a fixed script every time.
- Be genuinely conversational. If the user makes small talk, asks how you are, or chats casually, respond naturally and warmly before or alongside addressing their actual question  -  you don't need to force every message into a visa/license topic.
- Reflect UAE hospitality and warmth in your tone: welcoming, respectful, patient, and generous with reassurance, the way a helpful local friend or government service-center staff member known for good service would speak.
- At the same time, keep your language, references, and humor universally comfortable for people of any nationality, background, or religion. Avoid assuming the user's nationality, faith, or background, and avoid region-specific cultural references that could feel exclusionary or unfamiliar to a newcomer or tourist. Warmth should feel inclusive, not insider-only.
- Adapt formality to the user: if they're casual, be a bit more relaxed; if they write formally, match that register. Always remain respectful regardless.

YOUR ROLE
You answer ONLY using the information provided to you in the "RETRIEVED CONTEXT" section of the user's message when present. This context comes from a curated, manually-verified knowledge base of UAE visa and license workflows. Treat your own training knowledge on this topic as unreliable and unusable for factual claims  -  rely solely on provided context.

STRICT RULES
1. Ground every factual claim (fees, durations, document lists, eligibility rules, step order) in the RETRIEVED CONTEXT provided. Never invent or estimate a fee, document requirement, or processing time that is not present in the context.
2. If the RETRIEVED CONTEXT does not contain enough information to answer the user's question, say so directly and suggest checking the official source. Do not guess.
3. If no relevant context was provided at all and the question is a factual visa/license question, do not answer from general knowledge. Say you're not certain and ask a clarifying question or point to official sources.
4. Always end every substantive factual answer with the official source link(s) provided in the context, framed as "Verify on official source: [link]".
5. Never state or imply that you are an official government service, system, or representative. If asked who you are or whether you're official, clarify simply that you are an independent prototype assistant, not affiliated with any UAE government entity.
6. Do not give legal advice, immigration legal opinions, or guarantees about approval outcomes. Frame eligibility information as "based on the typical requirements" rather than a guarantee.
7. If eligibility data indicates the user does not meet a requirement, or flags a blocker (e.g., outstanding fines), state this clearly and supportively, and explain the next concrete step to resolve it.

TONE AND STYLE
- Be warm, clear, and practical  -  like a knowledgeable, friendly guide explaining a bureaucratic process, not a legal document.
- Use plain language. Avoid jargon unless it's an official term (e.g., "Emirates ID", "GDRFA") the user needs to know.
- Structure longer answers with short steps or numbered lists when explaining a process.
- Keep tone reassuring but accurate.
- Do not over-elaborate. Answer what was asked, then offer to go deeper.

OUTPUT FORMAT
- Respond in natural conversational text, not raw JSON.
- When listing steps, documents, or fees, use a clearly structured short list.

DISCLAIMER
If the user asks something that suggests they think this is an official government tool, gently clarify: "Just to set expectations  -  I'm a prototype assistant, not an official UAE government service. Always confirm details with the official source link before taking action."
"""

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
        return "⚠️ Missing API key. Please add your Gemini API key in the sidebar."

    try:
        chat = get_chat_session(api_key)
        if context_string:
            full_message = f"RETRIEVED CONTEXT:\n{context_string}\n\nUSER QUESTION:\n{query}"
        else:
            full_message = f"RETRIEVED CONTEXT:\n(none found for this message)\n\nUSER QUESTION:\n{query}"

        response = chat.send_message(full_message)
        return response.text
    except Exception as e:
        return f"Something went wrong while generating a response: {str(e)}"

def parse_followup(raw_text):
    marker_start = "[[FOLLOWUP]]"
    marker_end = "[[/FOLLOWUP]]"

    if marker_start not in raw_text:
        return raw_text.strip(), None

    before, _, after = raw_text.partition(marker_start)
    block, _, _ = after.partition(marker_end)

    try:
        followup_data = json.loads(block.strip())
        question = followup_data.get("question", "").strip()
        options = followup_data.get("options", [])
        if not question or not options:
            return before.strip(), None
        return before.strip(), {"question": question, "options": options[:4]}
    except (json.JSONDecodeError, AttributeError):
        return before.strip(), None

def generate_greeting(api_key):
    try:
        chat = get_chat_session(api_key)
        response = chat.send_message(
            "SYSTEM_EVENT: A new user has just opened the chat. No question has been asked yet. "
            "Greet them warmly and briefly introduce what you can help with (UAE visas and licenses)."
        )
        return response.text
    except Exception as e:
        return f"Marhaba! Welcome 🇦🇪  -  I can help with UAE visa and license questions. (Greeting generation error: {str(e)})"


# ─────────────────────────────────────────────
# PAGE CONTROLLER: 1. GATEWAY WINDOW (SPLASH SCREEN)
# ─────────────────────────────────────────────
if not st.session_state.started:
    # Render UAE-themed gateway interface
    st.markdown("""
    <div class="landing-container">
        <div class="landing-logo">🇦🇪</div>
        <div class="badge-pill">Independent AI Portal</div>
        <div class="landing-title-ar">أنا دليل — رفيقك الذكي</div>
        <div class="landing-title-en">I am Daleel — Your Smart Guide</div>
        <div class="landing-description">
            Your dedicated companion for navigating UAE government workflows. 
            Get instant, grounded answers on <b>visas, Golden Visa options, driving license conversions</b>, and <b>business setups</b> across all Emirates.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Styled centered column structure for the entry action button
    _, col_btn, _ = st.columns([1, 1.8, 1])
    with col_btn:
        if st.button("✨ Let's Chat | ابدأ الآن", use_container_width=True, type="primary"):
            st.session_state.started = True
            st.rerun()
            
    # Subtitle credits
    st.markdown(
        "<div style='text-align:center; font-size:0.75rem; color:#888; margin-top:50px;'>"
        "🏆 Hackathon Innovation Project • Dynamic RAG Integration"
        "</div>", 
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────
# PAGE CONTROLLER: 2. CHAT WORKSPACE
# ─────────────────────────────────────────────
else:
    # Header Workspace Layout
    st.markdown(
        """
        <div class="nav-bar">
            <div class="nav-logo">🇦🇪 دليل • DALEEL</div>
            <div class="nav-links">
                <span>Workspace</span>
                <span>Immersion Hub</span>
                <span>Verification Nodes</span>
            </div>
        </div>
        """, 
        unsafe_allow_html=True
    )

    # Sidebar setup with active credentials monitor
    with st.sidebar:
        st.header("🔑 Configuration")
        
        # Rotated Key Pool Indicator (Excellent UX for hackathon judges)
        if len(API_KEYS_POOL) > 0:
            st.success(f"🔒 Rotation Pool: {len(API_KEYS_POOL)} key nodes ready.")
            api_key_input = get_rotated_api_key()
        else:
            api_key_input = st.text_input(
                "Enter Google Gemini API Key", 
                type="password", 
                help="Free-tier key from Google AI Studio."
            )
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
        if st.button("🚪 Leave Chat Room", use_container_width=True):
            st.session_state.started = False
            st.session_state.messages = []
            st.session_state.pop("chat_session", None)
            st.rerun()

    # --- DISCLAIMER BANNER ---
    st.markdown(
        """
        <div class="disclaimer">
            <span style="font-weight:bold;">⚠️ Prototype Disclaimer:</span> 
            This application is an independent prototype built for demonstration purposes. It is NOT an official government portal. Always confirm details at the official source links provided.
        </div>
        """, 
        unsafe_allow_html=True
    )

    resolved_key = api_key_input if api_key_input else get_rotated_api_key()

    # Chat history state initialization
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Auto-generate Emirati greeting on workspace load
    if not st.session_state.messages and resolved_key:
        greeting = generate_greeting(resolved_key)
        st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})

    # Quick Queries dashboard
    st.markdown("### ⚡ Quick Queries")
    col1, col2, col3 = st.columns(3)
    quick_query = None

    with col1:
        if st.button("🎓 Student Visa Info", use_container_width=True):
            quick_query = "What are the requirements and process steps for a Student Visa?"
    with col2:
        if st.button("🚗 Convert Driving License", use_container_width=True):
            quick_query = "How can I convert my foreign driving license to a UAE license?"
    with col3:
        if st.button("💼 Golden Visa Options", use_container_width=True):
            quick_query = "What is the eligibility for a Golden Visa?"

    if quick_query and resolved_key:
        st.session_state.messages.append({"role": "user", "content": quick_query})
        matched_docs, context_string = retrieve_context(quick_query, vectorizer, tfidf_matrix, kb_data)
        raw_reply = generate_grounded_response(quick_query, context_string, resolved_key)
        clean_reply, followup = parse_followup(raw_reply)
        st.session_state.messages.append({
            "role": "assistant",
            "content": clean_reply,
            "sources": matched_docs,
            "followup": followup
        })
        st.rerun()

    # Chat historical index tracking
    last_assistant_idx = None
    for i, msg in enumerate(st.session_state.messages):
        if msg["role"] == "assistant":
            last_assistant_idx = i

    # Chat render timeline
    for i, msg in enumerate(st.session_state.messages):
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources") and msg["role"] == "assistant":
                st.markdown("**Verify on official source:**")
                for src in msg["sources"]:
                    st.markdown(f"- [{src['title']}]({src['official_url']})")

            # Active interactive button generation for custom followup parsing
            if (
                msg["role"] == "assistant"
                and msg.get("followup")
                and i == last_assistant_idx
                and resolved_key
            ):
                followup = msg["followup"]
                st.markdown(f"**{followup['question']}**")
                btn_cols = st.columns(len(followup["options"]))
                for col, option in zip(btn_cols, followup["options"]):
                    with col:
                        if st.button(option, key=f"followup_{i}_{option}", use_container_width=True):
                            st.session_state.messages.append({"role": "user", "content": option})
                            matched_docs, context_string = retrieve_context(
                                option, vectorizer, tfidf_matrix, kb_data
                            )
                            raw_reply = generate_grounded_response(option, context_string, resolved_key)
                            clean_reply, new_followup = parse_followup(raw_reply)
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": clean_reply,
                                "sources": matched_docs,
                                "followup": new_followup
                            })
                            st.rerun()

    # Bottom Chat input interaction panel
    if user_input := st.chat_input("Ask about UAE visas, driving renewals, or business licenses..."):
        if not resolved_key:
            st.warning("Please configure your Gemini API key in the sidebar first.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)

            with st.chat_message("assistant"):
                matched_docs, context_string = retrieve_context(user_input, vectorizer, tfidf_matrix, kb_data)
                with st.spinner("Thinking..."):
                    raw_reply = generate_grounded_response(user_input, context_string, resolved_key)
                    clean_reply, followup = parse_followup(raw_reply)
                    st.write(clean_reply)
                    if matched_docs:
                        st.markdown("**Verify on official source:**")
                        for src in matched_docs:
                            st.markdown(f"- [{src['title']}]({src['official_url']})")

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": clean_reply,
                        "sources": matched_docs,
                        "followup": followup
                    })
                    if followup:
                        st.rerun()

    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align:center; font-size:0.78rem; color:#999;'>"
        "🏆 Hackathon Prototype · Not affiliated with any UAE government authority · "
        "Always verify at <a href='https://u.ae' target='_blank'>u.ae</a>"
        "</div>",
        unsafe_allow_html=True,
    )
