import streamlit as st
import json
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import google.generativeai as genai

# Page Configuration Settings
st.set_page_config(
    page_title="UAE Gov Services AI Assistant",
    page_icon="🇦🇪",
    layout="centered"
)

# --- DISCLAIMER BANNER ---
st.markdown(
    """
    <div style="background-color:#fff3cd; padding:14px; border-radius:8px; border-left: 6px solid #ffc107; margin-bottom:25px;">
        <span style="color:#856404; font-weight:bold;">⚠️ Prototype Disclaimer:</span> 
        <span style="color:#856404;">This application is an independent prototype built for demonstration purposes. It is NOT an official government portal. Always confirm details at the official source links provided.</span>
    </div>
    """, 
    unsafe_allow_html=True
)

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

# Load Local Knowledge Base
@st.cache_data
def load_knowledge_base():
    if os.path.exists("knowledge_base.json"):
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

kb_data = load_knowledge_base()

# Build TF-IDF Retrieval Index
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

# Retrieval Function
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
    """Initialize the Gemini model once per API key."""
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )


def get_chat_session(api_key):
    """
    Maintain one persistent chat session in Streamlit session_state so the
    model has real conversational memory, instead of a stateless one-shot
    call per message.
    """
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
            full_message = (
                f"RETRIEVED CONTEXT:\n{context_string}\n\n"
                f"USER QUESTION:\n{query}"
            )
        else:
            # No matching KB entry — let the model handle small talk or
            # ask for clarification, per the system prompt's rules.
            full_message = (
                f"RETRIEVED CONTEXT:\n(none found for this message)\n\n"
                f"USER QUESTION:\n{query}"
            )

        response = chat.send_message(full_message)
        return response.text
    except Exception as e:
        return f"Something went wrong while generating a response: {str(e)}"


def generate_greeting(api_key):
    """Ask the model to produce the opening greeting itself, so tone stays consistent with the system prompt instead of being hardcoded."""
    try:
        chat = get_chat_session(api_key)
        response = chat.send_message(
            "SYSTEM_EVENT: A new user has just opened the chat. No question has been asked yet. "
            "Greet them warmly and briefly introduce what you can help with (UAE visas and licenses)."
        )
        return response.text
    except Exception as e:
        return f"Marhaba! Welcome 🇦🇪 — I can help with UAE visa and license questions. (Greeting generation error: {str(e)})"


# --- UI HEADER ---
st.title("UAE Government Services Assistant")
st.caption("Prototype RAG-based assistant for visas and licenses")

# Sidebar
with st.sidebar:
    st.header("🔑 Configuration")

    if "GEMINI_API_KEY" in st.secrets:
        api_key_input = st.secrets["GEMINI_API_KEY"]
        st.success("🔒 API key loaded from secrets.")
    else:
        api_key_input = st.text_input("Enter Google Gemini API Key", type="password", help="Free-tier key from Google AI Studio.")
        if not api_key_input:
            st.info("💡 Paste your Gemini API key above to begin.")

    st.markdown("---")
    st.markdown("### Trusted Verification Hubs")
    st.markdown("- [Official UAE Portal](https://u.ae)")
    st.markdown("- [ICP Portal](https://icp.gov.ae)")
    st.markdown("- [GDRFA Portal](https://gdrfad.gov.ae)")
    st.markdown("- [RTA Portal](https://rta.ae)")
    st.markdown("- [MOHRE Portal](https://mohre.gov.ae)")

# Chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- GREET FIRST, BEFORE ANY USER INPUT ---
if not st.session_state.messages and api_key_input:
    greeting = generate_greeting(api_key_input)
    st.session_state.messages.append({"role": "assistant", "content": greeting, "sources": []})

# Quick Query Buttons
st.markdown("### ⚡ Quick Queries")
col1, col2, col3 = st.columns(3)
quick_query = None

with col1:
    if st.button("🎓 Student Visa Info"):
        quick_query = "What are the requirements and process steps for a Student Visa?"
with col2:
    if st.button("🚗 Convert Driving License"):
        quick_query = "How can I convert my foreign driving license to a UAE license?"
with col3:
    if st.button("💼 Golden Visa Options"):
        quick_query = "What is the eligibility for a Golden Visa?"

if quick_query and api_key_input:
    st.session_state.messages.append({"role": "user", "content": quick_query})
    matched_docs, context_string = retrieve_context(quick_query, vectorizer, tfidf_matrix, kb_data)
    reply = generate_grounded_response(quick_query, context_string, api_key_input)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": matched_docs})

# Render chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources") and msg["role"] == "assistant":
            st.markdown("**Verify on official source:**")
            for src in msg["sources"]:
                st.markdown(f"- [{src['title']}]({src['official_url']})")

# Chat input
if user_input := st.chat_input("Ask about UAE visas, driving renewals, or business licenses..."):
    if not api_key_input:
        st.warning("Please enter your Gemini API key in the sidebar first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            matched_docs, context_string = retrieve_context(user_input, vectorizer, tfidf_matrix, kb_data)
            with st.spinner("Thinking..."):
                reply = generate_grounded_response(user_input, context_string, api_key_input)
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
