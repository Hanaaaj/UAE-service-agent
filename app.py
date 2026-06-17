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

# --- REQUIRED HACKATHON DISCLAIMER BANNER ---
st.markdown(
    """
    <div style="background-color:#fff3cd; padding:14px; border-radius:8px; border-left: 6px solid #ffc107; margin-bottom:25px;">
        <span style="color:#856404; font-weight:bold;">⚠️ Hackathon Prototype Disclaimer:</span> 
        <span style="color:#856404;">This application is an independent, student-built prototype developed exclusively for evaluation. It is NOT an official government portal and has no direct structural affiliation with UAE ministries. Always confirm guidelines at official registration portals.</span>
    </div>
    """, 
    unsafe_allow_html=True
)

# Load Local Repository Context Blocks Safely
@st.cache_data
def load_knowledge_base():
    if os.path.exists("knowledge_base.json"):
        with open("knowledge_base.json", "r", encoding="utf-8") as f:
            return json.load(f)
    return []

kb_data = load_knowledge_base()

# Vectorization Index Core Logic (The Semantic Matchmaker)
@st.cache_resource
def build_retrieval_index(_data):
    if not _data:
        return None, None
    documents = []
    for item in _data:
        # Combine structural fields to perform mathematical alignment queries
        text_blob = f"{item['category']} {item['subcategory']} {item['title']} {item['eligibility']} {item['documents']} {item['steps']}"
        documents.append(text_blob.lower())
    
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    return vectorizer, tfidf_matrix

vectorizer, tfidf_matrix = build_retrieval_index(kb_data)

# Retrieval Layer Matrix Extraction Routine
def retrieve_context(query, vectorizer, tfidf_matrix, data, top_n=2, threshold=0.12):
    if not vectorizer or tfidf_matrix is None:
        return [], "No local documentation structural arrays discovered."
    
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
                f"### Structural Target: {item['title']} ({item['category']}/{item['subcategory']})\n"
                f"Eligibility Rules: {item['eligibility']}\n"
                f"Required Documents Profile: {item['documents']}\n"
                f"Process Steps Path: {item['steps']}\n"
                f"Fees Framework: {item['fees']}\n"
                f"Processing Time Window: {item['processing_time']}\n"
                f"Official Verification Link: {item['official_url']}\n\n"
            )
    return results, context_str

# Conversational LLM Layer Grounding Execution

def generate_grounded_response(query, context, api_key):
    if not api_key:
        return "⚠️ Configuration Blocked: Missing API Token parameter.", None
        
    try:
        genai.configure(api_key=api_key)
        # Use the updated stable generation model
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        # New Hybrid System Prompt: Allows both RAG and normal small talk!
        system_prompt = 
         f"""You are the UAE Government Services Assistant, a friendly prototype AI agent that helps residents, tourists, and people relocating to the UAE understand visa and license requirements, processes, fees, and timelines.
            GREETING AND CONVERSATION STYLE
            - When a conversation begins, greet the user warmly before diving into business. A natural UAE-style welcome works well — for example, opening with a warm "Marhaba" or "Welcome" alongside an English greeting feels appropriate, but keep it light and optional rather than a fixed script every time.
            - Be genuinely conversational. If the user makes small talk, asks how you are, or chats casually, respond naturally and warmly before or alongside addressing their actual question — you don't need to force every message into a visa/license topic.
            - Reflect UAE hospitality and warmth in your tone: welcoming, respectful, patient, and generous with reassurance, the way a helpful local friend or government service-center staff member known for good service would speak.
            - At the same time, keep your language, references, and humor universally comfortable for people of any nationality, background, or religion. Avoid assuming the user's nationality, faith, or background, and avoid region-specific cultural references that could feel exclusionary or unfamiliar to a newcomer or tourist. Warmth should feel inclusive, not insider-only.
            - Adapt formality to the user: if they're casual, be a bit more relaxed; if they write formally, match that register. Always remain respectful regardless.
            
            YOUR ROLE
            You answer ONLY using the information provided to you in the "RETRIEVED CONTEXT" section below each user message. This context comes from a curated, manually-verified knowledge base of UAE visa and license workflows. Treat your own training knowledge on this topic as unreliable and unusable for this task — rely solely on the provided context for factual claims.
            
            STRICT RULES
            1. Ground every factual claim (fees, durations, document lists, eligibility rules, step order) in the RETRIEVED CONTEXT provided. Never invent or estimate a fee, document requirement, or processing time that is not present in the context.
            2. If the RETRIEVED CONTEXT does not contain enough information to answer the user's question, say so directly: "I don't have verified information on that specific point — please check [official source link from context]." Do not guess.
            3. If no relevant context was retrieved at all, do not attempt to answer from general knowledge. Respond: "I'm not certain about this — please verify directly with the relevant UAE authority." and ask a clarifying question if the user's request was ambiguous.
            4. Always end every substantive answer with the official source link(s) provided in the context, framed as "Verify on official source: [link]".
            5. Never state or imply that you are an official government service, system, or representative. If asked who you are or whether you're official, clarify simply that you are an independent prototype assistant, not affiliated with any UAE government entity — no need to mention hackathons or development context.
            6. Do not give legal advice, immigration legal opinions, or guarantees about approval outcomes (e.g., never say "you will definitely get approved"). Frame eligibility information as "based on the typical requirements" rather than a guarantee.
            7. If eligibility data in the context indicates the user does not meet a requirement, or flags a blocker (e.g., outstanding fines), state this clearly and supportively, and explain the next concrete step to resolve it rather than just saying "you're not eligible."
            
            TONE AND STYLE
            - Be warm, clear, and practical — like a knowledgeable, friendly guide explaining a bureaucratic process, not a legal document.
            - Use plain language. Avoid jargon unless it's an official term (e.g., "Emirates ID", "GDRFA") the user needs to know.
            - Structure longer answers with short steps or numbered lists when explaining a process, since these are inherently sequential.
            - Keep tone reassuring but accurate — visa/license processes can be stressful for users, especially tourists or new movers, so be clear without minimizing real requirements like fees or fines.
            - Do not over-elaborate. Answer what was asked, then offer to go deeper ("Want me to walk through the documents needed for step 2?") rather than dumping the entire workflow unprompted.
            
            OUTPUT FORMAT
            - Respond in natural conversational text, not raw JSON.
            - When listing steps, documents, or fees, use a clearly structured short list since this is reference information the user will act on.
            - If a roadmap or checklist has already been generated by the system (provided in context), summarize and explain it in your own words rather than just repeating it verbatim — add helpful framing (e.g., "Since you're currently on a visit visa, here's what changes for you compared to applying from outside the UAE").
            
            DISCLAIMER
            If this is the first response in a conversation, or if the user asks something that suggests they think this is an official government tool, gently clarify: "Just to set expectations — I'm a prototype assistant, not an official UAE government service. Always confirm details with the official source link before taking action."""
        
        
        response = model.generate_content(system_prompt)
        return response.text, context
    except Exception as e:
        return f"Processor Pipeline Error: {str(e)}", None

# --- UI APP LAYER HEADER ---
st.title("Unified UAE Government Services Assistant")
st.caption("AI Agent Engineering Prototype System Framework - RAG Vector Matching Engine v1.1")

# Sidebar Dynamic Credentials Injection Handling Interface
with st.sidebar:
    st.header("🔑 Engine Configurations")
    
    # Secure Configuration: Auto-check for hidden production key first, fall back to field input
    if "GEMINI_API_KEY" in st.secrets:
        api_key_input = st.secrets["GEMINI_API_KEY"]
        st.success("🔒 API Token linked automatically via secure secrets cloud vault.")
    else:
        api_key_input = st.text_input("Enter Google Gemini API Key", type="password", help="Input your free-tier token.")
        if not api_key_input:
            st.info("💡 Running in local manual configuration mode. Paste your key above to proceed.")
            
    st.markdown("---")
    st.markdown("### Trusted Verification Hubs")
    st.markdown("- [Official UAE Portal](https://u.ae)")
    st.markdown("- [ICP Portal](https://icp.gov.ae)")
    st.markdown("- [GDRFA Portal](https://gdrfad.gov.ae)")
    st.markdown("- [RTA Portal](https://rta.ae)")
    st.markdown("- [MOHRE Portal](https://mohre.gov.ae)")

# Chat Memory Array Processing Pipeline Init
if "messages" not in st.session_state:
    st.session_state.messages = []

# Quick Selection Intent Quick-Buttons Category Strip
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
        quick_query = "What is the eligibility for an outstanding student Golden Visa?"

# Process Quick-Button Activation Intent Route
if quick_query:
    st.session_state.messages.append({"role": "user", "content": quick_query})
    matched_docs, context_string = retrieve_context(quick_query, vectorizer, tfidf_matrix, kb_data)
    if not matched_docs:
        reply = "I'm not certain — please verify with official government sources directly on [u.ae](https://u.ae)."
    else:
        reply, _ = generate_grounded_response(quick_query, context_string, api_key_input)
    st.session_state.messages.append({"role": "assistant", "content": reply, "sources": matched_docs})

# Output Historic Session Log Data Segments
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("sources") and msg["role"] == "assistant":
            st.markdown("**Verify on official source:**")
            for src in msg["sources"]:
                st.markdown(f"- [{src['title']}]({src['official_url']})")

# Manual Dynamic Query Input Handling Processing Pipeline Loop
if user_input := st.chat_input("Ask about UAE visas, driving renewals, or business licenses..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.chat_message("assistant"):
        matched_docs, context_string = retrieve_context(user_input, vectorizer, tfidf_matrix, kb_data)
        
        if not matched_docs:
            reply = "I'm not certain — please verify with official government sources directly on [u.ae](https://u.ae)."
            st.write(reply)
            st.session_state.messages.append({"role": "assistant", "content": reply, "sources": []})
        else:
            with st.spinner("Processing local tracking array data metrics..."):
                reply, _ = generate_grounded_response(user_input, context_string, api_key_input)
                st.write(reply)
                st.markdown("**Verify on official source:**")
                for src in matched_docs:
                    st.markdown(f"- [{src['title']}]({src['official_url']})")
                
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": reply, 
                    "sources": matched_docs
                })
