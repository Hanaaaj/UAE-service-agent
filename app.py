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
        return "⚠️ Configuration Blocked: Missing API Token parameter. Supply token parameter string via application dashboard console or secrets configuration manifest.", None
        
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel('gemini-2.5-flash')
        
        system_prompt = (
            "You are a helpful conversational UAE Government Services Assistant dashboard system.\n"
            "Your main task is to guide users through rules using ONLY the validated context parameters below.\n\n"
            "CRITICAL OPERATIONAL RULES:\n"
            "1. Ground information delivery only within the specified facts block. Do not extrapolate data.\n"
            "2. If the facts block does not explicitly contain structural answers to resolve the prompt, or if sources are ambiguous, precisely output: "
            "'I'm not certain — please verify with [official link]' where the link corresponds exactly to the official link found in the metadata.\n"
            "3. Append standard tag notation '(Subject to change)' next to any mentioned monetary figures.\n\n"
            f"VALIDATED DATA PARAMETERS:\n{context}\n"
            f"USER QUERY PROMPT: {query}"
        )
        
        response = model.generate_content(system_prompt)
        return response.text, context
    except Exception as e:
        return f"Processor Pipeline Exception Error Encountered: {str(e)}", None

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
