import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
import re
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv


MAX_PDF_SIZE_MB = 10
PDF_FOLDER = "pdfs"
SUPPORT_EMAIL = "support@example.com"


load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


LAW_ENTRANCE_KEYWORDS = [
    "clat", "ailet", "lsat", "slat", "mht cet law", "law entrance", "law exam", "law exams",
    "law colleges", "nlus", "nlu", "syllabus", "pattern", "admit card", "result", "application",
    "eligibility", "reservation", "cutoff", "counselling", "exam date", "registration",
    "constitution", "law", "legal", "court", "judiciary", "rights", "parliament",
    "contract", "criminal", "tort", "ipc", "crpc", "evidence", "gk", "current affairs",
    "english", "logical reasoning", "quantitative", "aptitude", "legal studies"
]
GENERAL_CONVO_KEYWORDS = [
    "hello", "hi", "hey", "how are you", "good morning", "good evening", "good night",
    "thank you", "thanks", "bye", "see you", "what's up", "how's it going", "who are you",
    "your name", "help", "can you help", "nice to meet you"
]


def ensure_pdf_folder():
    if not os.path.exists(PDF_FOLDER):
        os.makedirs(PDF_FOLDER)

def sanitize_filename(filename):
    return re.sub(r'[^a-zA-Z0-9_.-]', '_', filename)

def list_pdfs():
    ensure_pdf_folder()
    return [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith('.pdf')]

def save_uploaded_pdfs(uploaded_files):
    ensure_pdf_folder()
    for uploaded_file in uploaded_files:
        if uploaded_file.size > MAX_PDF_SIZE_MB * 1024 * 1024:
            st.error(f"{uploaded_file.name} exceeds the {MAX_PDF_SIZE_MB}MB limit and was not saved.")
            continue
        file_path = os.path.join(PDF_FOLDER, sanitize_filename(uploaded_file.name))
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

def get_pdf_text_from_files(pdf_filenames):
    text = ""
    for filename in pdf_filenames:
        path = os.path.join(PDF_FOLDER, filename)
        try:
            pdf_reader = PdfReader(path)
            for page in pdf_reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text
        except Exception as e:
            st.error(f"Failed to process {filename}: {e}")
   
    return text.lower()

def get_text_chunks(text):

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    prompt_template = """
    You are Lexa, a formal and professional assistant. Use the conversation history and provided context to answer follow-up questions.
    If the answer is not in the provided context but is related to law entrance exams (CLAT, AILET, etc.), answer from your knowledge.
    If the question is general conversation, respond politely and formally.
    If you cannot answer, politely state that you are unable to assist with that query.

    Conversation history:
    {context}

    Context from PDF:
    {context}

    Question:
    {question}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def is_law_exam_related(question):
    q = question.lower()
    return any(word in q for word in LAW_ENTRANCE_KEYWORDS)

def is_general_conversation(question):
    q = question.lower()
    return any(word in q for word in GENERAL_CONVO_KEYWORDS)

def get_general_conversation_reply(question):
    q = question.lower()
    if "hello" in q or "hi" in q or "hey" in q:
        return "Hello! How may I assist you today?"
    if "how are you" in q:
        return "I am an AI assistant and do not possess feelings, but I am here to help you."
    if "thank" in q:
        return "You are most welcome. If you have any further queries, please let me know."
    if "bye" in q or "see you" in q:
        return "Goodbye! If you have more questions in the future, feel free to return."
    if "your name" in q:
        return "My name is Lexa, your formal assistant for law entrance exam and PDF queries."
    if "help" in q:
        return "Certainly! You may ask me questions about your uploaded PDFs or law entrance exams such as CLAT, AILET, and more."
    return "I am here to assist you with your queries regarding law entrance exams and your uploaded documents."

def build_conversation_context(messages, max_turns=5):
    context = ""
    relevant_msgs = messages[-max_turns*2:] if max_turns else messages
    for msg in relevant_msgs:
        if msg["role"] == "user":
            context += f"User: {msg['content']}\n"
        else:
            context += f"Lexa: {msg['content']}\n"
    return context


def main():
    st.set_page_config("Lexa PDF Chatbot", page_icon="📚", layout="wide")
    # Gradient animated background
    st.markdown(
        '''
        <style>
        body {
            background: linear-gradient(135deg, #4F8BF9 0%, #A770EF 100%) fixed;
        }
        .glass-container {
            background: rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 24px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            padding: 2rem 2.5rem;
            margin: 2rem auto 2rem auto;
            max-width: 700px;
        }
        .chat-bubble-user {
            background: linear-gradient(90deg, #4F8BF9 60%, #A770EF 100%);
            color: #fff;
            border-radius: 18px 18px 4px 18px;
            padding: 1rem 1.2rem;
            margin: 1rem 0 1rem auto;
            max-width: 80%;
            box-shadow: 0 2px 12px rgba(79,139,249,0.15);
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 1.1rem;
            transition: all 0.3s;
        }
        .chat-bubble-assistant {
            background: rgba(255,255,255,0.85);
            color: #222;
            border-radius: 18px 18px 18px 4px;
            padding: 1rem 1.2rem;
            margin: 1rem auto 1rem 0;
            max-width: 80%;
            box-shadow: 0 2px 12px rgba(167,112,239,0.10);
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            font-size: 1.1rem;
            transition: all 0.3s;
        }
        .floating-input {
            position: fixed;
            bottom: 2.5rem;
            left: 50%;
            transform: translateX(-50%);
            width: 90vw;
            max-width: 700px;
            z-index: 100;
            background: rgba(255,255,255,0.7);
            border-radius: 18px;
            box-shadow: 0 4px 24px rgba(79,139,249,0.10);
            padding: 1rem 1.5rem;
        }
        .sidebar-content {
            font-family: 'Segoe UI', 'Roboto', sans-serif;
            color: #333;
        }
        .sidebar-section {
            margin-bottom: 1.5rem;
            border-bottom: 1px solid #e0e0e0;
            padding-bottom: 1rem;
        }
        .welcome-anim {
            animation: fadeInDown 1.2s cubic-bezier(.39,.575,.56,1.000) both;
        }
        @keyframes fadeInDown {
            0% { opacity: 0; transform: translateY(-40px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 800px) {
            .glass-container, .floating-input { max-width: 98vw; padding: 1rem 0.5rem; }
            .chat-bubble-user, .chat-bubble-assistant { font-size: 1rem; }
        }
        </style>
        ''', unsafe_allow_html=True
    )
    
    st.markdown(
        '''
        <div class="welcome-anim" style="text-align: center; margin-top: 1.5rem;">
            <img src="https://img.icons8.com/ios-filled/100/4F8BF9/law.png" width="60" alt="Law Icon"/>
            <h1 style="color: #4F8BF9; margin-bottom: 0; font-family: 'Segoe UI', 'Roboto', sans-serif;">Lexa PDF Chatbot</h1>
            <p style="color: #888; margin-top: 0; font-size: 1.1rem;">Your formal assistant for Law Entrance Exams and PDF Q&A</p>
        </div>
        ''', unsafe_allow_html=True
    )
    st.markdown("<div style='height: 1.5rem'></div>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "pdfs_processed" not in st.session_state:
        st.session_state.pdfs_processed = False

    with st.sidebar:
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        st.title("📂 PDF Selection")
        st.info("Upload your own PDF files or select from existing ones in the `pdfs/` folder. Max file size: 10MB per file.")
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        uploaded_files = st.file_uploader(
            "Upload your PDF(s)", type=["pdf"], accept_multiple_files=True
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} file(s) ready to be saved and processed.")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        pdf_files = list_pdfs()
        selected_pdfs = st.multiselect("Select PDF(s) to use", pdf_files, default=pdf_files[:1])
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        if st.button("✨ Process Selected PDFs / Uploaded PDFs"):
            if uploaded_files:
                save_uploaded_pdfs(uploaded_files)
                pdf_files = list_pdfs()
                selected_pdfs = [sanitize_filename(f.name) for f in uploaded_files]
            if not selected_pdfs:
                st.warning("Please select at least one PDF to process.")
            else:
                with st.spinner("Processing selected PDFs..."):
                    try:
                        raw_text = get_pdf_text_from_files(selected_pdfs)
                        if not raw_text.strip():
                            st.error("No extractable text found in the selected PDF(s).")
                        else:
                            text_chunks = get_text_chunks(raw_text)
                            get_vector_store(text_chunks)
                            st.success("PDFs processed and indexed!")
                            st.session_state.messages = []
                            st.session_state.pdfs_processed = True
                    except Exception as e:
                        st.error(f"Failed to process PDF(s): {e}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown(f"**Developed by Your Team**  <br>For support, contact: <a href='mailto:{SUPPORT_EMAIL}'>{SUPPORT_EMAIL}</a>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

  
    st.markdown(
        '''
        <div style="max-width:700px;margin:0 auto 2rem auto;padding:1.5rem 2rem;background:rgba(255,255,255,0.10);border-radius:18px;box-shadow:0 2px 16px rgba(79,139,249,0.10);font-family:'Segoe UI','Roboto',sans-serif;">
            <h2 style="color:#4F8BF9;margin-bottom:0.5rem;font-size:1.5rem;">Welcome to Lexa PDF Chatbot!</h2>
            <ul style="color:#eee;font-size:1.08rem;margin:0 0 0.5rem 1.2rem;padding:0;">
                <li>Ask questions about your uploaded PDFs or law entrance exams (CLAT, AILET, etc.).</li>
                <li>Follow-up questions are supported—Lexa remembers your conversation context.</li>
                <li>For best results, process your PDFs before starting the chat.</li>
            </ul>
            <div style="color:#b3b3b3;font-size:0.98rem;margin-top:0.5rem;">🔒 <b>Privacy:</b> Your files and questions are processed securely and are not shared with third parties.</div>
        </div>
        ''', unsafe_allow_html=True
    )
    st.markdown("### Start your conversation below")

    prompt = st.chat_input("Type your question here and press Enter", key="crazy_input")

    if prompt:
        if len(prompt) > 500:
            st.error("Please limit your question to 500 characters.")
        else:
            normalized_prompt = prompt.lower()
            st.session_state.messages.append({"role": "user", "content": prompt})
            conversation_context = build_conversation_context(st.session_state.messages[:-1], max_turns=5)
            if st.session_state.get("pdfs_processed", False):
                embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
                try:
                    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
                    docs = new_db.similarity_search(normalized_prompt)
                except Exception as e:
                    st.error(f"Error loading vector store: {e}")
                    docs = []
                if not docs or not docs[0].page_content.strip():
                    if is_law_exam_related(prompt):
                        chain = get_conversational_chain()
                        response = chain(
                            {
                                "input_documents": [],
                                "question": prompt,
                                "context": conversation_context
                            },
                            return_only_outputs=True
                        )
                        reply = response["output_text"]
                    elif is_general_conversation(prompt):
                        reply = get_general_conversation_reply(prompt)
                    else:
                        reply = "I am sorry, but I am unable to assist with that query."
                else:
                    chain = get_conversational_chain()
                    response = chain(
                        {
                            "input_documents": docs,
                            "question": prompt,
                            "context": conversation_context
                        },
                        return_only_outputs=True
                    )
                    reply = response["output_text"]
            else:
                if is_law_exam_related(prompt):
                    chain = get_conversational_chain()
                    response = chain(
                        {
                            "input_documents": [],
                            "question": prompt,
                            "context": conversation_context
                        },
                        return_only_outputs=True
                    )
                    reply = response["output_text"]
                elif is_general_conversation(prompt):
                    reply = get_general_conversation_reply(prompt)
                else:
                    reply = "I am sorry, but I am unable to assist with that query."
            st.session_state.messages.append({"role": "assistant", "content": reply})

   
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"<div class='chat-bubble-user'><b>You:</b><br>{msg['content']}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='chat-bubble-assistant'><b>Lexa:</b><br>{msg['content']}</div>", unsafe_allow_html=True)


    st.markdown(
        "<hr><div style='text-align:center; color:#aaa; font-size:13px;'>© 2024 Lexa PDF Chatbot. All rights reserved.</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
