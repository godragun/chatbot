import os
import streamlit as st
import tempfile
import fitz  # PyMuPDF
import ollama
import random
import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain
import json

# Set page configuration
st.set_page_config(
    page_title="PDF Document Agent",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        padding: 2rem;
    }
    .stApp {
        background-color: #f8f9fa;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton button {
        background-color: #4e73df;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .file-uploader {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .content-box {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .chat-message {
        padding: 1rem;
        margin-bottom: 0.5rem;
        border-radius: 5px;
    }
    .user-message {
        background-color: #e2f0ff;
        text-align: right;
    }
    .assistant-message {
        background-color: #f0f2f5;
    }
    .question-card {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    .mcq-option {
        display: block;
        padding: 0.5rem;
        margin: 0.3rem 0;
        background-color: #f8f9fa;
        border-radius: 5px;
        cursor: pointer;
    }
    .mcq-option:hover {
        background-color: #e9ecef;
    }
    .mcq-selected {
        background-color: #4e73df;
        color: white;
    }
    .card-header {
        background-color: #4e73df;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px 5px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'pdf_text' not in st.session_state:
    st.session_state.pdf_text = ""
if 'summary' not in st.session_state:
    st.session_state.summary = ""
if 'questions' not in st.session_state:
    st.session_state.questions = []
if 'mcqs' not in st.session_state:
    st.session_state.mcqs = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'file_name' not in st.session_state:
    st.session_state.file_name = ""
if 'qa_chain' not in st.session_state:
    st.session_state.qa_chain = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "llama3"
if 'page' not in st.session_state:
    st.session_state.page = "Upload"

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_path = tmp_file.name
    
    doc = fitz.open(tmp_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    doc.close()
    os.unlink(tmp_path)
    
    return text

# Function to get available OLLAMA models
def get_ollama_models():
    try:
        models = ollama.list()
        model_names = [model['name'] for model in models['models']]
        return model_names
    except Exception as e:
        st.error(f"Error fetching OLLAMA models: {e}")
        return ["llama3", "mistral"]  # Default models if unable to fetch

# Function to generate summary
def generate_summary(text, model_name):
    try:
        prompt = f"""
        Please provide a comprehensive summary of the following document. 
        Focus on the main points, key findings, and important details:
        
        {text[:15000]}  # Using first 15000 chars to stay within token limits
        """
        
        response = ollama.generate(model=model_name, prompt=prompt)
        return response['response']
    except Exception as e:
        st.error(f"Error generating summary: {e}")
        return "Failed to generate summary. Please check if OLLAMA is running properly."

# Function to generate questions and answers
def generate_questions(text, model_name, num_questions=5):
    try:
        prompt = f"""
        Based on the following document, generate {num_questions} important questions and their detailed answers.
        Format your response as a JSON array with each object having 'question' and 'answer' keys.
        
        Document:
        {text[:15000]}  # Using first 15000 chars to stay within token limits
        """
        
        response = ollama.generate(model=model_name, prompt=prompt)
        
        # Try to parse the response as JSON
        try:
            # First try to find JSON-like content in the response
            response_text = response['response']
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                questions = json.loads(json_str)
            else:
                # If JSON not found, create a structured format from text
                questions = []
                lines = response_text.split('\n')
                current_question = None
                current_answer = ""
                
                for line in lines:
                    if line.strip().startswith(('Q:', 'Question:')):
                        if current_question:
                            questions.append({
                                'question': current_question,
                                'answer': current_answer.strip()
                            })
                        current_question = line.split(':', 1)[1].strip()
                        current_answer = ""
                    elif line.strip().startswith(('A:', 'Answer:')):
                        current_answer += line.split(':', 1)[1].strip() + " "
                    elif current_question:
                        current_answer += line.strip() + " "
                
                # Add the last question/answer pair
                if current_question:
                    questions.append({
                        'question': current_question,
                        'answer': current_answer.strip()
                    })
                    
            # Ensure we have the requested number of questions
            return questions[:num_questions]
        except Exception as e:
            st.error(f"Error parsing questions: {e}")
            return []
            
    except Exception as e:
        st.error(f"Error generating questions: {e}")
        return []

# Function to generate multiple-choice questions
def generate_mcqs(text, model_name, num_questions=5):
    try:
        prompt = f"""
        Based on the following document, generate {num_questions} multiple-choice questions.
        Each question should have 4 options, with one correct answer.
        Format your response as a JSON array with each object having 'question', 'options' (array of 4 strings), and 'correct_answer' (index 0-3) keys.
        
        Document:
        {text[:15000]}  # Using first 15000 chars to stay within token limits
        """
        
        response = ollama.generate(model=model_name, prompt=prompt)
        
        # Try to parse the response as JSON
        try:
            # First try to find JSON-like content in the response
            response_text = response['response']
            start_idx = response_text.find('[')
            end_idx = response_text.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response_text[start_idx:end_idx]
                mcqs = json.loads(json_str)
            else:
                # If JSON not found, try to create a structured format from text
                mcqs = []
                # Placeholder for MCQ extraction logic
                # This is simplified and may need enhancement for real-world use
                
            # Ensure each MCQ has the required format
            validated_mcqs = []
            for mcq in mcqs:
                if 'question' in mcq and 'options' in mcq and len(mcq['options']) == 4 and 'correct_answer' in mcq:
                    validated_mcqs.append(mcq)
            
            return validated_mcqs[:num_questions]
        except Exception as e:
            st.error(f"Error parsing MCQs: {e}")
            return []
            
    except Exception as e:
        st.error(f"Error generating MCQs: {e}")
        return []

# Function to set up the retrieval chain for chat
def setup_retrieval_chain(text, model_name):
    try:
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)
        
        # Create embeddings and vector store
        embeddings = OllamaEmbeddings(model=model_name)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        
        # Create retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
        
        # Setup LLM
        llm = Ollama(model=model_name)
        
        # Create chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            return_source_documents=True
        )
        
        return qa_chain
    except Exception as e:
        st.error(f"Error setting up retrieval chain: {e}")
        return None

# Navigation sidebar
def sidebar_navigation():
    with st.sidebar:
        st.title("PDF Document Agent")
        
        # Navigation buttons
        if st.button("📄 Upload", use_container_width=True):
            st.session_state.page = "Upload"
        
        if st.button("📝 Summary", use_container_width=True):
            st.session_state.page = "Summary"
            
        if st.button("❓ Questions", use_container_width=True):
            st.session_state.page = "Questions"
            
        if st.button("🔍 Multiple Choice", use_container_width=True):
            st.session_state.page = "Multiple Choice"
            
        if st.button("💬 Chat", use_container_width=True):
            st.session_state.page = "Chat"
        
        st.markdown("---")
        
        # Display current file if any
        if st.session_state.file_name:
            st.success(f"Current file: {st.session_state.file_name}")

# Upload Page
def upload_page():
    st.title("Upload PDF Document")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("<div class='file-uploader'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf", key="pdf_uploader")
        
        available_models = get_ollama_models()
        model_name = st.selectbox("Select OLLAMA Model", options=available_models, index=0)
        st.session_state.selected_model = model_name
        
        if uploaded_file is not None:
            if st.button("Process PDF"):
                with st.spinner("Processing PDF..."):
                    # Extract text from PDF
                    text = extract_text_from_pdf(uploaded_file)
                    st.session_state.pdf_text = text
                    st.session_state.file_name = uploaded_file.name
                    
                    # Generate summary
                    st.session_state.summary = generate_summary(text, model_name)
                    
                    # Generate questions
                    st.session_state.questions = generate_questions(text, model_name)
                    
                    # Generate MCQs
                    st.session_state.mcqs = generate_mcqs(text, model_name)
                    
                    # Setup QA chain for chat
                    st.session_state.qa_chain = setup_retrieval_chain(text, model_name)
                    
                    st.success("PDF processed successfully! Navigate to other sections to view results.")
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Instructions")
        st.write("1. Upload a PDF document")
        st.write("2. Select an OLLAMA model")
        st.write("3. Click 'Process PDF'")
        st.write("4. Navigate to other sections to view generated content")
        st.markdown("</div>", unsafe_allow_html=True)
        
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.subheader("Features")
        st.write("• Automatic summary generation")
        st.write("• Question and answer extraction")
        st.write("• Multiple-choice questions")
        st.write("• Interactive chat with document")
        st.markdown("</div>", unsafe_allow_html=True)

# Summary Page
def summary_page():
    st.title("Document Summary")
    
    if st.session_state.summary:
        st.markdown("<div class='content-box'>", unsafe_allow_html=True)
        st.markdown("### Summary")
        st.write(st.session_state.summary)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Word count and statistics
        word_count = len(st.session_state.pdf_text.split())
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Word Count", f"{word_count:,}")
        with col2:
            st.metric("Pages (est.)", f"{max(1, word_count // 500)}")
        with col3:
            st.metric("Reading Time", f"{max(1, word_count // 200)} min")
            
    else:
        st.info("Please upload and process a PDF document first.")

# Questions Page
def questions_page():
    st.title("Questions & Answers")
    
    if st.session_state.questions:
        for i, qa in enumerate(st.session_state.questions):
            with st.expander(f"Q{i+1}: {qa['question']}"):
                st.write(qa['answer'])
    else:
        st.info("Please upload and process a PDF document first.")

# Multiple Choice Page
def mcq_page():
    st.title("Multiple Choice Questions")
    
    if st.session_state.mcqs:
        # Create state for tracking answers if not exists
        if 'mcq_answers' not in st.session_state:
            st.session_state.mcq_answers = [-1] * len(st.session_state.mcqs)
        if 'mcq_submitted' not in st.session_state:
            st.session_state.mcq_submitted = False
            
        for i, mcq in enumerate(st.session_state.mcqs):
            st.markdown(f"<div class='question-card'>", unsafe_allow_html=True)
            st.markdown(f"<div class='card-header'>Question {i+1}</div>", unsafe_allow_html=True)
            st.markdown(f"### {mcq['question']}")
            
            # Radio buttons for options
            answer = st.radio(
                "Select your answer:",
                mcq['options'],
                key=f"mcq_{i}"
            )
            
            # Record answer
            st.session_state.mcq_answers[i] = mcq['options'].index(answer) if answer in mcq['options'] else -1
            
            st.markdown("</div>", unsafe_allow_html=True)
        
        # Submit button
        if st.button("Submit Answers"):
            st.session_state.mcq_submitted = True
            
        # Show results if submitted
        if st.session_state.mcq_submitted:
            correct_count = 0
            for i, mcq in enumerate(st.session_state.mcqs):
                user_answer = st.session_state.mcq_answers[i]
                correct_answer = mcq['correct_answer']
                
                if isinstance(correct_answer, str) and correct_answer.isdigit():
                    correct_answer = int(correct_answer)
                
                if user_answer == correct_answer:
                    correct_count += 1
            
            st.success(f"You got {correct_count} out of {len(st.session_state.mcqs)} correct!")
            
            # Reset button
            if st.button("Try Again"):
                st.session_state.mcq_submitted = False
                st.session_state.mcq_answers = [-1] * len(st.session_state.mcqs)
                st.experimental_rerun()
    else:
        st.info("Please upload and process a PDF document first.")

# Chat Page
def chat_page():
    st.title("Chat with Document")
    
    if st.session_state.qa_chain:
        # Display chat history
        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:  # User message
                st.markdown(f"<div class='chat-message user-message'><strong>You:</strong> {message}</div>", unsafe_allow_html=True)
            else:  # Assistant message
                st.markdown(f"<div class='chat-message assistant-message'><strong>PDF Agent:</strong> {message}</div>", unsafe_allow_html=True)
        
        # Chat input
        user_question = st.text_input("Ask a question about the document:", key="chat_input")
        
        if user_question:
            if st.button("Send"):
                with st.spinner("Thinking..."):
                    # Add user question to history
                    st.session_state.chat_history.append(user_question)
                    
                    # Get chat history in the format expected by the chain
                    chain_history = []
                    for i in range(0, len(st.session_state.chat_history)-1, 2):
                        if i+1 < len(st.session_state.chat_history):
                            chain_history.append((st.session_state.chat_history[i], st.session_state.chat_history[i+1]))
                    
                    # Get response from chain
                    result = st.session_state.qa_chain({"question": user_question, "chat_history": chain_history})
                    answer = result["answer"]
                    
                    # Add response to history
                    st.session_state.chat_history.append(answer)
                    
                    # Rerun to update the UI
                    st.experimental_rerun()
    else:
        st.info("Please upload and process a PDF document first.")

# Main application
def main():
    # Show sidebar navigation
    sidebar_navigation()
    
    # Display the selected page
    if st.session_state.page == "Upload":
        upload_page()
    elif st.session_state.page == "Summary":
        summary_page()
    elif st.session_state.page == "Questions":
        questions_page()
    elif st.session_state.page == "Multiple Choice":
        mcq_page()
    elif st.session_state.page == "Chat":
        chat_page()

if __name__ == "__main__":
    main()
