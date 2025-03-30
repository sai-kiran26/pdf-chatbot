import streamlit as st
import os
import time
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import tempfile

# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="ChatBOT")
st.title("Chat With Your PDF Files")
st.markdown("""
    <style>
    div.stButton > button:first-child {
        background-color: #ffd0d0;
    }
    div.stButton > button:active {
        background-color: #ff6262;
    }
    div[data-testid="stStatusWidget"] div button {
        display: none;
    }
    .reportview-container {
        margin-top: -2em;
    }
    #MainMenu {visibility: hidden;}
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    #stDecoration {display:none;}
    button[title="View fullscreen"] {
        visibility: hidden;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize embeddings
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# Process uploaded files
def process_uploaded_files(uploaded_files):
    with st.status("Processing uploaded files..."):
        temp_dir = tempfile.mkdtemp()
        all_docs = []
        
        for uploaded_file in uploaded_files:
            # Save the uploaded file to the temp directory
            temp_file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Load the PDF
            loader = PyPDFLoader(temp_file_path)
            docs = loader.load()
            
            # Add source metadata
            for doc in docs:
                doc.metadata['source'] = uploaded_file.name
            
            all_docs.extend(docs)
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(all_docs)
        
        # Create vector store
        vector_store = FAISS.from_documents(split_docs, embeddings)
        
        # If existing vector store exists, merge with it
        user_vector_store_path = "user_vector_store"
        if os.path.exists(user_vector_store_path) and os.path.isdir(user_vector_store_path):
            try:
                existing_db = FAISS.load_local(user_vector_store_path, embeddings, allow_dangerous_deserialization=True)
                vector_store.merge_from(existing_db)
            except Exception as e:
                st.error(f"Error merging with existing vector store: {e}")
        
        # Save the vector store
        vector_store.save_local(user_vector_store_path)
        st.session_state.db = vector_store
        st.session_state.db_updated = True
        return vector_store

# Reset conversation function
def reset_conversation():
    st.session_state.messages = []
    st.session_state.memory.clear()

# Reset knowledge base function
def reset_knowledge_base():
    user_vector_store_path = "user_vector_store"
    if os.path.exists(user_vector_store_path):
        import shutil
        shutil.rmtree(user_vector_store_path)
    st.session_state.db = None
    st.session_state.db_updated = True
    st.session_state.messages = []
    st.session_state.memory.clear()
    st.success("Knowledge base has been reset.")

# Initialize session state variables
if "messages" not in st.session_state:
    st.session_state.messages = []

if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferWindowMemory(k=2, memory_key="chat_history", return_messages=True)

if "db" not in st.session_state:
    st.session_state.db = None

if "db_updated" not in st.session_state:
    st.session_state.db_updated = False

# File uploader
with st.sidebar:
    st.header("Document Upload")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if uploaded_files:
        if st.button("Process Documents"):
            vector_store = process_uploaded_files(uploaded_files)
            st.success(f"Processed {len(uploaded_files)} document(s)")
    
    st.button('Reset Knowledge Base', on_click=reset_knowledge_base)
    st.button('Reset Conversation', on_click=reset_conversation)

# Load or initialize vector store
if st.session_state.db is None or st.session_state.db_updated:
    user_vector_store_path = "user_vector_store"
    
    # Try to load user vector store first
    if os.path.exists(user_vector_store_path) and os.path.isdir(user_vector_store_path):
        try:
            st.session_state.db = FAISS.load_local(user_vector_store_path, embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading user vector store: {e}")
    
    # If user vector store doesn't exist or couldn't be loaded, try the original vector store
    if st.session_state.db is None and os.path.exists("my_vector_store"):
        try:
            st.session_state.db = FAISS.load_local("my_vector_store", embeddings, allow_dangerous_deserialization=True)
        except Exception as e:
            st.error(f"Error loading original vector store: {e}")
    
    st.session_state.db_updated = False

# Check if we have a vector store to use
if st.session_state.db is not None:
    db_retriever = st.session_state.db.as_retriever(search_type="similarity", search_kwargs={"k": 4})
    
    # Define the prompt template
    prompt_template = """
    <s>[INST]This is a chat template. As a helpful chatbot, your primary objective is to provide accurate and concise information based on the user's questions. Do not generate your own questions and answers. You will adhere strictly to the instructions provided, offering relevant context from the knowledge base while avoiding unnecessary details. Your responses will be brief, to the point, and in compliance with the established format. If a question falls outside the given context, you will refrain from utilizing the chat history and instead rely on your own knowledge base to generate an appropriate response. You will prioritize the user's query and refrain from posing additional questions. The aim is to deliver professional, precise, and contextually relevant information pertaining to context provided.Final answer should be provided only in Telugu.
    CONTEXT: {context}
    CHAT HISTORY: {chat_history}
    QUESTION: {question}
    ANSWER:
    </s>[INST]
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question', 'chat_history'])
    
    # Initialize the LLM
    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
    
    # Set up the QA chain
    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        memory=st.session_state.memory,
        retriever=db_retriever,
        combine_docs_chain_kwargs={'prompt': prompt}
    )
    
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message.get("role")):
            st.write(message.get("content"))
    
    # Input prompt
    input_prompt = st.chat_input("Ask a question about your documents")
    
    if input_prompt:
        with st.chat_message("user"):
            st.write(input_prompt)
        
        st.session_state.messages.append({"role": "user", "content": input_prompt})
        
        with st.chat_message("assistant"):
            with st.status("Thinking 💡...", expanded=True):
                result = qa.invoke(input=input_prompt)
                message_placeholder = st.empty()
                full_response = "\n\n\n"
                
                # If result is a string, convert it to expected format
                if isinstance(result["answer"], str):
                    answer = result["answer"]
                else:
                    # Assume it's iterable
                    answer = ""
                    for chunk in result["answer"]:
                        answer += chunk
                        full_response += chunk
                        time.sleep(0.02)
                        message_placeholder.markdown(full_response + " ▌")
                
                # Make sure final result is displayed
                message_placeholder.markdown(answer)
            
        st.session_state.messages.append({"role": "assistant", "content": answer if "answer" in locals() else result["answer"]})
else:
    st.info("Please upload PDF documents using the sidebar to begin.")