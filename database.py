import os
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up environment variables
load_dotenv()
os.environ['GOOGLE_API_KEY'] = os.getenv("GOOGLE_API_KEY")

# Load and embed the documents
def embed_and_save_documents(directory="./LEGAL-DATA", output_path="my_vector_store"):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        logging.info(f"Loading documents from {directory}")
        
        # Check if directory exists
        if not os.path.exists(directory):
            logging.error(f"Directory {directory} does not exist")
            return False
            
        loader = PyPDFDirectoryLoader(directory)
        logging.info("Loader initialized")
        
        docs = loader.load()
        logging.info(f"Loaded {len(docs)} documents")
        
        if not docs:
            logging.warning("No documents were loaded")
            return False
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        logging.info(f"Split into {len(final_documents)} chunks")
        
        # Ensure metadata includes the source file name
        for doc in final_documents:
            if 'source' in doc.metadata:
                source_file = doc.metadata['source']
                doc.metadata['source'] = os.path.basename(source_file)
            else:
                # If source metadata is not present, add it
                doc.metadata['source'] = os.path.basename(loader.directory)
        
        # Ensure the payload size is within limits by batching the documents
        batch_size = 100  # Adjust batch size as needed
        batched_documents = [final_documents[i:i + batch_size] for i in range(0, len(final_documents), batch_size)]
        logging.info(f"Created {len(batched_documents)} batches of documents")
        
        vector_stores = []
        for i, batch in enumerate(batched_documents):
            logging.info(f"Processing batch {i+1}/{len(batched_documents)}")
            vector_store = FAISS.from_documents(batch, embeddings)
            vector_stores.append(vector_store)
        
        # Merge the vector stores
        vectors = vector_stores[0]
        for vector_store in vector_stores[1:]:
            vectors.merge_from(vector_store)
        logging.info("Merged the vectors")
        
        # Save the vector store to disk
        vectors.save_local(output_path)
        logging.info(f"Vectors saved to {output_path}")
        
        return True
    
    except Exception as e:
        logging.error(f"Error in embed_and_save_documents: {str(e)}")
        return False

if __name__ == "__main__":
    success = embed_and_save_documents()
    if success:
        print("Successfully processed and saved documents")
    else:
        print("Failed to process documents")