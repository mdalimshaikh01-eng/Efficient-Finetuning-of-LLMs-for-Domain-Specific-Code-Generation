import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.documents.base import Document

# ---- Hugging Face model for creating embeddings ----
def get_embedding_model():
    '''Returns the embedding model using HuggingFaceBgeEmbeddings with GPU support.'''
    model_name = "BAAI/bge-base-en"
    encode_kwargs = {'normalize_embeddings': True}
    model_norm = HuggingFaceBgeEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cuda'},  # Change 'cpu' to 'cuda' for GPU
        encode_kwargs=encode_kwargs
    )  
    return model_norm

# ---- Chunking the test case file and saving embeddings in FAISS directory ----
def create_vectordb(file_path: str, save_path: str):
    '''Creates a vector database by chunking text separated by three empty lines and saves it to the specified path.'''
    with open(file_path, "r", encoding='utf-8') as f:
        text = f.read()

    # Split the text by one newlines to separate each test case
    split_text = text.split('\n\n\n\n\n\n')

    obj_list = []
    for chunk in split_text:
        # Create Document objects for each chunk
        obj = Document(page_content=chunk.strip())
        obj_list.append(obj)

    # Create the vector database using the embedding model
    model_norm = get_embedding_model()
    db = FAISS.from_documents(documents=obj_list, embedding=model_norm)
    db.save_local(save_path)

# Example Usage:
if __name__ == "__main__":
    # File path to the test cases
    file_path = '/home/ml/ollamawork/output_generated_steps.txt'
    faiss_path = "./faiss-colama_2"

    # Step 1: Create vector database by chunking based on three empty lines
    create_vectordb(file_path, faiss_path)
    print(f"Vector database created and saved to {faiss_path}")

# ---- Load vector database ----
def load_vectordb(save_path: str):
    '''Loads the vector database from the specified path.'''
    model_norm = get_embedding_model()
    vectordb = FAISS.load_local(save_path, model_norm, allow_dangerous_deserialization=True)
    return vectordb

# ---- Retrieving relevant documents from the vector database ----
def retrieve_documents_with_scores(query: str, vectordb: FAISS):
    '''Retrieves documents along with their similarity scores from the vector database.'''

    # Retrieve the top 3 most relevant documents
    docs_with_scores = vectordb.similarity_search_with_score(query, k=3)

    # Display the retrieved documents and their similarity scores
    for doc, score in docs_with_scores:
        print(f"Score: {score:3f}")
        print(f"Document Content:\n{doc.page_content}")
        print("--" * 50)

    return docs_with_scores

# Example Usage:
if __name__ == "__main__":
    # Path to the vector database
    faiss_path = "./faiss-colama_2"

    # Step 1: Check if the vector database exists; if not, display error message
    if os.path.exists(faiss_path):
        vectordb = load_vectordb(faiss_path)
        print("Vector database loaded successfully.")
    else:
        print(f"Error: Vector database not found at {faiss_path}")
        exit()

    # Step 2: Manually input query and retrieve top 3 similar documents
    query = '''
Request Type: GET	
API Endpoint: deploy/rest/deploy/	
Test Case: Validate raw device created by lvm with special character names.  (AGT-7967)	
Output: status_code:200  	
Parameters:
"{
 ""reserve_data_disk"": True
}"
    '''
    retrieve_documents_with_scores(query, vectordb)
