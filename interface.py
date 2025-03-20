import os
import re
import sys
import numpy as np
import pandas as pd
import torch
import pinecone
import kagglehub
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.util import ngrams
from dotenv import load_dotenv
from typing import TypedDict, List
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from sklearn.metrics.pairwise import cosine_similarity
from pinecone import Pinecone, ServerlessSpec

# ✅ LangChain & Vector Search
from langchain import hub
from langchain_openai import ChatOpenAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, START, END
from IPython.display import Image, display
from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles

# ✅ Load .env file for API Keys
load_dotenv()

path = kagglehub.dataset_download("mannacharya/blinkit-vs-zepto-vs-instamart-reviews")

# Load dataset into a dataframe
data = pd.read_csv(os.path.join(path,"reviews.csv"))

# Clean the 'review' column by removing new lines
data['review'] = data['review'].str.replace('\n                ', ' ', regex=False)

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size=100,
    chunk_overlap=20,
    length_function=len,
    is_separator_regex=False,
)

# Apply chunking to the 'review' column
data['review_chunks'] = data['review'].apply(lambda x: text_splitter.split_text(x))

# Flatten the chunks into a new DataFrame for further processing
chunked_data = data.explode('review_chunks', ignore_index=True)

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or 'PINECONE_API_KEY'

# configure client
pc = Pinecone(api_key=api_key)

cloud = os.environ.get('PINECONE_CLOUD') or 'aws'
region = os.environ.get('PINECONE_REGION') or 'us-east-1'

spec = ServerlessSpec(cloud=cloud, region=region)

index_name = "stitching"
index = pc.Index(index_name)

embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
tokenizer = embedding_model.tokenizer

class CustomEmbedding:
    def __init__(self, embed_function):
        self.embed_function = embed_function

    def embed_query(self, query):
        return self.embed_function(query)

    def __call__(self, text):
        # Make the object callable
        return self.embed_function(text)
    
# Define the embedding function
def embed_function(query: str):
    """Generate embeddings using a Sentence-Transformer model."""
    with torch.no_grad():
        embeddings = embedding_model.encode([query], convert_to_tensor=True)
    return embeddings[0].cpu().numpy().tolist()

# ✅ Wrap the function in `CustomEmbedding`
custom_embedding = CustomEmbedding(embed_function)

class CustomRetriever(BaseRetriever, BaseModel):
    vectorstore: PineconeVectorStore  # ✅ Update type to match `PineconeVectorStore`
    splits: list  # Keep other fields unchanged
    
    def _get_relevant_documents(self, query: str, k: int = 5):
        # Use vectorstore to retrieve documents
        return self.vectorstore.similarity_search(query, k)
    
# ✅ Corrected LangChain Pinecone integration
vectorstore = PineconeVectorStore(
    index_name=index_name,  # ✅ Use index_name, not index
    pinecone_api_key=os.getenv("PINECONE_API_KEY"),  # ✅ Provide API key
    embedding=custom_embedding,  # ✅ Ensure embedding is an `Embeddings` object
)

retriever = CustomRetriever(vectorstore=vectorstore, splits=chunked_data["review_chunks"])

def format_docs(docs):
    """Format documents by combining content and metadata."""
    formatted_docs = []
    for doc in docs:
        metadata_str = ", ".join(f"{key}: {value}" for key, value in doc.metadata.items())
        formatted_docs.append(f"Content: {doc.page_content}\nMetadata: {metadata_str}")
    return "\n\n".join(formatted_docs)  # Separate documents with a blank line

llm = ChatOpenAI(model="gpt-3.5-turbo", seed=0)

# Load Base Model & Tokenizer
base_model_path = "HuggingFaceTB/SmolLM2-360M-Instruct"
base_tokenizer = AutoTokenizer.from_pretrained(base_model_path)

# Load Base Model (Required for LoRA)
base_model = AutoModelForCausalLM.from_pretrained(base_model_path)

# Load Fine-Tuned LoRA Adapter & Merge with Base Model
lora_path = "./lora_fine_tuned"  # ✅ Path to LoRA fine-tuned weights
lora_model = PeftModel.from_pretrained(base_model, lora_path)

print("✅ LoRA Fine-Tuned Model Loaded Successfully!")


class RAGState(TypedDict):
    """Define State Schema for LangGraph"""
    query: str
    documents: list
    generation: str
    filtered_documents: list

# --------------------------- RETRIEVE DOCUMENTS ---------------------------
def retrieve_documents(state: RAGState):
    """Retrieve relevant documents using Pinecone."""
    print("----RETRIEVE DOCUMENTS----")
    query = state["query"]
    
    try:
        documents = retriever._get_relevant_documents(query, k=5)  # ✅ Use correct retrieval method
    except Exception as e:
        print(f"⚠ Retrieval Error: {str(e)}")
        documents = []

    print(f"✅ Retrieved {len(documents)} documents")
    return {"documents": documents}

# --------------------------- FILTER DOCUMENTS ---------------------------
def filter_documents(state):
    """Remove duplicate or irrelevant documents before passing to generation."""
    print("----FILTER DOCUMENTS----")
    question = state["query"]
    documents = state.get("documents", [])

    if not documents:
        print("⚠ No documents to filter.")
        return {"filtered_documents": []} 

    # ✅ Extract document contents
    doc_texts = [doc.page_content for doc in documents]

    # ✅ Compute embedding similarity (Assume `embed_function` generates vector representations)
    doc_vectors = np.array([embed_function(text) for text in doc_texts])
    similarity_matrix = cosine_similarity(doc_vectors)

    # ✅ Remove highly similar documents (Threshold: 0.85)
    unique_docs = []
    seen_indices = set()
    for i in range(len(doc_texts)):
        if i in seen_indices:
            continue
        unique_docs.append(documents[i])
        seen_indices.update(np.where(similarity_matrix[i] > 0.85)[0])

    print(f"✅ Filtered {len(documents) - len(unique_docs)} redundant documents.")
    return {"filtered_documents": unique_docs}

# --------------------------- GENERATE RESPONSE ---------------------------
def generate_response(state):
    """Generate a response using the retrieved documents."""
    print("----GENERATE RESPONSE----")
    question = state["query"]
    documents = state.get("filtered_documents", [])

    prompt = f"""
    Query: {question}
    Context: {format_docs(documents)}
    
    Provide a detailed response that includes:
    - Key unique features
    - Customer sentiment (positive/negative)
    """
    
    response = llm.invoke(prompt)
    response_text = response.content

    print(f"✅ Generated Response: {response_text}")
    return {"generation": response_text}

# --------------------------- TRANSFORM QUERY ---------------------------
def transform_query(state: RAGState):
    """Transform the query to a better version if documents are not relevant."""
    print("---TRANSFORM QUERY---")
    question = state["query"]
    documents = state["documents"]
    
    # Prompt
    system = """You a question re-writer that converts an input question to a better version that is optimized \n 
         for vectorstore retrieval. Look at the input and try to reason about the underlying semantic intent / meaning."""
    re_write_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                "Here is the initial question: \n\n {question} \n Formulate an improved question.",
            ),
        ]
    )

    question_rewriter = re_write_prompt | llm | StrOutputParser()

    # ✅ Re-write question
    better_question = question_rewriter.invoke({"question": question})

    return {"documents": documents, "query": better_question}

# --------------------------- DETECT HALLUCINATIONS ---------------------------
class GradeHallucinations(BaseModel):
    """Binary score for hallucination detection."""
    binary_score: str = Field(description="Answer is grounded in facts, 'yes' or 'no'")

def detect_hallucinations(state):
    """Check if the generated response is grounded in facts using LoRA."""
    print("----CHECK HALLUCINATIONS (LoRA)----")
    
    question = state["query"]
    documents = state.get("filtered_documents", [])
    generation = state.get("generation", "")

    if not documents:
        print("⚠ No documents retrieved to check hallucinations.")
        return {"hallucination_score": "no_documents"}

    # ✅ Construct hallucination detection prompt for LoRA
    hallucination_prompt = f"""
    Compare the following response with the retrieved documents.
    - If the response **only contains minor missing details that do not alter meaning**, return **"no"**.
    - If the response **includes claims that contradict the retrieved documents**, return **"yes"**.

    Query: {question}
    Retrieved Documents:
    {format_docs(documents)}

    Generated Response:
    {generation}

    Does the response contain hallucinations? (Answer ONLY with 'yes' or 'no'. Write exactly one answer.")
    """

    # ✅ Tokenize and run LoRA model
    inputs = base_tokenizer(hallucination_prompt, return_tensors="pt")
    
    # ✅ Generate hallucination assessment using LoRA
    output_ids = lora_model.generate(
        **inputs,
        max_new_tokens=1,   # ✅ Limit response length
        temperature=0.2,      # ✅ Reduce randomness
        top_p=0.8,            # ✅ Limit unexpected words
        do_sample=True       # ✅ Use deterministic output
    )

    # ✅ Decode and normalize LoRA output
    lora_output = base_tokenizer.decode(output_ids[0], skip_special_tokens=True).strip().lower()
    
    extracted_answer = lora_output[-3:]

    # ✅ Ensure binary response
    hallucination_score = "yes" if "yes" in extracted_answer else "no"

    # ✅ Log result
    print(f"✅ Hallucination Score: {hallucination_score}")
    
    if hallucination_score == "yes":
        print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
        return 'not useful'
    else:
        print("---DECISION: GENERATION ADDRESS QUESTION---")
        return 'useful'

# --------------------------- END NODE ---------------------------
def end_node(state):
    """Final state of the LangGraph RAG system."""
    print(f"✅ Query: {state['query']}\n")
    print(f"📌 Retrieved Context: {state.get('filtered_documents', 'No context available')}\n")
    print(f"📝 Generated Response: {state.get('generation', 'No response generated')}\n")
    return {}

# --------------------------- BUILD RAG GRAPH (LORA) ---------------------------
graph = StateGraph(RAGState)

# ✅ Define Nodes
graph.add_node("retrieve_documents", retrieve_documents)  
graph.add_node("filter_documents", filter_documents)  
graph.add_node("generate_response", generate_response)  
# graph.add_node("detect_hallucinations", detect_hallucinations)  
graph.add_node("transform_query", transform_query)  
graph.add_node("end_node", end_node)  

# ✅ Define Initial Transitions
graph.add_edge(START, "retrieve_documents")  
graph.add_edge("retrieve_documents", "filter_documents")  
graph.add_edge("filter_documents", "generate_response")
# graph.add_edge("generate_response", "detect_hallucinations")  

# ✅ Add Conditional Logic for Hallucination Detection
graph.add_conditional_edges(
    "generate_response",
    detect_hallucinations,  
    {
        "useful": "end_node",  
        "not useful": "transform_query",  
    },
)

# ✅ Handle Query Transformation
graph.add_edge("transform_query", "retrieve_documents")  

# ✅ Compile and Run the RAG Graph
app = graph.compile()

# ✅ Define a function to run the RAG pipeline using command-line arguments
def run_rag_pipeline():
    """Run the multi-agent RAG pipeline using command-line input."""
    
    if len(sys.argv) < 2:
        print("\n⚠️  Usage: python interface.py '<your_query>'\n")
        sys.exit(1)

    input_query = sys.argv[1].strip()

    inputs = {"query": input_query}

    print("\n🔎 Processing query through multi-agent RAG...\n")

    # ✅ Stream responses from the LangGraph RAG system
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Node '{key}':", value, "\n---\n")

# ✅ Run the function when the script is executed
if __name__ == "__main__":
    run_rag_pipeline()
