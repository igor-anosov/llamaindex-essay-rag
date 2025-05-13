import os
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, load_index_from_storage
from llama_index.core.node_parser import SentenceSplitter
import gradio as gr
import openai

load_dotenv()

# Increase timeout to handle connection issues
openai.timeout = 60

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# Use your LLM
llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=0.0
)

# Configure your embedding model explicitly
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",  # This is the model name
    deployment_name= "text-embedding-ada-002",  # This should be your embedding deployment name in Azure
    api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

# Path to your data
data_dir = "./data"
index_dir = "./storage"

# Load documents and create index (or load from disk if already exists)
def get_index():
    if os.path.exists(index_dir):
        # Load existing index
        print("Loading existing index...")
        storage_context = StorageContext.from_defaults(persist_dir=index_dir)
        return load_index_from_storage(storage_context, embed_model=embed_model)
    else:
        # Create new index
        print("Creating new index...")
        # Load the document
        documents = SimpleDirectoryReader(data_dir).load_data()
        
        # Split text into chunks
        parser = SentenceSplitter(chunk_size=1024, chunk_overlap=200)
        nodes = parser.get_nodes_from_documents(documents)
        
        # Build index with explicit embedding model
        index = VectorStoreIndex(nodes, embed_model=embed_model)
        
        # Save to disk
        os.makedirs(index_dir, exist_ok=True)
        index.storage_context.persist(persist_dir=index_dir)
        
        return index

# Create index and query engine
print("Setting up the RAG system...")
index = get_index()

# Create a chat engine instead of a query engine
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.chat_engine import ContextChatEngine

memory = ChatMemoryBuffer.from_defaults(token_limit=3900)

# Create a retriever from the index
retriever = index.as_retriever(similarity_top_k=3)

chat_engine = ContextChatEngine.from_defaults(
    index=index,
    retriever=retriever,
    llm=llm,
    memory=memory,
    system_prompt=(
        "You are an AI assistant that has read Paul Graham's essay 'What I Worked On'. "
        "Answer questions about Paul Graham's life, work, and experiences based on the essay. "
        "Be helpful, detailed and accurate. If you don't know something or if it's not mentioned "
        "in the essay, say so. Use a friendly, conversational tone."
    )
)

# Function to process user queries
def process_query(message, history):
    response = chat_engine.chat(message)
    return str(response)

# Create a simple Gradio interface
with gr.Blocks(title="Paul Graham Essay Chat") as demo:
    gr.Markdown("# Chat with Paul Graham's Essay\nAsk questions about 'What I Worked On' by Paul Graham")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(placeholder="Ask about Paul Graham's life, work, and experiences...", 
                    label="Your question")
    clear = gr.Button("Clear Conversation")
    
    def user(message, history):
        return "", history + [[message, None]]
    
    def bot(history):
        response = process_query(history[-1][0], history[:-1])
        history[-1][1] = response
        return history
    
    msg.submit(user, [msg, chatbot], [msg, chatbot], queue=False).then(
        bot, chatbot, chatbot
    )
    
    clear.click(lambda: None, None, chatbot, queue=False)

if __name__ == "__main__":
    # Install gradio if not already installed
    try:
        import gradio
    except ImportError:
        print("Installing gradio...")
        import subprocess
        subprocess.check_call(["pip", "install", "gradio"])
        
    # Set SSL environment variables to avoid certificate issues
    os.environ['SSL_CERT_VERIFY'] = 'false'
    
    print("Starting the chat interface...")
    # Added ssl_verify=False to avoid SSL certificate verification issues
    demo.launch(share=True, inbrowser=True, ssl_verify=False)