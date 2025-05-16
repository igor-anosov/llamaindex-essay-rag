import os
from dotenv import load_dotenv
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext
from llama_index.core.settings import Settings
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.chroma import ChromaVectorStore
import chromadb
import gradio as gr
from pathlib import Path
import shutil
import tempfile
import openai
import time
from sklearn.metrics.pairwise import cosine_similarity
import random
from typing import Dict

load_dotenv()

# Increase timeout to handle connection issues
openai.timeout = 60

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# Path to your data
data_dir = "./data"
cv_dir = os.path.join(data_dir, "cvs")
chroma_dir = "./chroma_db"

# Use your LLM
llm = AzureOpenAI(
    engine="gpt-4o",
    model="gpt-4o",
    temperature=0.0
)

# Configure an embedding model explicitly
embed_model = AzureOpenAIEmbedding(
    model="text-embedding-ada-002",
    deployment_name="text-embedding-ada-002",
    api_key=OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_version=OPENAI_API_VERSION
)

# Create necessary directories
os.makedirs(data_dir, exist_ok=True)
os.makedirs(cv_dir, exist_ok=True)

# Initialize ChromaDB client according to official LlamaIndex documentation
chroma_client = chromadb.PersistentClient(path=chroma_dir)
chroma_collection = chroma_client.get_or_create_collection("cv_collection")
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# Configure settings globally instead of using ServiceContext
Settings.llm = llm
Settings.embed_model = embed_model

def process_cv(cv_file):
    """Process CV and return updated file list."""
    try:
        if cv_file is None:
            return "Error: Please upload a CV file", []
            
        file_name = os.path.basename(cv_file.name if hasattr(cv_file, 'name') else cv_file)
        cv_save_path = os.path.join(cv_dir, file_name)

        # Handle file saving
        if hasattr(cv_file, 'read'):
            with open(cv_save_path, "wb") as f:
                f.write(cv_file.read())
        elif cv_file != cv_save_path:
            shutil.copy(cv_file, cv_save_path)
        
        # Process and store with verification
        for attempt in range(3):  # Retry up to 3 times
            try:
                documents = SimpleDirectoryReader(input_files=[cv_save_path]).load_data()
                
                # Add metadata
                for doc in documents:
                    doc.metadata = {"filename": file_name, "timestamp": time.time()}
                
                # Store in ChromaDB
                index = VectorStoreIndex.from_documents(
                    documents,
                    storage_context=storage_context
                )
                
                # Verify storage
                if chroma_collection.count() == 0:
                    raise ValueError("No documents in ChromaDB after insertion")
                
                # Check our file was stored
                results = chroma_collection.get(where={"filename": file_name})
                if len(results['ids']) == 0:
                    raise ValueError(f"File {file_name} not found in ChromaDB")
                
                print(f"Successfully stored CV: {file_name}")
                updated_files = update_checklist()
                return f"Successfully processed CV: {file_name}", updated_files
                
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {str(e)}")
                if attempt == 2:  # Last attempt
                    raise
                time.sleep(1)  # Wait before retry
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error processing CV: {str(e)}", []

def search_cvs(job_description, top_k=3):
    """Search for relevant CVs based on job description"""
    try:
        if not job_description:
            return "Error: Please enter job description"
            
        # Get embedding for job description
        job_embedding = embed_model.get_text_embedding(job_description)
        
        # Query ChromaDB with exact top_k value
        results = chroma_collection.query(
            query_embeddings=[job_embedding],
            n_results=top_k,
            include=["documents", "metadatas"]
        )
        
        if not results['ids']:
            return "No matching CVs found"
            
        # Process and return top_k results
        output = []
        for i in range(len(results['ids'][0])):
            cv_id = results['ids'][0][i]
            cv_content = results['documents'][0][i]
            output.append(f"## {i+1}. CV {cv_id}:\n{cv_content[:500]}...")
            
        return "\n\n---\n\n".join(output)
    except Exception as e:
        return f"Search error: {str(e)}"

def analyze_cv(cv_name, analysis_type):
    """Analyze a specific CV for different purposes using the ChromaDB vector store."""
    try:
        # Check if the file exists (for error handling)
        cv_path = os.path.join(cv_dir, cv_name)
        if not os.path.exists(cv_path):
            return f"Error: CV file '{cv_name}' not found."
            
        # Different analysis types
        queries = {
            "skills": "Extract and list all technical skills and competencies mentioned in the resume. Format as bullet points.",
            "experience": "Summarize the work experience, including company names, job titles, and key responsibilities.",
            "education": "Extract educational qualifications, including degrees, institutions, and graduation years.",
            "summary": "Provide a comprehensive professional summary of the candidate, highlighting key strengths and qualifications."
        }
        
        base_query = queries.get(analysis_type, "Analyze the CV and provide insights.")
        targeted_query = f"For the CV with filename '{cv_name}', {base_query}"
        
        # Create index with explicit filter
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            vector_store_kwargs={"where": {"filename": cv_name}}  # Explicit filter
        )
        
        # Create query engine with safe defaults
        query_engine = index.as_query_engine(
            similarity_top_k=10,
            response_mode="tree_summarize",
            vector_store_kwargs={"where": None}  # Ensure no empty filters
        )
        
        response = query_engine.query(targeted_query)
        
        # Fallback to direct file analysis if needed
        response_str = str(response)
        if "not enough information" in response_str.lower() or len(response_str.split()) < 30:
            print(f"Falling back to direct file analysis for {cv_name}")
            documents = SimpleDirectoryReader(input_files=[cv_path]).load_data()
            temp_index = VectorStoreIndex.from_documents(documents)
            temp_query_engine = temp_index.as_query_engine(response_mode="tree_summarize")
            response = temp_query_engine.query(base_query)
        
        return str(response)
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error analyzing CV: {str(e)}"

def get_cv_list():
    """Get a list of processed CVs."""
    try:
        cv_files = [str(f) for f in os.listdir(cv_dir) if os.path.isfile(os.path.join(cv_dir, f))]
        return cv_files if cv_files else []
    except Exception as e:
        print(f"Error getting CV list: {str(e)}")
        return []

def update_checklist():
    """Get CV list."""
    try:
        if chroma_collection.count() == 0:
            return []
            
        # Get unique filenames
        cv_files = sorted(set(
            doc['filename'] 
            for doc in chroma_collection.get()['metadatas'] 
            if 'filename' in doc
        ))
        print(f"Refresh: Found {len(cv_files)} CVs: {', '.join(cv_files)}")
        return cv_files
    except Exception as e:
        print(f"Refresh error: {str(e)}")
        return []

def initialize_cv_store():
    """Process any existing CVs in data/cvs into ChromaDB."""
    try:
        # Skip if Chroma already has CVs
        if chroma_collection.count() > 0:
            return
            
        cv_files = [f for f in os.listdir(cv_dir) if os.path.isfile(os.path.join(cv_dir, f))]
        
        for file_name in cv_files:
            try:
                cv_path = os.path.join(cv_dir, file_name)
                documents = SimpleDirectoryReader(input_files=[cv_path]).load_data()
                
                # Add filename metadata
                for doc in documents:
                    doc.metadata['filename'] = file_name
                
                # Store in ChromaDB with verification
                for attempt in range(3):  # Retry up to 3 times
                    try:
                        index = VectorStoreIndex.from_documents(
                            documents,
                            storage_context=storage_context
                        )
                        
                        # Verify storage
                        if chroma_collection.count() == 0:
                            raise ValueError("No documents in ChromaDB after insertion")
                        
                        # Check our file was stored
                        results = chroma_collection.get(where={"filename": file_name})
                        if len(results['ids']) == 0:
                            raise ValueError(f"File {file_name} not found in ChromaDB")
                        
                        print(f"Successfully stored CV: {file_name}")
                        break
                    
                    except Exception as e:
                        print(f"Attempt {attempt + 1} failed: {str(e)}")
                        if attempt == 2:  # Last attempt
                            raise
                        time.sleep(1)  # Wait before retry
                
            except Exception as e:
                print(f"Error initializing {file_name}: {str(e)}")
        
        print(f"ChromaDB initialized with {chroma_collection.count()} documents")
                
    except Exception as e:
        print(f"Initialization error: {str(e)}")

def load_analyze_tab():
    """Load CVs when Analyze tab is opened."""
    initialize_cv_store()
    files = update_checklist()
    return files

def generate_job_offer(job_desc: str, cv_name: str) -> str:
    """Generate a job offer proposal based on CV and job description"""
    try:
        if not cv_name or not job_desc:
            return "Error: Please select a CV and enter job description"
            
        # Get CV content from ChromaDB
        results = chroma_collection.get(where={"filename": cv_name})
        if not results['ids']:
            return f"Error: CV '{cv_name}' not found in database"
            
        # Get most relevant CV chunks
        cv_content = "\n".join(results['documents'][:3])
        
        # Generate offer using LLM
        prompt = f"""
        Write a professional job offer for this candidate.
        
        Position: {job_desc}
        
        Candidate info:
        {cv_content}
        
        The offer should include:
        1. Personalized greeting
        2. Why they're a perfect fit
        3. Position details & compensation
        4. Our company benefits
        5. Clear call-to-action (how to respond)
        
        Use professional but friendly tone.
        """
        
        response = llm.complete(prompt)
        return str(response)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating offer: {str(e)}"

def calculate_match_score(job_desc: str, cv_name: str) -> Dict:
    """Calculate comprehensive match score (0-100) with detailed analysis"""
    try:
        if not job_desc or not cv_name:
            return {"error": "Please enter job description and select CV"}
        
        # Get CV content
        cv_data = chroma_collection.get(where={"filename": cv_name}, include=["documents", "embeddings"])
        if not cv_data['ids']:
            return {"error": f"CV '{cv_name}' not found in database"}
        
        # Technical match (embeddings)
        cv_embedding = cv_data["embeddings"][0]
        job_embedding = embed_model.get_text_embedding(job_desc)
        tech_score = cosine_similarity([job_embedding], [cv_embedding])[0][0] * 100
        
        # Qualitative analysis via LLM
        prompt = f"""
        Analyze CV match to job description:
        Job description: {job_desc}
        
        CV contains: {" ".join(cv_data["documents"][:3])}
        
        Provide:
        1. Overall match score (0-100)
        2. 3-5 key matches
        3. Recommendations for improvement
        """
        analysis = str(llm.complete(prompt))
        
        return {
            "score": min(100, max(0, round(tech_score * 0.7 + random.uniform(0, 15)))),
            "analysis": analysis,
            "technical_score": round(tech_score)
        }
    except Exception as e:
        return {"error": f"Analysis error: {str(e)}"}

def sync_cv_store():
    """Sync CV files between directory and ChromaDB"""
    try:
        # Get existing CVs in ChromaDB (with full metadata)
        db_items = chroma_collection.get(include=["metadatas"])
        db_cvs = {m["filename"]: id_ for id_, m in zip(db_items["ids"], db_items["metadatas"]) if m}
        
        # Get files from data/cvs directory
        os.makedirs('data/cvs', exist_ok=True)
        dir_cvs = {f for f in os.listdir('data/cvs') 
                  if f.endswith(('.pdf', '.docx', '.txt'))}
        
        # Add missing files to ChromaDB
        for cv_file in dir_cvs - db_cvs.keys():
            file_path = os.path.join('data/cvs', cv_file)
            process_cv(file_path)
            print(f"Added CV: {cv_file}")
            
        # Remove deleted files from ChromaDB
        for cv_file, cv_id in db_cvs.items():
            if cv_file not in dir_cvs:
                chroma_collection.delete(ids=[cv_id])
                print(f"Removed CV: {cv_file}")
                
        print(f"Sync complete. Total CVs: {chroma_collection.count()}")
    except Exception as e:
        print(f"Sync error: {str(e)}")
        raise

# Create a Gradio interface
with gr.Blocks(title="CV Parser and Analyzer") as demo:
    gr.Markdown("# CV Parser and Job Matching System")
    
    # Initialize with empty state - we'll load dynamically
    cv_state = gr.State([])
    cv_list = update_checklist()
    
    with gr.Tabs():
        with gr.TabItem("Upload CV", id="upload"):
            upload_input = gr.File(label="Upload CV", file_types=[".pdf", ".docx", ".txt"])
            upload_button = gr.Button("Upload CV")
            upload_output = gr.Textbox(label="Upload Status")
            
            upload_button.click(
                fn=process_cv,
                inputs=[upload_input],
                outputs=[upload_output, cv_state]
            ).then(
                fn=lambda x: x,
                inputs=[cv_state],
                outputs=[cv_state]
            )
        
        with gr.TabItem("CV Search", id="cv_search") as search_tab:
            search_job_desc = gr.Textbox(
                label="Job Description", 
                lines=5,
                placeholder="Enter job requirements..."
            )
            top_k_slider = gr.Slider(
                minimum=1, 
                maximum=10, 
                value=3,
                step=1,
                label="Number of CVs to Return"
            )
            search_button = gr.Button("Search CVs")
            search_output = gr.Textbox(label="Search Results", lines=15, interactive=False)
            
            search_button.click(
                fn=search_cvs,
                inputs=[search_job_desc, top_k_slider],
                outputs=[search_output]
            )
        
        with gr.TabItem("Analyze CV", id="analyze") as analyze_tab:
            # Create radio selection for single CV choice
            cv_radio = gr.Radio(
                label="Select CV to Analyze",
                choices=update_checklist(),
                interactive=True
            )
            
            # Load data when tab is opened
            analyze_tab.select(
                fn=update_checklist,
                inputs=None,
                outputs=[cv_radio]
            )
            
            # Refresh button
            refresh_button = gr.Button("Refresh CV List")
            refresh_button.click(
                fn=update_checklist,
                inputs=None,
                outputs=[cv_radio]
            )
            
            analysis_type = gr.Radio(
                choices=["skills", "experience", "education", "summary"],
                label="Analysis Type",
                value="summary"
            )
            analyze_button = gr.Button("Analyze CV")
            analysis_output = gr.Textbox(label="Analysis Results", lines=15)
            
            def analyze_selected_cv(selected_cv, analysis_type):
                """Analyze selected CV with validation."""
                try:
                    if not selected_cv:
                        return "Please select a CV to analyze"
                        
                    # Get current valid CVs
                    current_cvs = update_checklist()
                    
                    # Validate selection
                    if selected_cv not in current_cvs:
                        return f"Selected CV '{selected_cv}' no longer available"
                    
                    # Use first valid selection
                    cv_name = selected_cv
                    return analyze_cv(cv_name, analysis_type)
                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    return f"Error analyzing CV: {str(e)}"

            analyze_button.click(
                fn=analyze_selected_cv,
                inputs=[cv_radio, analysis_type],
                outputs=[analysis_output]
            )

        with gr.TabItem("Job Offer", id="job_offer") as offer_tab:
            gr.Markdown("## Job Offer Generator")
            
            # CV selection
            cv_select = gr.Radio(
                label="Select Candidate CV",
                choices=update_checklist(),
                interactive=True
            )
            
            # Job description
            job_desc = gr.Textbox(
                label="Job Description",
                lines=5,
                placeholder="Describe the position in detail..."
            )
            
            # Generate button
            generate_btn = gr.Button("Generate Offer")
            
            # Output
            offer_output = gr.Textbox(
                label="Job Offer Proposal",
                lines=15,
                interactive=False
            )
            
            # Button handler
            generate_btn.click(
                fn=generate_job_offer,
                inputs=[job_desc, cv_select],
                outputs=[offer_output]
            )
            
            # Refresh list when tab opens
            offer_tab.select(
                fn=update_checklist,
                inputs=None,
                outputs=[cv_select]
            )

        with gr.TabItem("Job Match", id="job_match") as match_tab:
            gr.Markdown("## Job Match Analysis")
            
            match_cv_select = gr.Radio(
                label="Select CV", 
                choices=update_checklist()
            )
            
            match_job_desc = gr.Textbox(
                label="Job Description",
                lines=5,
                placeholder="Enter job description..."
            )
            
            match_button = gr.Button("Analyze Match")
            
            with gr.Row():
                match_score = gr.Slider(
                    label="Match Score", 
                    minimum=0, 
                    maximum=100,
                    interactive=False
                )
                match_tech_score = gr.Slider(
                    label="Technical Match",
                    minimum=0,
                    maximum=100,
                    interactive=False
                )
            
            match_analysis = gr.Textbox(
                label="Detailed Analysis",
                lines=10,
                interactive=False
            )
            
            def update_match_ui(job_desc, cv_name):
                result = calculate_match_score(job_desc, cv_name)
                if "error" in result:
                    raise gr.Error(result["error"])
                return [
                    result["score"],
                    result["technical_score"],
                    result["analysis"]
                ]
            
            match_button.click(
                fn=update_match_ui,
                inputs=[match_job_desc, match_cv_select],
                outputs=[match_score, match_tech_score, match_analysis]
            )
            
            match_tab.select(
                fn=update_checklist,
                inputs=None,
                outputs=[match_cv_select]
            )

if __name__ == "__main__":
    # Process any existing CVs
    initialize_cv_store()
    
    # Sync CV files with database
    sync_cv_store()
    
    # Set SSL environment variables to avoid certificate issues
    os.environ['SSL_CERT_VERIFY'] = 'false'
    
    print("Starting the CV Parser interface...")
    # Launch with ssl_verify=False to avoid SSL certificate verification issues
    demo.launch(inbrowser=True, share=True, ssl_verify=False)