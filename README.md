# CV Parser and Job Matching System

A RAG (Retrieval Augmented Generation) application for parsing, analyzing, and matching CVs to job descriptions using LlamaIndex and ChromaDB.

## Features

- **CV Upload and Processing**: Upload PDF CVs which are processed and indexed using ChromaDB as the vector database.
- **CV Search with Job Descriptions**: Find the most relevant candidates by entering a job description.
- **CV Analysis**: Analyze individual CVs for skills, experience, education, or generate a comprehensive summary.
- **Multi-tab Interface**: User-friendly Gradio interface for different operations.

## Setup

1. **Install dependencies**:

   ```
   pip install -r requirements.txt
   ```

2. **Environment Variables**:
   Create a `.env` file with your Azure OpenAI credentials:

   ```
   OPENAI_API_KEY=your_api_key
   AZURE_OPENAI_ENDPOINT=your_azure_endpoint
   OPENAI_API_VERSION=your_api_version
   ```

3. **Directory Structure**:
   The application will automatically create:
   - `cvs/` - Directory for storing uploaded CVs
   - `chroma_db/` - Directory for ChromaDB vector database

## Usage

Run the application with:

```
python cv_parser.py
```

The application will launch in your default browser with three main tabs:

1. **Upload CV**: Upload and process PDF CVs
2. **Search CVs**: Enter a job description to find matching candidates
3. **Analyze CV**: Select a CV from your database and analyze it for specific information

## How It Works

1. **CV Processing**:

   - PDFs are processed using SimpleDirectoryReader
   - Text is split into chunks using SentenceSplitter
   - Chunks are embedded and stored in ChromaDB

2. **CV Search**:

   - Job descriptions are converted to a search query
   - Vector similarity search finds the most relevant CV chunks
   - Results are synthesized into a comprehensive answer

3. **CV Analysis**:
   - Different analysis types target specific information in the CV
   - The query engine retrieves and summarizes relevant sections

## Customization

- Adjust `chunk_size` and `chunk_overlap` parameters in the code for better text splitting
- Modify the `top_k` parameter to control how many results are returned
- Edit the query prompts in the `search_cvs` and `analyze_cv` functions for different analysis approaches

## License

MIT

## Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Gradio](https://www.gradio.app/) for the web interface
