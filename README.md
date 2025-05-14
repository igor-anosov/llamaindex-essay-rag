# CV Parser and Job Matching System

A RAG (Retrieval Augmented Generation) application for parsing, analyzing, and matching CVs to job descriptions using LlamaIndex and ChromaDB.

## Features

1. **CV Upload**:
   - PDF and DOCX files are processed and stored in ChromaDB
   - Automatic text extraction and chunking
   - Metadata preservation (filename, upload time)

2. **CV Search**:
   - Job descriptions are converted to a search query
   - Vector similarity search finds the most relevant CV chunks
   - Results are synthesized into a comprehensive answer

3. **CV Analysis**:
   - Single CV selection via radio buttons
   - Multiple analysis types (skills, experience, education, summary)
   - Dynamic refresh of available CVs
   - The query engine retrieves and summarizes relevant sections

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

- **Upload**: Add new CVs via the upload tab
- **Search**: Enter a job description to find matching CVs
- **Analyze**:
  - Select a single CV from the radio list
  - Choose analysis type
  - Click "Analyze CV"
  - Use "Refresh CV List" to update available CVs

Run the application with:

```
python cv_parser.py
```

The application will launch in your default browser with three main tabs.

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

- Adjust `chunk_size` and `chunk_overlap` parameters for better text splitting
- Modify the `top_k` parameter to control search results
- Edit query prompts in the functions for different analysis approaches
- Set `persist_dir` in ChromaDB for persistent storage

## License

MIT

## Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Gradio](https://www.gradio.app/) for the web interface
- [ChromaDB](https://www.trychroma.com/) for vector storage
