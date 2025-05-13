# RAG Chat Interface for Paul Graham's Essay

This application provides a chat interface that allows users to ask questions about Paul Graham's essay "What I Worked On". The system uses Retrieval-Augmented Generation (RAG) to provide accurate answers based on the content of the essay.

## Features

- 💬 Interactive chat interface using Gradio
- 🔍 Semantic search using Azure OpenAI Embeddings
- 🤖 Context-aware responses using GPT-4o through Azure OpenAI
- 📝 Persistent vector index storage for faster startup
- 🧠 Memory buffer to maintain conversation context

## Demo

![RAG Chat Demo](https://github.com/igor-anosov/llamaindex-essay/raw/main/assets/demo.gif)

## Setup

### Prerequisites

- Python 3.10+ (tested with Python 3.10.13)
- Azure OpenAI API access with deployments for:
  - GPT-4o (or other chat completion model)
  - text-embedding-ada-002 (for embeddings)

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/igor-anosov/llamaindex-essay.git
   cd llamaindex-essay
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Azure OpenAI credentials:
   ```
   OPENAI_API_KEY=your_azure_openai_api_key
   AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
   OPENAI_API_VERSION=2023-05-15
   ```

### Running the Application

1. Make sure the essay file is in the `data` directory:

   ```bash
   mkdir -p data
   # Place paul_graham_essay.txt in the data directory
   ```

2. Run the application:

   ```bash
   python main.py
   ```

3. Open your browser at the URL shown in the terminal (typically http://127.0.0.1:7860)

> **Note**: The first time you run the app, it will generate an index of the essay which may take a few moments. Subsequent runs will be faster as the index is stored locally.

## How It Works

1. **Data Ingestion**: The system reads and processes Paul Graham's essay, splitting it into manageable chunks.

2. **Embedding Generation**: Each chunk is converted into a vector embedding using Azure OpenAI's embedding model.

3. **Query Processing**: When a user asks a question, the system:

   - Converts the question to an embedding
   - Finds the most relevant chunks from the essay using vector similarity search
   - Sends those chunks along with the question to GPT-4o
   - Returns the generated response to the user

4. **Memory**: The system maintains context of the conversation, allowing for follow-up questions.

## Troubleshooting

If you encounter SSL or connection issues:

- The code already includes fixes for common SSL certificate verification issues
- Ensure your Azure OpenAI deployments are correctly named and match what's in the code
- Check that your API key and endpoint are correct in the .env file

## Deployment Options

Note that GitHub Pages only supports static websites, not Python applications. To share this with others, consider:

1. **Local Demo**:

   - Run the app locally and share your screen
   - Host a locally-tunneled version using ngrok

2. **Cloud Deployment**:

   - [Hugging Face Spaces](https://huggingface.co/spaces) (free tier available)
   - [Streamlit Cloud](https://streamlit.io/cloud) (good for data apps)
   - [Railway](https://railway.app) or [Render](https://render.com) (easy deployment)

3. **For Quick Demos**:
   - Record a short video showing the functionality
   - Create screenshots of the chat interface

## License

MIT

## Acknowledgements

- [LlamaIndex](https://www.llamaindex.ai/) for the RAG framework
- [Gradio](https://www.gradio.app/) for the web interface
- Paul Graham for the essay "What I Worked On"
