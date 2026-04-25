# Chat with PDF

A Streamlit RAG application for asking questions across one or more PDF files.
The app extracts PDF text with PyMuPDF, chunks it with LangChain, embeds chunks
with OpenAI `text-embedding-ada-002`, stores them in FAISS, and answers with
OpenAI `gpt-4o-mini` through a conversational retrieval chain.

## Features

- Upload and index multiple PDFs in one shared FAISS vector store.
- Preserve source metadata for each chunk, including document name and page.
- Ask follow-up questions with chat history.
- View retrieved source chunks below each assistant answer.
- Remove indexed PDFs, clear chat history, or reset the app state.
- Works locally with `.env` and on Streamlit Cloud with `st.secrets`.

## Local Setup

1. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

   On Windows PowerShell:

   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Create your local environment file:

   ```bash
   cp .env.example .env
   ```

4. Add your OpenAI API key to `.env`:

   ```bash
   OPENAI_API_KEY=sk-your-openai-api-key-here
   ```

5. Run the app:

   ```bash
   streamlit run app.py
   ```

## Streamlit Cloud Deployment

1. Push this `chat-with-pdf` folder to a GitHub repository.
2. In Streamlit Cloud, create a new app from the repository.
3. Set the main file path to:

   ```text
   app.py
   ```

   If this project is inside a larger repository, set the path to:

   ```text
   chat-with-pdf/app.py
   ```

4. In **Advanced settings**, select Python `3.12`.

5. Leave the **Secrets** field empty if users should provide their own OpenAI
   API key in the sidebar.

6. Deploy. Streamlit Cloud installs `requirements.txt` automatically.

## Notes

- Scanned PDFs without embedded text may need OCR before the app can index them.
- If the model cannot find an answer in the retrieved chunks, it is instructed to
  say: `I couldn't find that information in the uploaded documents.`
- The sidebar API key field is only used when `st.secrets["OPENAI_API_KEY"]` and
  the `OPENAI_API_KEY` environment variable are not available.
