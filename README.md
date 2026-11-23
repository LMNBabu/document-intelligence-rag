# Document Intelligence RAG Assistant

Upload PDF and DOCX files, then ask questions. Answers come **only** from your documents
and show the source passages, so you can verify them.

**Stack:** Python 3.11+, Streamlit, PyMuPDF, python-docx, Sentence Transformers
(`all-MiniLM-L6-v2`), ChromaDB (persistent), Groq API, python-dotenv.

---

## Quick start (Windows PowerShell)

### Option A: one command

```powershell
cd path\to\document-intelligence-rag
powershell -ExecutionPolicy Bypass -File .\setup_and_run.ps1
```

This checks your Python version, creates `.venv`, installs dependencies, creates `.env`
(and opens it so you can add your Groq key and model), runs the self-check, and starts the app.
Flags: `-SkipChecks`, `-NoRun`, `-Live` (also test the real model and Groq).

### Option B: manual

```powershell
cd path\to\document-intelligence-rag
py -3.11 -m venv .venv                 # any Python 3.11+ works; use `py --list` to see yours
.\.venv\Scripts\Activate.ps1
# If activation is blocked:  Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt        # large: pulls in PyTorch, takes a few minutes
Copy-Item .env.example .env
notepad .env                           # set GROQ_API_KEY and GROQ_MODEL
python check_project.py                # offline self-check
python -m streamlit run app.py
```

The app opens at http://localhost:8501.

### Configure `.env`

| Variable | Purpose | Default |
|---|---|---|
| `GROQ_API_KEY` | Your key from https://console.groq.com/keys | *(required)* |
| `GROQ_MODEL` | Groq model name. Pick a **currently listed** one at https://console.groq.com/docs/models (Groq retires models over time) | *(required)* |
| `EMBEDDING_MODEL` | Sentence Transformers model | `all-MiniLM-L6-v2` |
| `CHROMA_DIR` | Where the vector database is stored | `./chroma_db` |
| `COLLECTION_NAME` | Chroma collection name | `documents` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Chunk size and overlap, in characters | `1000` / `200` |
| `TOP_K` | Default number of passages retrieved | `4` |
| `MAX_FILE_MB` | Upload size limit | `20` |

Restart the app after editing `.env`. If you change `EMBEDDING_MODEL`, click **Clear all
documents** and re-index, because vectors from different models cannot be compared.

---

## Sample testing workflow

1. `python check_project.py`: expect `26/26 checks passed` (offline, no downloads, no API calls).
2. `python check_project.py --live`: downloads the embedding model on first run (~90 MB), tests
   retrieval, and makes one real Groq call. Needs a valid `.env`.
3. Start the app and upload a PDF you know well (a few pages). Click **Index documents**.
   The status table should say *Indexed* with a chunk count.
4. Click **Index documents** again with the same file: status should be *Skipped, Already indexed*.
5. Ask something the document answers. Check that the cited file/page match where the answer really is.
6. Ask something it does **not** cover (e.g. "What is the capital of Australia?"). Expected reply:
   *"The documents do not contain sufficient information to answer this question."*
7. Upload a DOCX and a corrupt file (rename a `.txt` to `.pdf`): the corrupt one should show *Failed*
   with a readable reason, and the others still index.
8. Close the app, start it again: indexed documents are still listed (persistence).
9. Click **Clear all documents**: the list empties.

---

## How data flows

```
INDEXING (Index documents button)
 uploaded bytes
   -> file_hash()              fingerprint; skip if already stored        (vector_store.py)
   -> load_document()          list of PageContent(file, "Page 3", text)  (document_loader.py)
   -> split_pages()            list of Chunk(text, file, page, index)     (text_splitter.py)
   -> embed_texts()            one 384-number vector per chunk            (embeddings.py)
   -> vector_store.add_chunks  text + vector + metadata saved to disk     (vector_store.py)

QUERYING (chat box)
 question
   -> retrieve()               embed question (same model), search Chroma (retriever.py)
   -> RetrievedChunk list      original text + file + page + similarity
   -> generate_answer()        numbered passages + strict rules -> Groq   (llm.py)
   -> app.py                   shows answer; maps [1],[2] back to real chunks
```

## The modules in plain language

- **`document_loader.py`**: opens PDFs (PyMuPDF) and DOCX files (python-docx) and returns text with a
  location label. PDFs get real page numbers. DOCX files have no stored pages, so paragraphs are
  grouped into "Section N" blocks of about one page; citations say *Section*, never a fake page.
  Every problem (wrong type, empty, corrupt, password-protected, scanned) raises `DocumentLoadError`.
- **`text_splitter.py`**: **chunking.** Cuts each page into pieces of about `CHUNK_SIZE` characters,
  preferring paragraph/sentence boundaries. Neighbouring chunks share `CHUNK_OVERLAP` characters so
  an answer is not cut in half. Chunks never cross pages, so page citations are exact.
- **`embeddings.py`**: **embeddings.** Converts text into a vector that represents its meaning. Similar
  meaning means nearby vectors. The same cached model embeds documents and questions.
- **`vector_store.py`**: **vector storage.** ChromaDB on disk. Stores chunk text, vector and metadata.
  Does cosine **similarity search**, duplicate detection by content hash, and clearing. Uploading a
  changed file with the same name replaces the old version.
- **`retriever.py`**: embeds the question and returns the top-k closest chunks as `RetrievedChunk`s.
- **`llm.py`**: **LLM integration.** Sends the question and numbered passages to Groq with rules:
  answer only from the passages, say "insufficient information" otherwise, never invent. Citations
  are numbers; the app maps them to stored metadata, and numbers the model made up are flagged.
- **`app.py`**: the Streamlit interface that wires it together.
- **`check_project.py`**: self-check. **`setup_and_run.ps1`**: one-step setup and launch.

## Known limitations

- Scanned PDFs (images) are rejected: there is no OCR.
- DOCX tables, headers, footers and text boxes are not extracted (body paragraphs only).
- Each question is answered independently: the chat does not remember earlier turns.
- Retrieval returns the top-k chunks even for unrelated questions; the LLM's rules then make it
  decline. Check the similarity numbers in *Retrieved context* if an answer seems off.
- `list_documents` reads all metadata; fine for tens of documents, not for huge libraries.

## Common errors and fixes

| Symptom | Cause | Fix |
|---|---|---|
| `running scripts is disabled on this system` | PowerShell execution policy | `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`, or run the setup script with `-ExecutionPolicy Bypass` as shown above |
| `py` is not recognised | Python launcher not installed | Install Python 3.11+ from python.org with "Add to PATH", reopen PowerShell |
| `ModuleNotFoundError` | Venv not active, or install incomplete | `.\.venv\Scripts\Activate.ps1`, then `pip install -r requirements.txt` |
| `streamlit` is not recognised | Venv not active | Use `python -m streamlit run app.py` |
| Missing in .env warning / "Missing setting(s)" | Key or model not set | Fill `GROQ_API_KEY` and `GROQ_MODEL` in `.env`, restart |
| "Groq rejected the API key" | Wrong or revoked key | Create a new key at console.groq.com/keys |
| "Groq does not recognise the model" | Model name wrong or retired | Choose a current name from console.groq.com/docs/models |
| "Groq rate limit reached" | Free-tier limits | Wait and retry |
| "Could not load embedding model" | No internet on first run, or firewall blocks huggingface.co | Connect once so the model downloads; it is cached afterwards |
| First indexing is very slow | Model download + CPU encoding | Normal the first time |
| "has no extractable text" | Scanned PDF | Run OCR first (e.g. in Adobe Acrobat) and re-upload |
| "password-protected" | Encrypted PDF | Remove the password and re-upload |
| "could not be opened as a DOCX" | Corrupt file or old `.doc` renamed | Open in Word, Save As `.docx` |
| Answers refer to old content | Embedding model was changed | Clear all documents and re-index |
| `Could not open the vector database` | `CHROMA_DIR` not writable | Point it to a writable folder |
| Port 8501 busy | Another Streamlit running | `python -m streamlit run app.py --server.port 8502` |
