# Medical RAG Application

A Retrieval-Augmented Generation (RAG) application specialized for processing medical textbook PDFs and enabling high-quality medical question answering with multimodal support (text + tables + images) using OpenAI models.

## Key Features

- Medical PDF processing (optimized for large structured clinical/imaging content)
- Multimodal awareness (text, tables, references to images)
- Entity and relation rich context via structured chunking
- English and Chinese query support (examples focus on English)
- Interactive and scripted query modes
- Batch document processing with progress and summary
- Consistent embedding model: `text-embedding-3-small` (1536 dimensions)

## Project Structure
```
medical-rag-app/
├── README.md
├── requirements.txt
├── .env.example
├── data/
│   ├── raw/                  # Original medical PDF textbooks
│   └── indexes/              # Generated index artifacts
├── logs/                     # Processing / query logs
├── output/                   # Per-document processed outputs
├── rag_storage/              # Persistent RAG storage (embeddings, metadata)
├── scripts/
│   ├── medical_rag_processor.py      # Single document processor + examples
│   ├── batch_medical_processor.py    # Batch processing tool
│   └── medical_rag_query.py          # Query-only tool (no re-processing)
└── src/
    └── medical_rag/ (optional module placeholder)
```

## Quick Start

### 1. Python Environment
```bash
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Environment Variables
```bash
cp .env.example .env
# Edit .env and set your key
OPENAI_API_KEY=sk-...
```

### 3. Process a Medical PDF
```bash
python scripts/medical_rag_processor.py --file "data/raw/CT and MRI of the Whole Body, 2-Volume Set, 6e, Volume I ( PDFDrive ).pdf"
```

### 4. Run Query Tool (recommended after initial processing)
```bash
python scripts/medical_rag_query.py --working-dir ./rag_storage
```

### 5. Interactive Query Mode
```bash
python scripts/medical_rag_query.py --working-dir ./rag_storage --interactive
```

## Environment Variables
| Name | Description | Required | Default |
|------|-------------|----------|---------|
| OPENAI_API_KEY | OpenAI API key | Yes | — |
| OPENAI_BASE_URL | Override API base URL (optional) | No | https://api.openai.com/v1 |
| LOG_MAX_BYTES | Log file rotation size | No | 10485760 |
| LOG_BACKUP_COUNT | Rotated log file count | No | 5 |

## Processing Modes
| Mode | Script | Purpose |
|------|--------|---------|
| Single PDF | `medical_rag_processor.py` | Parse + embed + example queries |
| Batch PDFs | `batch_medical_processor.py` | Process all / selected PDFs with concurrency control |
| Query Only | `medical_rag_query.py` | Query previously processed corpus (fast) |

Embedding model is fixed to `text-embedding-3-small` (1536 dim) for both processing and querying to ensure compatibility.

## Usage Examples

### Single Document
```bash
python scripts/medical_rag_processor.py \
  --file "data/raw/Diagnostic Imaging_ Abdomen_ Published by Amirsys® ( PDFDrive ).pdf" \
  --output ./output \
  --parser mineru
```

### Batch Processing
```bash
# Process all PDFs
python scripts/batch_medical_processor.py --all

# Process specific files
python scripts/batch_medical_processor.py --files "CT and MRI of the Whole Body, 2-Volume Set, 6e, Volume I ( PDFDrive ).pdf" "Liver imaging _ MRI with CT correlation ( PDFDrive ).pdf"

# Limit workers (default 2)
python scripts/batch_medical_processor.py --all --max-workers 3
```

### Query Pre-Processed Data
```bash
python scripts/medical_rag_query.py --working-dir ./rag_storage --query "What imaging techniques are discussed?"
```

### Interactive
```bash
python scripts/medical_rag_query.py --working-dir ./rag_storage --interactive
```

## Query Examples
- What is the main content of this medical textbook?
- What imaging techniques are discussed in this document?
- What are the differences between CT and MRI imaging?
- What clinical scenarios are described?

## Outputs
```
output/
└── <document_stem>/
    ├── parsed_content/
    ├── images/
    ├── tables/
    └── metadata.json
```
`rag_storage/` contains persistent structured indexes and embeddings.

## Logs
- Processor log: `logs/medical_rag_processor.log`
- Batch console output includes per-document summary.

## Troubleshooting
| Issue | Symptom | Resolution |
|-------|---------|------------|
| Missing API key | ❌ OpenAI API key not provided | Set OPENAI_API_KEY in `.env` |
| File not found | ❌ File does not exist | Check path / filename correctness |
| Rate limit | ❌ Rate limit reached | Add delay, reduce query frequency |
| Timeout on large PDF | Processing stalls | Retry; ensure network stability |
| Empty answers | Irrelevant response | Rephrase query; ensure document processed |
| Encoding issues | Unicode errors | Ensure UTF-8 locale / filenames |

## Performance Notes
- Embedding model kept small for cost efficiency.
- Keep `--max-workers` ≤ 3 to avoid API concurrency throttling.
- Reuse `medical_rag_query.py` instead of re-processing PDFs.
- Processing large multi-hundred page PDFs may take several minutes.

## Recommended Workflow
1. Place PDFs in `data/raw/`
2. Run single or batch processing
3. Use query tool for iterative exploration
4. Use interactive mode for exploratory Q&A

## Roadmap (Optional Enhancements)
- Add structured evaluation metrics
- Add optional image OCR enrichment
- Add vector DB backend (Chroma / LanceDB) if scaling

## License
Internal / research use. Add explicit license if distributing.

---
Feel free to adapt for your medical RAG experimentation.
