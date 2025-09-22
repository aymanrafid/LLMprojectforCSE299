# LLMprojectforCSE299


# Bangla PDF Processor

A comprehensive tool for extracting, summarizing, and interacting with Bangla (Bengali) PDF documents. This application combines OCR technology, natural language processing, and AI to make Bangla PDF content accessible and interactive.

## Features

- **Text Extraction**: Extract Bangla text from PDF files, including image-based content using OCR technology
- **Summarization**: Generate concise summaries of lengthy documents using Google's Gemini AI model
- **Question Answering**: Ask questions about document content and receive accurate responses
- **Dual Interface**: Both command-line and graphical user interfaces available
- **Batch Processing**: Efficiently handle large documents with intelligent batch processing
- **Rate Limiting**: Built-in rate limiting to handle API constraints gracefully
- **Session Management**: Save and load processing sessions for later use

## Installation
   Requirements:pip install PyMuPDF pytesseract Pillow tiktoken tenacity langchain langchain_google_genai langchain_community google-generativeai faiss-cpu                  tqdm nest-asyncio

### Prerequisites

- Python 3.8 or higher
- Google Gemini API key (get one at [Google AI Studio](https://makersuite.google.com/app/apikey))
- Tesseract OCR with Bangla language data



### Install Tesseract OCR

**Windows:**
1. Download installer from [Tesseract at UB Mannheim](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install with Bangla language support
3. Update the path in `config.json` if needed



## Configuration

Create a `config.json` file in the project root:

```json
{
  "tesseract_path": "C:\\Program Files\\Tesseract-OCR\\tesseract.exe",
  "ocr_dpi": 300,
  "max_input_tokens": 900000,
  "max_tokens_per_chunk": 100000,
  "max_output_tokens": 8192,
  "model_name": "models/gemini-2.0-flash",
  "batch_size": 3,
  "rate_limit_delay": 2.0,
  "batch_delay": 5.0,
  "chunk_delay": 3.0,
  "retry_attempts": 3,
  "retry_min_wait": 30,
  "retry_max_wait": 90
}
```

## Usage

### Command Line Interface

```bash
python bangla_pdf_processor.py --pdf path/to/document.pdf --api-key YOUR_API_KEY --mode all
```

#### Options:
- `--pdf`: Path to the Bangla PDF file (required)
- `--api-key`: Google Gemini API key (required)
- `--output-dir`: Directory to save output files (default: "output")
- `--mode`: Processing mode - extract, summarize, qa, or all (default: "all")
- `--config`: Path to configuration JSON file
- `--session`: Load previous session (timestamp)

#### Examples:

Extract text only:
```bash
python bangla_pdf_processor.py --pdf document.pdf --api-key YOUR_API_KEY --mode extract
```

Summarize document:
```bash
python bangla_pdf_processor.py --pdf document.pdf --api-key YOUR_API_KEY --mode summarize
```

Interactive Q&A:
```bash
python bangla_pdf_processor.py --pdf document.pdf --api-key YOUR_API_KEY --mode qa
```

Full processing:
```bash
python bangla_pdf_processor.py --pdf document.pdf --api-key YOUR_API_KEY --mode all
```

### Graphical User Interface

Launch the GUI application:
```bash
python bangla_pdf_ui.py
```

#### GUI Features:
1. **File Selection**: Browse and select PDF files
2. **API Key Input**: Securely enter your Google Gemini API key
3. **Processing Modes**: Choose between extraction, summarization, Q&A, or all
4. **Output Directory**: Select where to save results
5. **Progress Tracking**: Real-time progress updates and processing logs
6. **Results Tabs**: View extracted text, summaries, and Q&A sessions
7. **Save Results**: Save extracted text, summaries, and Q&A sessions to files

## Project Structure

```
├── bangla_pdf_processor.py  # Core processing functionality
├── bangla_pdf_ui.py         # Graphical user interface
├── config.json              # Configuration file
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── output/                  # Default output directory
    ├── extracted_bangla_text.txt
    ├── summarized_bangla_text.txt
    ├── qa_output.txt
    ├── vectorstore_*
    └── memory_*.json
```

## Dependencies

- PyMuPDF (fitz) - PDF processing
- pytesseract - OCR functionality
- Pillow - Image processing
- langchain - AI integration framework
- langchain_google_genai - Google Generative AI integration
- FAISS - Vector similarity search
- tiktoken - Token counting
- tenacity - Retry mechanisms
- tqdm - Progress bars
- tkinter - GUI framework (usually included with Python)

## How It Works

1. **Text Extraction**:
   - First attempts direct text extraction from PDF
   - Falls back to OCR for image-based content
   - Uses Tesseract with Bangla language support

2. **Summarization**:
   - Splits text into manageable chunks
   - Processes chunks in batches to avoid rate limits
   - Uses custom prompts optimized for Bangla
   - Combines partial summaries into a comprehensive summary

3. **Question Answering**:
   - Creates vector embeddings of document chunks
   - Stores embeddings in a FAISS vector store
   - Retrieves relevant passages for each question
   - Generates context-aware answers using Gemini AI

## Team Members

- **Kazi Ayman Rafid Sachcha**


## Acknowledgments

- Google for the Gemini AI model and Tesseract OCR
- LangChain community for the framework
- FAISS team for efficient similarity search
- North South University for supporting this project

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the project
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/your-repo/bangla-pdf-processor/issues) page
2. Create a new issue with detailed information
3. Contact the development team

---

**Note**: This tool is specifically designed for Bangla (Bengali) language documents. While it may work with other languages supported by Tesseract, optimal performance is guaranteed only for Bangla content.
