import fitz  # PyMuPDF library for PDF processing
import pytesseract
from PIL import Image
import io
import time
import os
import re
import tiktoken
import random
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from typing import List, Dict, Any, Optional, Tuple
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime
import asyncio  # Add this import
import nest_asyncio  # Add this import

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# LangChain imports
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from google.api_core.exceptions import TooManyRequests, InvalidArgument

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bangla_pdf_processor.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_CONFIG = {
    "tesseract_path": r'C:\Program Files\Tesseract-OCR\tesseract.exe',
    "ocr_dpi": 300,
    "max_input_tokens": 900000,
    "max_tokens_per_chunk": 100000,
    "max_output_tokens": 8192,
    "model_name": 'models/gemini-2.0-flash',
    "batch_size": 3,
    "rate_limit_delay": 2.0,
    "batch_delay": 5.0,
    "chunk_delay": 3.0,
    "retry_attempts": 3,
    "retry_min_wait": 30,
    "retry_max_wait": 90
}

# --- Constants for token management ---
MAX_INPUT_TOKENS = DEFAULT_CONFIG["max_input_tokens"]
MAX_TOKENS_PER_CHUNK = DEFAULT_CONFIG["max_tokens_per_chunk"]
MAX_OUTPUT_TOKENS = DEFAULT_CONFIG["max_output_tokens"]

# --- Token counting setup ---
try:
    encoding = tiktoken.encoding_for_model("gpt-4")
    def count_tokens(text):
        return len(encoding.encode(text))
    tiktoken_available = True
except ImportError:
    def count_tokens(text):
        return len(text) // 3
    tiktoken_available = False

class BanglaPDFProcessor:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._set_tesseract_path()
        self.llm = None
        self.embeddings = None
        self.text_splitter = None
        self.memory = None
        self.qa_chain = None
        self.vectorstore = None
        
    def _set_tesseract_path(self):
        """Set Tesseract OCR path"""
        try:
            pytesseract.pytesseract.tesseract_cmd = self.config["tesseract_path"]
            logger.info(f"Tesseract path set to: {self.config['tesseract_path']}")
        except Exception as e:
            logger.error(f"Error setting Tesseract path: {e}")
            raise
            
    @retry(
        stop=stop_after_attempt(DEFAULT_CONFIG["retry_attempts"]),
        wait=wait_exponential(multiplier=1, min=DEFAULT_CONFIG["retry_min_wait"], max=DEFAULT_CONFIG["retry_max_wait"]),
        retry=retry_if_exception_type((TooManyRequests, InvalidArgument))
    )
    def _generate_with_retry(self, *args, **kwargs):
        """Custom LLM generation with retry logic"""
        return self.llm._generate(*args, **kwargs)
        
    def extract_bangla_text_from_pdf(self, pdf_path: str, output_filename: str = "extracted_bangla_text.txt") -> Tuple[str, Optional[str], float, str]:
        """Extracts Bangla text from all pages of a PDF, using OCR for image-based text."""
        full_text_list = []
        start_time = time.time()
        error_message = None
        output_file_path = None
        extracted_full_text = ""
        
        try:
            if not os.path.exists(pdf_path):
                raise FileNotFoundError(f"The PDF file was not found at: {pdf_path}")
                
            doc = fitz.open(pdf_path)
            logger.info(f"Starting text extraction from '{os.path.basename(pdf_path)}'...")
            logger.info(f"Processing {len(doc)} page(s)...")
            
            for page_num in tqdm(range(len(doc)), desc="Extracting text from PDF"):
                page = doc.load_page(page_num)
                full_text_list.append(f"--- Page {page_num + 1} ---\n")
                
                text_content = page.get_text("text")
                if text_content.strip():
                    full_text_list.append(text_content)
                else:
                    pix = page.get_pixmap(matrix=fitz.Matrix(self.config["ocr_dpi"] / 72, self.config["ocr_dpi"] / 72))
                    img_bytes = pix.tobytes("png")
                    img = Image.open(io.BytesIO(img_bytes))
                    ocr_text = pytesseract.image_to_string(img, lang='ben')
                    full_text_list.append(ocr_text)
                
                full_text_list.append("\n\n")
            
            doc.close()
            extracted_full_text = "".join(full_text_list)
            output_file_path = output_filename
            
            with open(output_file_path, 'w', encoding='utf-8') as f:
                f.write(extracted_full_text)
                
            end_time = time.time()
            runtime = end_time - start_time
            message = f"OCR process complete. Extracted text saved to '{output_file_path}'."
            message += f"\nOCR Runtime: {runtime:.2f} seconds."
            return message, output_file_path, runtime, extracted_full_text
            
        except FileNotFoundError as e:
            error_message = str(e)
        except fitz.FileDataError:
            error_message = "Invalid or corrupted PDF file. Please check the file."
        except pytesseract.TesseractNotFoundError:
            error_message = "Tesseract OCR engine not found. Please ensure Tesseract is installed and added to your system's PATH, or specify its path in the script."
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            
        end_time = time.time()
        runtime = end_time - start_time
        return f"Error: {error_message}\nRuntime before error: {runtime:.2f} seconds.", None, runtime, ""
        
    def create_langchain_components(self, api_key: str):
        """Initialize LangChain components for summarization and Q&A."""
        # Ensure we have an event loop in this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Initialize LLM with retry capability
        self.llm = ChatGoogleGenerativeAI(
            model=self.config["model_name"],
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=MAX_OUTPUT_TOKENS
        )
        
        # Initialize embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=api_key
        )
        
        # Create text splitter with optimized parameters
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=500,
            length_function=count_tokens if tiktoken_available else len
        )
        
        # Create memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        
        logger.info("LangChain components initialized successfully")
        
    def rate_limited_api_call(self, func, *args, **kwargs):
        """Wrapper to add rate limiting to API calls with longer delays"""
        max_retries = self.config["retry_attempts"]
        for attempt in range(max_retries):
            try:
                result = func(*args, **kwargs)
                # Add a delay after successful API calls to avoid rate limits
                time.sleep(self.config["rate_limit_delay"])
                return result
            except TooManyRequests:
                # Wait for a random time before retrying
                wait_time = random.uniform(self.config["retry_min_wait"], self.config["retry_max_wait"])
                logger.warning(f"Rate limit exceeded. Waiting {wait_time:.2f} seconds before retrying (attempt {attempt + 1}/{max_retries})...")
                time.sleep(wait_time)
            except InvalidArgument as e:
                if "API key expired" in str(e):
                    logger.error("Error: API key has expired. Please generate a new API key.")
                    raise e
                raise e
        raise Exception(f"Failed after {max_retries} attempts")
        
    def summarize_with_langchain(self, text: str) -> str:
        """Summarize text using a more robust batch processing approach."""
        # Ensure we have an event loop in this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        logger.info(f"Starting LangChain summarization. Text length: {len(text)} characters")
        logger.info(f"Approximate token count: {count_tokens(text)} tokens")
        
        # Clean and preprocess text
        cleaned_text = re.sub(r'\s+', ' ', text).strip()
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(cleaned_text)
        logger.info(f"Split text into {len(chunks)} chunks")
        
        # Define improved custom prompts for Bengali
        map_prompt = PromptTemplate(
            template="""নিম্নলিখিত বাংলা টেক্সটের অংশটির একটি বিস্তারিত সারংক্ষেপ তৈরি করুন। সারংক্ষেপটি অন্তত ১৫০ শব্দের হওয়া উচিত এবং নিম্নলিখিত বিষয়গুলি অন্তর্ভুক্ত করতে হবে:
            ১) প্রধান বিষয়বস্তু এবং থিম
            ২) গুরুত্বপূর্ণ বিবরণ এবং উদাহরণ
            ৩) যেকোনো উল্লেখযোগ্য চরিত্র, স্থান বা ঘটনা
            ৪) লেখকের মূল যুক্তি বা বার্তা
            
            টেক্সট:
            {text}
            
            বিস্তারিত সারংক্ষেপ:""",
            input_variables=["text"]
        )
        
        combine_prompt = PromptTemplate(
            template="""নিম্নলিখিত বাংলা টেক্সটের সারংক্ষেপগুলির সমন্বয় করে একটি বিস্তারিত চূড়ান্ত সারংক্ষেপ তৈরি করুন। চূড়ান্ত সারংক্ষেপটি অন্তত ৪০০ শব্দের হওয়া উচিত এবং নিম্নলিখিত বিষয়গুলি অন্তর্ভুক্ত করতে হবে:
            ১) সমগ্র নথির প্রধান বিষয়বস্তু এবং থিম
            ২) সমস্ত গুরুত্বপূর্ণ বিবরণ এবং উদাহরণ
            ৩) নথিতে আলোচিত প্রধান বিষয়গুলির মধ্যে সম্পর্ক
            ৪) লেখকের মূল যুক্তি, বার্তা বা উপসংহার
            
            সারংক্ষেপগুলি:
            {text}
            
            বিস্তারিত চূড়ান্ত সারংক্ষেপ:""",
            input_variables=["text"]
        )
        
        # Process chunks in batches to avoid rate limits
        batch_size = self.config["batch_size"]
        all_summaries = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Processing batches"):
            batch_chunks = chunks[i:i+batch_size]
            batch_docs = [Document(page_content=chunk) for chunk in batch_chunks]
            
            logger.info(f"Processing batch {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1} with {len(batch_chunks)} chunks")
            
            # Create summarization chain for this batch
            chain = load_summarize_chain(
                llm=self.llm,
                chain_type="map_reduce",
                map_prompt=map_prompt,
                combine_prompt=combine_prompt,
                return_intermediate_steps=True
            )
            
            try:
                # Process this batch with rate limiting
                result = self.rate_limited_api_call(chain.invoke, {"input_documents": batch_docs})
                batch_summary = result['output_text']
                all_summaries.append(batch_summary)
                
                # Add a delay between batches
                time.sleep(self.config["batch_delay"])
                
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size + 1}: {str(e)}")
                # If a batch fails, try to process chunks individually
                for chunk in batch_chunks:
                    try:
                        chunk_doc = [Document(page_content=chunk)]
                        chunk_result = self.rate_limited_api_call(chain.invoke, {"input_documents": chunk_doc})
                        all_summaries.append(chunk_result['output_text'])
                        time.sleep(self.config["chunk_delay"])
                    except Exception as chunk_error:
                        logger.error(f"Error processing individual chunk: {str(chunk_error)}")
                        all_summaries.append(f"[Error processing chunk: {str(chunk_error)}]")
        
        logger.info(f"Generated {len(all_summaries)} batch summaries")
        
        # If we have too many summaries, combine them in stages
        if len(all_summaries) > 5:
            logger.info("Combining summaries in stages...")
            # First combine in groups of 5
            combined_summaries = []
            for i in range(0, len(all_summaries), 5):
                group = all_summaries[i:i+5]
                group_text = "\n\n".join(group)
                group_docs = [Document(page_content=group_text)]
                
                # Use the combine prompt to summarize the group
                group_chain = load_summarize_chain(
                    llm=self.llm,
                    chain_type="stuff",
                    prompt=combine_prompt
                )
                
                try:
                    group_result = self.rate_limited_api_call(group_chain.invoke, {"input_documents": group_docs})
                    combined_summaries.append(group_result['output_text'])
                    time.sleep(self.config["chunk_delay"])
                except Exception as e:
                    logger.error(f"Error combining group: {str(e)}")
                    combined_summaries.extend(group)
            
            all_summaries = combined_summaries
        
        # Final combination
        final_text = "\n\n".join(all_summaries)
        final_docs = [Document(page_content=final_text)]
        
        # Create final summarization chain
        final_chain = load_summarize_chain(
            llm=self.llm,
            chain_type="stuff",
            prompt=combine_prompt
        )
        
        try:
            # Generate final summary with rate limiting
            final_result = self.rate_limited_api_call(final_chain.invoke, {"input_documents": final_docs})
            return final_result['output_text']
        except Exception as e:
            logger.error(f"Error during final summarization: {str(e)}")
            return f"Error during final summarization: {str(e)}\n\nPartial summaries:\n" + "\n\n".join(all_summaries)
            
    def setup_qa_with_langchain(self, text: str):
        """Setup Q&A system using LangChain's retrieval-based approach with improved prompts."""
        # Ensure we have an event loop in this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        logger.info("Setting up LangChain Q&A system...")
        
        # Split text into chunks
        chunks = self.text_splitter.split_text(text)
        logger.info(f"Split text into {len(chunks)} chunks for Q&A")
        
        # Create documents
        docs = [Document(page_content=chunk) for chunk in chunks]
        
        # Create vector store with rate limiting
        self.vectorstore = self.rate_limited_api_call(FAISS.from_documents, docs, self.embeddings)
        
        # Define improved custom prompt for Bengali Q&A
        qa_prompt = PromptTemplate(
            template="""You are an expert assistant providing detailed answers about Bangla text. 
            Use ONLY the provided context to answer questions. If the context doesn't contain the answer, 
            say so clearly. Provide comprehensive explanations with examples when possible.
            
            Context: {context}
            
            Question: {question}
            
            Detailed Answer in Bengali:""",
            input_variables=["context", "question"]
        )
        
        # Use RetrievalQA with increased document retrieval
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 8}),
            chain_type_kwargs={"prompt": qa_prompt},
            return_source_documents=True,
            verbose=True
        )
        
        logger.info("Q&A system setup complete")
        
    def answer_question(self, question: str) -> Dict[str, Any]:
        """Answer a question using the Q&A system"""
        # Ensure we have an event loop in this thread
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
        if not self.qa_chain:
            raise ValueError("Q&A system not initialized. Call setup_qa_with_langchain first.")
            
        try:
            # Get answer from LangChain with rate limiting
            result = self.rate_limited_api_call(self.qa_chain.invoke, {"query": question})
            answer = result['result']
            source_docs = result.get('source_documents', [])
            
            return {
                "question": question,
                "answer": answer,
                "source_docs": source_docs
            }
        except Exception as e:
            logger.error(f"Error answering question: {str(e)}")
            return {
                "question": question,
                "answer": f"Error answering question: {str(e)}",
                "source_docs": []
            }
            
    def save_qa_session(self, qa_pairs: List[Dict[str, Any]], output_file: str):
        """Save Q&A session to a file"""
        with open(output_file, 'w', encoding='utf-8') as f:
            for qa in qa_pairs:
                f.write(f"প্রশ্ন: {qa['question']}\n")
                f.write(f"উত্তর: {qa['answer']}\n")
                if qa['source_docs']:
                    f.write("উত্তরের উৎস:\n")
                    for i, doc in enumerate(qa['source_docs'][:3]):
                        f.write(f"  উৎস {i+1}: {doc.page_content[:100]}...\n")
                f.write("\n---\n\n")
        logger.info(f"Q&A session saved to {output_file}")
        
    def save_session_data(self, output_dir: str):
        """Save session data for later use"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save vectorstore
        if self.vectorstore:
            vectorstore_path = os.path.join(output_dir, f"vectorstore_{timestamp}")
            self.vectorstore.save_local(vectorstore_path)
            logger.info(f"Vectorstore saved to {vectorstore_path}")
            
        # Save memory
        if self.memory:
            memory_path = os.path.join(output_dir, f"memory_{timestamp}.json")
            with open(memory_path, 'w', encoding='utf-8') as f:
                json.dump(self.memory.load_memory_variables({}), f, ensure_ascii=False, indent=2)
            logger.info(f"Memory saved to {memory_path}")
            
        return timestamp
        
    def load_session_data(self, session_dir: str, timestamp: str):
        """Load session data from previous run"""
        vectorstore_path = os.path.join(session_dir, f"vectorstore_{timestamp}")
        memory_path = os.path.join(session_dir, f"memory_{timestamp}.json")
        
        # Load vectorstore
        if os.path.exists(vectorstore_path):
            self.vectorstore = FAISS.load_local(vectorstore_path, self.embeddings)
            logger.info(f"Vectorstore loaded from {vectorstore_path}")
            
        # Load memory
        if os.path.exists(memory_path):
            with open(memory_path, 'r', encoding='utf-8') as f:
                memory_data = json.load(f)
                self.memory.chat_memory.messages = memory_data.get("chat_history", [])
            logger.info(f"Memory loaded from {memory_path}")

def validate_api_key(api_key: str) -> bool:
    """Check if API key is valid"""
    if not api_key or api_key == "YOUR_NEW_API_KEY_HERE":
        logger.error("Invalid API key. Please update in the configuration.")
        return False
    return True

def generate_new_api_key():
    """Instructions for generating a new API key"""
    print("\n" + "="*60)
    print("YOUR GEMINI API KEY HAS EXPIRED!")
    print("Please generate a new API key by following these steps:")
    print("1. Go to https://makersuite.google.com/app/apikey")
    print("2. Sign in with your Google account")
    print("3. Click 'Create API Key'")
    print("4. Copy the new API key")
    print("5. Replace the old API key in the script with the new one")
    print("="*60)

def interactive_qa(processor: BanglaPDFProcessor, output_file: str):
    """Interactive Q&A session"""
    print("\n" + "="*50)
    print("Starting Question Answering Mode with LangChain.")
    print("Type your questions (in Bangla) and press Enter. Type 'exit' to quit.")
    print("="*50)
    
    qa_output_content = []
    
    while True:
        user_question = input("\nআপনার প্রশ্ন (বাংলায়): ")
        if user_question.lower() == 'exit':
            break
            
        try:
            result = processor.answer_question(user_question)
            answer = result['answer']
            source_docs = result['source_docs']
            
            print(f"\nউত্তর: {answer}")
            
            if source_docs:
                print("\nউত্তরের উৎস:")
                for i, doc in enumerate(source_docs[:3]):
                    print(f"  উৎস {i+1}: {doc.page_content[:100]}...")
            
            qa_output_content.append(result)
            
        except Exception as e:
            if "API key expired" in str(e):
                generate_new_api_key()
                exit(1)
            print(f"Error answering question: {str(e)}")
    
    if qa_output_content:
        processor.save_qa_session(qa_output_content, output_file)
        print(f"\nAll questions and answers saved to '{output_file}'.")
        print(f"Find your Q&A output in the file located at: {os.path.abspath(output_file)}")
    else:
        print("\nNo questions were asked or answered.")

def main():
    # Ensure we have an event loop in this thread
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
    parser = argparse.ArgumentParser(description="Bangla PDF Processor with OCR and Q&A")
    parser.add_argument("--pdf", type=str, required=True, help="Path to the Bangla PDF file")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory to save output files")
    parser.add_argument("--api-key", type=str, required=True, help="Google Gemini API key")
    parser.add_argument("--config", type=str, help="Path to configuration JSON file")
    parser.add_argument("--session", type=str, help="Load previous session (timestamp)")
    parser.add_argument("--mode", choices=["extract", "summarize", "qa", "all"], default="all", 
                       help="Processing mode: extract, summarize, qa, or all")
    
    args = parser.parse_args()
    
    # Load configuration
    config = DEFAULT_CONFIG.copy()
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Initialize processor
    processor = BanglaPDFProcessor(config)
    
    # Validate API key
    if not validate_api_key(args.api_key):
        generate_new_api_key()
        exit(1)
    
    # Initialize LangChain components
    try:
        logger.info("Initializing LangChain components...")
        processor.create_langchain_components(args.api_key)
    except Exception as e:
        logger.error(f"Error initializing LangChain components: {str(e)}")
        exit(1)
    
    # Extract text from PDF
    if args.mode in ["extract", "all"]:
        output_file = os.path.join(args.output_dir, "extracted_bangla_text.txt")
        status_message, file_path, time_taken, full_text = processor.extract_bangla_text_from_pdf(args.pdf, output_file)
        
        print("\n" + "="*50)
        print(status_message)
        print("="*50)
        
        if not file_path:
            logger.error("Could not proceed due to an error in text extraction.")
            exit(1)
    
    # Summarize text
    if args.mode in ["summarize", "all"] and 'full_text' in locals():
        summary_file = os.path.join(args.output_dir, "summarized_bangla_text.txt")
        
        try:
            logger.info("Starting summarization process...")
            summarized_text = processor.summarize_with_langchain(full_text)
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summarized_text)
                
            print(f"\nSummarization complete. Summary saved to '{summary_file}'.")
            print(f"Find your summarized text in the file located at: {os.path.abspath(summary_file)}")
        except Exception as e:
            logger.error(f"Error during summarization: {str(e)}")
    
    # Setup Q&A system
    if args.mode in ["qa", "all"] and 'full_text' in locals():
        qa_file = os.path.join(args.output_dir, "qa_output.txt")
        
        try:
            processor.setup_qa_with_langchain(full_text)
            
            # Save session data for later use
            timestamp = processor.save_session_data(args.output_dir)
            logger.info(f"Session data saved with timestamp: {timestamp}")
            
            # Start interactive Q&A
            interactive_qa(processor, qa_file)
        except Exception as e:
            logger.error(f"Error setting up Q&A system: {str(e)}")

if __name__ == "__main__":
    main()