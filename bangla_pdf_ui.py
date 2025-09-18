import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
import json
from datetime import datetime
import asyncio  # Add this import
import nest_asyncio  # Add this import

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

class BanglaPDFProcessorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Bangla PDF Processor")
        self.root.geometry("900x700")
        self.root.configure(bg="#f0f0f0")
        
        # Set up styles
        self.setup_styles()
        
        # Create main frames
        self.create_header()
        self.create_input_section()
        self.create_processing_section()
        self.create_output_section()
        self.create_status_bar()
        
        # Initialize processor
        self.processor = None
        self.full_text = ""
        self.qa_pairs = []
        
    def setup_styles(self):
        """Configure UI styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure button styles
        style.configure('TButton', font=('Arial', 10), padding=5)
        style.map('TButton', 
                 foreground=[('pressed', 'white'), ('active', 'white')],
                 background=[('pressed', '#4a6fa5'), ('active', '#5a7fb5')])
        
        # Configure frame styles
        style.configure('Title.TFrame', background='#2c3e50')
        style.configure('Input.TFrame', background='#ecf0f1', relief='ridge', borderwidth=2)
        style.configure('Output.TFrame', background='#ecf0f1', relief='ridge', borderwidth=2)
        
        # Configure label styles
        style.configure('Title.TLabel', background='#2c3e50', foreground='white', font=('Arial', 16, 'bold'))
        style.configure('Header.TLabel', background='#34495e', foreground='white', font=('Arial', 12, 'bold'))
        style.configure('Input.TLabel', background='#ecf0f1', font=('Arial', 10, 'bold'))
        style.configure('Output.TLabel', background='#ecf0f1', font=('Arial', 10, 'bold'))
        
    def create_header(self):
        """Create application header"""
        header_frame = ttk.Frame(self.root, style='Title.TFrame')
        header_frame.pack(fill=tk.X)
        
        title_label = ttk.Label(header_frame, text="বাংলা PDF প্রসেসর", style='Title.TLabel')
        title_label.pack(pady=10)
        
        subtitle_label = ttk.Label(header_frame, text="Extract, Summarize, and Q&A for Bangla PDFs", 
                                  style='Title.TLabel', font=('Arial', 10))
        subtitle_label.pack(pady=5)
        
    def create_input_section(self):
        """Create input section with file selection and settings"""
        input_frame = ttk.LabelFrame(self.root, text="Input Settings", style='Input.TFrame')
        input_frame.pack(fill=tk.X, padx=10, pady=10, ipadx=5, ipady=5)
        
        # PDF Selection
        pdf_frame = ttk.Frame(input_frame, style='Input.TFrame')
        pdf_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(pdf_frame, text="Select PDF File:", style='Input.TLabel').pack(side=tk.LEFT, padx=5)
        
        self.pdf_path_var = tk.StringVar()
        self.pdf_entry = ttk.Entry(pdf_frame, textvariable=self.pdf_path_var, width=50)
        self.pdf_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_pdf_btn = ttk.Button(pdf_frame, text="Browse", command=self.browse_pdf)
        self.browse_pdf_btn.pack(side=tk.LEFT, padx=5)
        
        # API Key
        api_frame = ttk.Frame(input_frame, style='Input.TFrame')
        api_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(api_frame, text="Google Gemini API Key:", style='Input.TLabel').pack(side=tk.LEFT, padx=5)
        
        self.api_key_var = tk.StringVar()
        self.api_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50, show="*")
        self.api_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.show_api_btn = ttk.Button(api_frame, text="Show", command=self.toggle_api_visibility)
        self.show_api_btn.pack(side=tk.LEFT, padx=5)
        
        # Processing Mode
        mode_frame = ttk.Frame(input_frame, style='Input.TFrame')
        mode_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(mode_frame, text="Processing Mode:", style='Input.TLabel').pack(side=tk.LEFT, padx=5)
        
        self.mode_var = tk.StringVar(value="all")
        modes = [("Extract Text", "extract"), 
                ("Summarize", "summarize"), 
                ("Q&A", "qa"), 
                ("All Tasks", "all")]
        
        for text, mode in modes:
            ttk.Radiobutton(mode_frame, text=text, variable=self.mode_var, 
                            value=mode, style='Input.TRadiobutton').pack(side=tk.LEFT, padx=5)
        
        # Output Directory
        output_frame = ttk.Frame(input_frame, style='Input.TFrame')
        output_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(output_frame, text="Output Directory:", style='Input.TLabel').pack(side=tk.LEFT, padx=5)
        
        self.output_dir_var = tk.StringVar(value="output")
        self.output_entry = ttk.Entry(output_frame, textvariable=self.output_dir_var, width=50)
        self.output_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.browse_output_btn = ttk.Button(output_frame, text="Browse", command=self.browse_output_dir)
        self.browse_output_btn.pack(side=tk.LEFT, padx=5)
        
        # Process Button
        self.process_btn = ttk.Button(input_frame, text="Process PDF", command=self.start_processing)
        self.process_btn.pack(pady=10)
        
    def create_processing_section(self):
        """Create processing section with progress and logs"""
        processing_frame = ttk.LabelFrame(self.root, text="Processing Status", style='Input.TFrame')
        processing_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(processing_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, padx=5, pady=5)
        
        # Status label
        self.status_var = tk.StringVar(value="Ready to process")
        self.status_label = ttk.Label(processing_frame, textvariable=self.status_var, style='Input.TLabel')
        self.status_label.pack(padx=5, pady=5)
        
        # Log area
        log_frame = ttk.Frame(processing_frame, style='Input.TFrame')
        log_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(log_frame, text="Processing Log:", style='Input.TLabel').pack(anchor=tk.W)
        
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=80, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Configure log text tags
        self.log_text.tag_config("info", foreground="black")
        self.log_text.tag_config("success", foreground="green")
        self.log_text.tag_config("error", foreground="red")
        self.log_text.tag_config("warning", foreground="orange")
        
    def create_output_section(self):
        """Create output section with tabs for results"""
        output_frame = ttk.LabelFrame(self.root, text="Results", style='Output.TFrame')
        output_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create notebook for tabs
        self.output_notebook = ttk.Notebook(output_frame)
        self.output_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Extracted Text Tab
        self.extracted_tab = ttk.Frame(self.output_notebook, style='Output.TFrame')
        self.output_notebook.add(self.extracted_tab, text="Extracted Text")
        
        self.extracted_text = scrolledtext.ScrolledText(self.extracted_tab, wrap=tk.WORD, width=80, height=15)
        self.extracted_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Summary Tab
        self.summary_tab = ttk.Frame(self.output_notebook, style='Output.TFrame')
        self.output_notebook.add(self.summary_tab, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(self.summary_tab, wrap=tk.WORD, width=80, height=15)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Q&A Tab
        self.qa_tab = ttk.Frame(self.output_notebook, style='Output.TFrame')
        self.output_notebook.add(self.qa_tab, text="Q&A")
        
        qa_input_frame = ttk.Frame(self.qa_tab, style='Output.TFrame')
        qa_input_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Label(qa_input_frame, text="Ask a question (in Bangla):", style='Output.TLabel').pack(anchor=tk.W)
        
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(qa_input_frame, textvariable=self.question_var, width=70)
        self.question_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.ask_btn = ttk.Button(qa_input_frame, text="Ask", command=self.ask_question)
        self.ask_btn.pack(side=tk.LEFT, padx=5)
        
        self.qa_text = scrolledtext.ScrolledText(self.qa_tab, wrap=tk.WORD, width=80, height=15)
        self.qa_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Save buttons
        save_frame = ttk.Frame(output_frame, style='Output.TFrame')
        save_frame.pack(fill=tk.X, padx=5, pady=5)
        
        self.save_extracted_btn = ttk.Button(save_frame, text="Save Extracted Text", command=self.save_extracted_text)
        self.save_extracted_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_summary_btn = ttk.Button(save_frame, text="Save Summary", command=self.save_summary)
        self.save_summary_btn.pack(side=tk.LEFT, padx=5)
        
        self.save_qa_btn = ttk.Button(save_frame, text="Save Q&A Session", command=self.save_qa_session)
        self.save_qa_btn.pack(side=tk.LEFT, padx=5)
        
    def create_status_bar(self):
        """Create status bar at the bottom"""
        status_frame = ttk.Frame(self.root, style='Input.TFrame')
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)
        
        self.status_bar_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_bar_var, style='Input.TLabel').pack(side=tk.LEFT, padx=5)
        
        # Clock
        self.clock_var = tk.StringVar()
        self.update_clock()
        ttk.Label(status_frame, textvariable=self.clock_var, style='Input.TLabel').pack(side=tk.RIGHT, padx=5)
        
    def update_clock(self):
        """Update the clock in status bar"""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.clock_var.set(now)
        self.root.after(1000, self.update_clock)
        
    def browse_pdf(self):
        """Open file dialog to select PDF"""
        file_path = filedialog.askopenfilename(
            title="Select PDF File",
            filetypes=[("PDF Files", "*.pdf"), ("All Files", "*.*")]
        )
        if file_path:
            self.pdf_path_var.set(file_path)
            
    def browse_output_dir(self):
        """Open directory dialog to select output directory"""
        dir_path = filedialog.askdirectory(title="Select Output Directory")
        if dir_path:
            self.output_dir_var.set(dir_path)
            
    def toggle_api_visibility(self):
        """Toggle API key visibility"""
        if self.api_entry.cget('show') == "":
            self.api_entry.config(show="*")
            self.show_api_btn.config(text="Show")
        else:
            self.api_entry.config(show="")
            self.show_api_btn.config(text="Hide")
            
    def log_message(self, message, level="info"):
        """Add message to log with appropriate styling"""
        self.log_text.config(state=tk.NORMAL)
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n", level)
        self.log_text.see(tk.END)
        self.log_text.config(state=tk.DISABLED)
        
    def update_progress(self, value):
        """Update progress bar"""
        self.progress_var.set(value)
        self.root.update_idletasks()
        
    def update_status(self, message):
        """Update status message"""
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def start_processing(self):
        """Start the PDF processing in a separate thread"""
        # Validate inputs
        if not self.pdf_path_var.get():
            messagebox.showerror("Error", "Please select a PDF file")
            return
            
        if not self.api_key_var.get():
            messagebox.showerror("Error", "Please enter your Google Gemini API key")
            return
            
        if not os.path.exists(self.output_dir_var.get()):
            os.makedirs(self.output_dir_var.get())
            
        # Disable UI elements during processing
        self.process_btn.config(state=tk.DISABLED)
        self.browse_pdf_btn.config(state=tk.DISABLED)
        self.browse_output_btn.config(state=tk.DISABLED)
        
        # Start processing in a separate thread
        threading.Thread(target=self.process_pdf, daemon=True).start()
        
    def process_pdf(self):
        """Process the PDF file"""
        loop = None
        try:
            # Create a new event loop for this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            # Import the processor here to avoid issues with threading
            from bangla_pdf_processor import BanglaPDFProcessor, DEFAULT_CONFIG
            
            # Initialize processor
            config = DEFAULT_CONFIG.copy()
            self.processor = BanglaPDFProcessor(config)
            
            # Create output directory if it doesn't exist
            output_dir = self.output_dir_var.get()
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            # Initialize LangChain components
            self.update_status("Initializing components...")
            self.update_progress(5)
            self.log_message("Initializing LangChain components...")
            
            self.processor.create_langchain_components(self.api_key_var.get())
            
            # Extract text from PDF
            self.update_status("Extracting text from PDF...")
            self.update_progress(20)
            self.log_message(f"Processing PDF: {self.pdf_path_var.get()}")
            
            output_file = os.path.join(output_dir, "extracted_bangla_text.txt")
            status_message, file_path, time_taken, full_text = self.processor.extract_bangla_text_from_pdf(
                self.pdf_path_var.get(), output_file
            )
            
            if not file_path:
                self.log_message(f"Error extracting text: {status_message}", "error")
                self.update_status("Error extracting text")
                self.reset_ui()
                return
                
            self.full_text = full_text
            self.log_message(f"Text extraction completed in {time_taken:.2f} seconds", "success")
            
            # Display extracted text
            self.root.after(0, lambda: self.display_extracted_text(full_text))
            
            # Summarize text if needed
            mode = self.mode_var.get()
            if mode in ["summarize", "all"]:
                self.update_status("Summarizing text...")
                self.update_progress(50)
                self.log_message("Starting summarization process...")
                
                summary = self.processor.summarize_with_langchain(full_text)
                
                # Save summary to file
                summary_file = os.path.join(output_dir, "summarized_bangla_text.txt")
                with open(summary_file, 'w', encoding='utf-8') as f:
                    f.write(summary)
                    
                self.log_message(f"Summary saved to {summary_file}", "success")
                
                # Display summary
                self.root.after(0, lambda: self.display_summary(summary))
                
            # Setup Q&A if needed
            if mode in ["qa", "all"]:
                self.update_status("Setting up Q&A system...")
                self.update_progress(80)
                self.log_message("Setting up Q&A system...")
                
                self.processor.setup_qa_with_langchain(full_text)
                
                # Save session data
                timestamp = self.processor.save_session_data(output_dir)
                self.log_message(f"Session data saved with timestamp: {timestamp}", "success")
                
                # Enable Q&A tab
                self.root.after(0, lambda: self.output_notebook.tab(2, state="normal"))
                
            self.update_progress(100)
            self.update_status("Processing completed successfully!")
            self.log_message("Processing completed successfully!", "success")
            
        except Exception as e:
            self.log_message(f"Error during processing: {str(e)}", "error")
            self.update_status("Error during processing")
            messagebox.showerror("Processing Error", f"An error occurred during processing: {str(e)}")
            
        finally:
            # Close the event loop if we created one
            if loop is not None:
                loop.close()
                
            # Re-enable UI elements
            self.root.after(0, self.reset_ui)
            
    def reset_ui(self):
        """Reset UI elements after processing"""
        self.process_btn.config(state=tk.NORMAL)
        self.browse_pdf_btn.config(state=tk.NORMAL)
        self.browse_output_btn.config(state=tk.NORMAL)
        
    def display_extracted_text(self, text):
        """Display extracted text in the UI"""
        self.extracted_text.config(state=tk.NORMAL)
        self.extracted_text.delete(1.0, tk.END)
        self.extracted_text.insert(tk.END, text)
        self.extracted_text.config(state=tk.DISABLED)
        
    def display_summary(self, summary):
        """Display summary in the UI"""
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(tk.END, summary)
        self.summary_text.config(state=tk.DISABLED)
        
    def ask_question(self):
        """Ask a question using the Q&A system"""
        if not self.processor or not hasattr(self.processor, 'qa_chain'):
            messagebox.showerror("Error", "Q&A system not initialized. Please process a PDF first.")
            return
            
        question = self.question_var.get().strip()
        if not question:
            messagebox.showerror("Error", "Please enter a question")
            return
            
        # Disable UI during processing
        self.ask_btn.config(state=tk.DISABLED)
        self.question_entry.config(state=tk.DISABLED)
        
        # Process question in a separate thread
        threading.Thread(target=self.process_question, args=(question,), daemon=True).start()
        
    def process_question(self, question):
        """Process a question in a separate thread"""
        loop = None
        try:
            # Create a new event loop for this thread
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            self.update_status("Processing question...")
            self.log_message(f"Processing question: {question}")
            
            result = self.processor.answer_question(question)
            answer = result['answer']
            source_docs = result['source_docs']
            
            # Store Q&A pair
            self.qa_pairs.append(result)
            
            # Display answer
            self.root.after(0, lambda: self.display_answer(question, answer, source_docs))
            
            self.update_status("Question answered successfully")
            self.log_message("Question answered successfully", "success")
            
        except Exception as e:
            self.log_message(f"Error answering question: {str(e)}", "error")
            self.update_status("Error answering question")
            messagebox.showerror("Q&A Error", f"An error occurred while answering the question: {str(e)}")
            
        finally:
            # Close the event loop if we created one
            if loop is not None:
                loop.close()
                
            # Re-enable UI elements
            self.root.after(0, lambda: self.ask_btn.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.question_entry.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.question_var.set(""))
            
    def display_answer(self, question, answer, source_docs):
        """Display Q&A result in the UI"""
        self.qa_text.config(state=tk.NORMAL)
        
        # Add question
        self.qa_text.insert(tk.END, f"প্রশ্ন: {question}\n\n", "question")
        
        # Add answer
        self.qa_text.insert(tk.END, f"উত্তর: {answer}\n\n", "answer")
        
        # Add sources if available
        if source_docs:
            self.qa_text.insert(tk.END, "উত্তরের উৎস:\n", "source")
            for i, doc in enumerate(source_docs[:3]):
                self.qa_text.insert(tk.END, f"  উৎস {i+1}: {doc.page_content[:100]}...\n\n")
                
        self.qa_text.insert(tk.END, "-" * 50 + "\n\n")
        self.qa_text.see(tk.END)
        self.qa_text.config(state=tk.DISABLED)
        
    def save_extracted_text(self):
        """Save extracted text to a file"""
        if not self.full_text:
            messagebox.showerror("Error", "No extracted text to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Extracted Text",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.full_text)
                messagebox.showinfo("Success", f"Extracted text saved to {file_path}")
                self.log_message(f"Extracted text saved to {file_path}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                self.log_message(f"Error saving extracted text: {str(e)}", "error")
                
    def save_summary(self):
        """Save summary to a file"""
        summary = self.summary_text.get(1.0, tk.END).strip()
        if not summary:
            messagebox.showerror("Error", "No summary to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Summary",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(summary)
                messagebox.showinfo("Success", f"Summary saved to {file_path}")
                self.log_message(f"Summary saved to {file_path}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                self.log_message(f"Error saving summary: {str(e)}", "error")
                
    def save_qa_session(self):
        """Save Q&A session to a file"""
        if not self.qa_pairs:
            messagebox.showerror("Error", "No Q&A session to save")
            return
            
        file_path = filedialog.asksaveasfilename(
            title="Save Q&A Session",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")]
        )
        
        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    for qa in self.qa_pairs:
                        f.write(f"প্রশ্ন: {qa['question']}\n")
                        f.write(f"উত্তর: {qa['answer']}\n")
                        if qa['source_docs']:
                            f.write("উত্তরের উৎস:\n")
                            for i, doc in enumerate(qa['source_docs'][:3]):
                                f.write(f"  উৎস {i+1}: {doc.page_content[:100]}...\n")
                        f.write("\n---\n\n")
                messagebox.showinfo("Success", f"Q&A session saved to {file_path}")
                self.log_message(f"Q&A session saved to {file_path}", "success")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save file: {str(e)}")
                self.log_message(f"Error saving Q&A session: {str(e)}", "error")

if __name__ == "__main__":
    root = tk.Tk()
    app = BanglaPDFProcessorUI(root)
    root.mainloop()