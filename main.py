"""Citation Generator Application

This module implements a graphical user interface for an AI-powered citation generation system.
It combines Google's Gemini AI for intelligent query generation with arXiv paper search
to help users find relevant academic citations for their text.

The application provides a user-friendly interface built with CustomTkinter, allowing users to:
1. Input text that needs citations
2. Configure search parameters (number of queries and citations)
3. Generate relevant citations using AI
4. View and copy both the cited text and BibTeX entries

Dependencies:
    - customtkinter: For modern GUI elements
    - pyperclip: For clipboard operations
    - utils: Custom module for AI and search functionality

Example:
    To run the application:
    >>> app = CitationGeneratorUI()
    >>> app.run()
"""

import customtkinter as ctk
import pyperclip
from utils import init_clients, generate_queries, search_arxiv_batch
import json
from loguru import logger
import asyncio
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

class CitationGeneratorUI:
    """A graphical user interface for the citation generation system.

    This class creates a window with text input/output areas and controls for generating
    citations using Google Gemini AI and arXiv paper search. The interface provides:
    - An input area for the text that needs citations
    - Controls for setting the number of queries and citations
    - Output areas for the cited text and BibTeX entries
    - Copy buttons for easy transfer of results

    Attributes:
        root (ctk.CTk): The main window of the application
        client (genai.GenerativeModel): The Gemini AI client for query generation
        input_text (ctk.CTkTextbox): Text input area for content needing citations
        output_text (ctk.CTkTextbox): Text output area for cited content
        bibtex_text (ctk.CTkTextbox): Text output area for BibTeX entries
        num_queries_var (ctk.StringVar): Variable storing number of queries setting
        citations_per_query_var (ctk.StringVar): Variable storing citations per query setting
        progress_label (ctk.CTkLabel): Label showing progress during citation generation
    """

    def __init__(self):
        """Initialize the citation generator interface.

        Sets up the main window, initializes the Gemini client, and creates all UI elements.
        The initialization process includes:
        1. Setting up the CustomTkinter appearance
        2. Creating the main window
        3. Initializing the Gemini AI client
        4. Creating and arranging all UI elements
        
        Raises:
            Exception: If client initialization fails
        """
        # Set the appearance mode and default color theme
        ctk.set_appearance_mode("System")  # Options: "System", "Dark", "Light"
        ctk.set_default_color_theme("blue")  # Options: "blue", "green", "dark-blue"

        # Create the main window
        self.root = ctk.CTk()
        self.root.title("AutoCitation")
        self.root.geometry("1200x800")  # Set a reasonable default size

        # Initialize undo history stacks
        self.input_history = []
        self.cited_history = []
        self.bibtex_history = []
        self.input_redo = []
        self.cited_redo = []
        self.bibtex_redo = []
        self.max_history = 50  # Maximum number of undo steps

        # Initialize thread pool for background tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.task_queue = queue.Queue()
        
        # Start background worker
        self.worker_thread = threading.Thread(target=self._process_background_tasks, daemon=True)
        self.worker_thread.start()

        success, self.client = init_clients()
        if not success:
            logger.error("Failed to initialize client")
            raise Exception("Client initialization failed")

        # Set up the user interface elements
        self.setup_ui()
        
    def _process_background_tasks(self):
        """Background worker to process tasks"""
        while True:
            try:
                task = self.task_queue.get()
                if task is None:
                    break
                task()
                self.task_queue.task_done()
            except Exception as e:
                logger.error(f"Error in background task: {str(e)}")

    def cleanup(self):
        """Clean up resources before closing"""
        try:
            # Stop background tasks first
            if hasattr(self, 'task_queue'):
                self.task_queue.put(None)
            if hasattr(self, 'worker_thread'):
                self.worker_thread.join(timeout=1.0)
            if hasattr(self, 'executor'):
                self.executor.shutdown(wait=False)
            
            # Clear text widgets if they exist
            for widget_name in ['input_text', 'cited_text', 'bibtex_text']:
                if hasattr(self, widget_name):
                    widget = getattr(self, widget_name)
                    if widget:
                        try:
                            widget.delete("1.0", ctk.END)
                        except Exception:
                            pass  # Ignore widget errors during cleanup
        except Exception as e:
            logger.error(f"Error during cleanup: {str(e)}")
        finally:
            # Always try to destroy the window last
            try:
                self.root.quit()
                self.root.destroy()
            except Exception:
                pass  # Ignore destroy errors

    def validate_int_input(self, value, default=2):
        """Validate integer input and return default if invalid"""
        try:
            val = int(value)
            return max(1, val)  # Ensure value is at least 1
        except (ValueError, TypeError):
            return default

    def setup_ui(self):
        """Create and arrange all UI elements in the window."""

        # Create the main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill=ctk.BOTH, expand=True, padx=20, pady=20)

        # Create the input frame for text entry
        input_frame = ctk.CTkFrame(main_frame)
        input_frame.pack(fill=ctk.BOTH, expand=True)

        # Add label and scrolled text widget for input
        input_label = ctk.CTkLabel(input_frame, text="Input Text:", anchor="w", font=ctk.CTkFont(size=16, weight="bold"))
        input_label.pack(anchor=ctk.W, pady=(0, 5))

        self.input_text = ctk.CTkTextbox(input_frame, height=200)
        self.input_text.pack(fill=ctk.BOTH, expand=True)
        self.input_text.bind('<KeyRelease>', lambda e: self.on_text_change(self.input_text, self.input_history, self.input_redo))
        self.input_text.bind('<Control-z>', lambda e: self.undo_text(self.input_text, self.input_history, self.input_redo))
        self.input_text.bind('<Control-Shift-Z>', lambda e: self.redo_text(self.input_text, self.input_history, self.input_redo))

        # Create the settings frame for controls
        settings_frame = ctk.CTkFrame(main_frame)
        settings_frame.pack(fill=ctk.X, pady=10)

        # Add controls for citations per query
        citations_label = ctk.CTkLabel(settings_frame, text="Citations per query:")
        citations_label.pack(side=ctk.LEFT, padx=5)

        self.citations_var = ctk.StringVar(value="2")
        citations_entry = ctk.CTkEntry(settings_frame, textvariable=self.citations_var, width=50)
        citations_entry.pack(side=ctk.LEFT, padx=5)

        # Add controls for number of queries
        queries_label = ctk.CTkLabel(settings_frame, text="Number of queries:")
        queries_label.pack(side=ctk.LEFT, padx=10)

        self.queries_var = ctk.StringVar(value="2")
        queries_entry = ctk.CTkEntry(settings_frame, textvariable=self.queries_var, width=50)
        queries_entry.pack(side=ctk.LEFT, padx=5)

        # Add the process button
        self.process_btn = ctk.CTkButton(settings_frame, text="Process", command=self.process_text)
        self.process_btn.pack(side=ctk.LEFT, padx=20)

        # Add progress frame
        progress_frame = ctk.CTkFrame(main_frame)
        progress_frame.pack(fill=ctk.X, pady=(0, 10))

        # Add progress bar and status label
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(side=ctk.LEFT, fill=ctk.X, expand=True, padx=(0, 10))
        self.progress_bar.set(0)  # Initialize to 0

        self.status_label = ctk.CTkLabel(progress_frame, text="Ready")
        self.status_label.pack(side=ctk.LEFT, padx=5)

        # Create the output frame for results
        output_frame = ctk.CTkFrame(main_frame)
        output_frame.pack(fill=ctk.BOTH, expand=True)

        # Create the frame for cited text output
        cited_frame = ctk.CTkFrame(output_frame)
        cited_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=5)

        cited_label = ctk.CTkLabel(cited_frame, text="Cited Text", font=ctk.CTkFont(size=16, weight="bold"))
        cited_label.pack(anchor=ctk.W, pady=(0, 5))

        # Add text area and copy button for cited text
        self.cited_text = ctk.CTkTextbox(cited_frame)
        self.cited_text.pack(fill=ctk.BOTH, expand=True)
        self.cited_text.bind('<KeyRelease>', lambda e: self.on_text_change(self.cited_text, self.cited_history, self.cited_redo))
        self.cited_text.bind('<Control-z>', lambda e: self.undo_text(self.cited_text, self.cited_history, self.cited_redo))
        self.cited_text.bind('<Control-Shift-Z>', lambda e: self.redo_text(self.cited_text, self.cited_history, self.cited_redo))

        copy_cited_btn = ctk.CTkButton(cited_frame, text="Copy", command=lambda: self.copy_text(self.cited_text))
        copy_cited_btn.pack(pady=5)

        # Create the frame for BibTeX output
        bibtex_frame = ctk.CTkFrame(output_frame)
        bibtex_frame.pack(side=ctk.LEFT, fill=ctk.BOTH, expand=True, padx=5)

        bibtex_label = ctk.CTkLabel(bibtex_frame, text="BibTeX Entries", font=ctk.CTkFont(size=16, weight="bold"))
        bibtex_label.pack(anchor=ctk.W, pady=(0, 5))

        # Add text area and copy button for BibTeX entries
        self.bibtex_text = ctk.CTkTextbox(bibtex_frame)
        self.bibtex_text.pack(fill=ctk.BOTH, expand=True)
        self.bibtex_text.bind('<KeyRelease>', lambda e: self.on_text_change(self.bibtex_text, self.bibtex_history, self.bibtex_redo))
        self.bibtex_text.bind('<Control-z>', lambda e: self.undo_text(self.bibtex_text, self.bibtex_history, self.bibtex_redo))
        self.bibtex_text.bind('<Control-Shift-Z>', lambda e: self.redo_text(self.bibtex_text, self.bibtex_history, self.bibtex_redo))

        copy_bibtex_btn = ctk.CTkButton(bibtex_frame, text="Copy", command=lambda: self.copy_text(self.bibtex_text))
        copy_bibtex_btn.pack(pady=5)

    def copy_text(self, text_widget):
        """Copy the contents of a text widget to the clipboard."""
        text = text_widget.get("1.0", ctk.END)
        pyperclip.copy(text)

    async def _process_text_async(self):
        """Asynchronous processing of text input"""
        try:
            # Disable process button and reset progress
            self.root.after(0, self._update_progress, 0, "Starting...")
            self.root.after(0, self.process_btn.configure, {"state": "disabled"})

            input_text = self.input_text.get("1.0", ctk.END).strip()
            citations_per_query = self.validate_int_input(self.citations_var.get(), 2)
            num_queries = self.validate_int_input(self.queries_var.get(), 2)

            if not input_text:
                self.root.after(0, self._update_progress, 0, "Please enter text to process")
                return

            # Generate queries in batches
            self.root.after(0, self._update_progress, 0.2, "Generating queries...")
            queries = []
            batch_size = 5
            for i in range(0, num_queries, batch_size):
                batch_queries = generate_queries(
                    self.client, 
                    input_text, 
                    min(batch_size, num_queries - i)
                )
                queries.extend(batch_queries)
                progress = 0.2 + (0.3 * (i + batch_size) / num_queries)
                self.root.after(0, self._update_progress, progress, f"Generated {len(queries)} queries...")
                await asyncio.sleep(0.1)

            if not queries:
                self.root.after(0, self._update_progress, 0, "Failed to generate queries")
                return

            # Search papers concurrently
            self.root.after(0, self._update_progress, 0.5, "Searching papers...")
            papers = await search_arxiv_batch(queries, citations_per_query)
            
            if not papers:
                self.root.after(0, self._update_progress, 0, "No papers found")
                return

            # Generate citations
            self.root.after(0, self._update_progress, 0.8, "Generating citations...")
            system_prompt = """You are a LaTeX citation assistant. Insert \\cite{} commands 
            into the text where appropriate. All provided papers must be cited at least once. 
            Only insert \\cite{} commands without changing anything else."""

            user_prompt = f"Insert \\cite{{}} commands into this text:\n\n" \
                          f"Papers:\n{json.dumps(papers, indent=2)}\n\n" \
                          f"Text:\n{input_text}"

            response = self.client.generate_content(f"{system_prompt}\n\n{user_prompt}")
            
            if response and hasattr(response, 'text'):
                self.root.after(0, self._update_progress, 1.0, "Complete!")
                self.root.after(0, self._update_ui, response.text, papers)
            else:
                self.root.after(0, self._update_progress, 0, "Failed to generate citations")

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            self.root.after(0, self._update_progress, 0, f"Error: {str(e)}")
        finally:
            # Re-enable process button
            self.root.after(0, self.process_btn.configure, {"state": "normal"})

    def _update_progress(self, progress, status_text):
        """Update progress bar and status label"""
        self.progress_bar.set(progress)
        self.status_label.configure(text=status_text)

    def _update_ui(self, cited_text, papers):
        """Update UI with results in main thread"""
        self.cited_text.delete("1.0", ctk.END)
        self.cited_text.insert("1.0", cited_text)

        self.bibtex_text.delete("1.0", ctk.END)
        bibtex_entries = [p['bibtex_entry'] for p in papers if 'bibtex_entry' in p]
        self.bibtex_text.insert("1.0", "\n\n".join(bibtex_entries))

    def process_text(self):
        """Process the input text in background"""
        self.task_queue.put(
            lambda: asyncio.run(self._process_text_async())
        )

    def on_text_change(self, text_widget, history_stack, redo_stack):
        """Handle text changes and update the undo history."""
        current_text = text_widget.get("1.0", ctk.END).rstrip()
        if not history_stack or current_text != history_stack[-1]:
            history_stack.append(current_text)
            if len(history_stack) > self.max_history:
                history_stack.pop(0)
            # Clear redo stack when new changes are made
            redo_stack.clear()

    def undo_text(self, text_widget, history_stack, redo_stack, event=None):
        """Perform undo operation on the given text widget."""
        if len(history_stack) > 1:
            # Move current state to redo stack
            current_state = history_stack.pop()
            redo_stack.append(current_state)
            # Get previous state
            previous_text = history_stack[-1]
            # Update text widget without triggering the change event
            text_widget.unbind('<KeyRelease>')
            text_widget.delete("1.0", ctk.END)
            text_widget.insert("1.0", previous_text)
            text_widget.bind('<KeyRelease>', lambda e: self.on_text_change(text_widget, history_stack, redo_stack))
        return "break"  # Prevent the event from propagating

    def redo_text(self, text_widget, history_stack, redo_stack, event=None):
        """Perform redo operation on the given text widget."""
        if redo_stack:
            # Get the state to redo
            redo_state = redo_stack.pop()
            # Add it to the history stack
            history_stack.append(redo_state)
            # Update text widget without triggering the change event
            text_widget.unbind('<KeyRelease>')
            text_widget.delete("1.0", ctk.END)
            text_widget.insert("1.0", redo_state)
            text_widget.bind('<KeyRelease>', lambda e: self.on_text_change(text_widget, history_stack, redo_stack))
        return "break"  # Prevent the event from propagating

    def run(self):
        """Start the application"""
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        try:
            self.root.mainloop()
        except Exception as e:
            logger.error(f"Error in main loop: {str(e)}")
        finally:
            self.cleanup()

# Create and run the application when the script is executed
if __name__ == "__main__":
    app = CitationGeneratorUI()
    app.run()
