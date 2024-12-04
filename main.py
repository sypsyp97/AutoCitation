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
from utils import init_clients, generate_queries, search_arxiv
import json
import time
from loguru import logger

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
        self.root.title("Citation Generator")
        self.root.geometry("1200x800")  # Set a reasonable default size

        # Initialize the Gemini client
        success, self.client = init_clients()
        if not success:
            logger.error("Failed to initialize client")
            raise Exception("Client initialization failed")

        # Set up the user interface elements
        self.setup_ui()

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

        # Create the settings frame for controls
        settings_frame = ctk.CTkFrame(main_frame)
        settings_frame.pack(fill=ctk.X, pady=10)

        # Add controls for citations per query
        citations_label = ctk.CTkLabel(settings_frame, text="Citations per query:")
        citations_label.pack(side=ctk.LEFT, padx=5)

        self.citations_var = ctk.IntVar(value=2)  # Default value
        citations_entry = ctk.CTkEntry(settings_frame, textvariable=self.citations_var, width=50)
        citations_entry.pack(side=ctk.LEFT, padx=5)

        # Add controls for number of queries
        queries_label = ctk.CTkLabel(settings_frame, text="Number of queries:")
        queries_label.pack(side=ctk.LEFT, padx=10)

        self.queries_var = ctk.IntVar(value=2)  # Default value
        queries_entry = ctk.CTkEntry(settings_frame, textvariable=self.queries_var, width=50)
        queries_entry.pack(side=ctk.LEFT, padx=5)

        # Add the process button
        process_btn = ctk.CTkButton(settings_frame, text="Process", command=self.process_text)
        process_btn.pack(side=ctk.LEFT, padx=20)

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

        copy_bibtex_btn = ctk.CTkButton(bibtex_frame, text="Copy", command=lambda: self.copy_text(self.bibtex_text))
        copy_bibtex_btn.pack(pady=5)

    def copy_text(self, text_widget):
        """Copy the contents of a text widget to the clipboard."""
        text = text_widget.get("1.0", ctk.END)
        pyperclip.copy(text)

    def process_text(self):
        """Process the input text to generate citations and BibTeX entries."""
        try:
            # Get input text and parameters
            input_text = self.input_text.get("1.0", ctk.END).strip()
            citations_per_query = int(self.citations_var.get())
            num_queries = int(self.queries_var.get())

            if not input_text:
                return

            # Generate search queries using Gemini
            queries = generate_queries(self.client, input_text, num_queries)
            if not queries:
                return

            # Search for papers on arXiv
            papers = []
            for query in queries:
                paper_results = search_arxiv(query, max_results=citations_per_query)
                if paper_results:
                    papers.extend(paper_results)
                    time.sleep(3)  # Rate limiting for API

            if not papers:
                return

            # Generate citations using Gemini
            system_prompt = """You are a LaTeX citation assistant. Insert \\cite{} commands 
            into the text where appropriate. All provided papers must be cited at least once. 
            Only insert \\cite{} commands without changing anything else."""

            user_prompt = f"Insert \\cite{{}} commands into this text:\n\n" \
                          f"Papers:\n{json.dumps(papers, indent=2)}\n\n" \
                          f"Text:\n{input_text}"

            full_prompt = f"{system_prompt}\n\n{user_prompt}"

            # Generate citations and update UI
            response = self.client.generate_content(full_prompt)
            cited_text = response.text

            self.cited_text.delete("1.0", ctk.END)
            self.cited_text.insert("1.0", cited_text)

            # Generate and display BibTeX entries
            bibtex_entries = []
            for paper in papers:
                bibtex_entries.append(paper['bibtex_entry'])

            self.bibtex_text.delete("1.0", ctk.END)
            self.bibtex_text.insert("1.0", '\n\n'.join(bibtex_entries))

        except Exception as e:
            logger.error(f"Error processing text: {str(e)}")
            self.cited_text.delete("1.0", ctk.END)
            self.cited_text.insert("1.0", f"Error: {str(e)}")

    def run(self):
        """Start the UI application."""
        self.root.mainloop()

# Create and run the application when the script is executed
if __name__ == "__main__":
    app = CitationGeneratorUI()
    app.run()
