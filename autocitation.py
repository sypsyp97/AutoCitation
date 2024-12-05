"""AutoCitation: A tool for automatic paper citation generation.

This module provides functionality to generate academic paper citations using ArXiv
and integrate them into text using LaTeX citation commands.
"""

# Standard library imports
import asyncio
import json
import os
import urllib.parse
import xml.etree.ElementTree as ET

# Third-party imports
import aiohttp
import gradio as gr
from dataclasses import dataclass, field
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from loguru import logger
from typing import Dict, List, Optional

load_dotenv()

@dataclass
class Config:
    """Configuration class for the citation generator.
    
    Attributes:
        gemini_api_key: API key for Google's Gemini model.
        max_queries: Maximum number of search queries to generate.
        max_citations_per_query: Maximum number of citations per search query.
        arxiv_base_url: Base URL for ArXiv API.
        default_headers: Default HTTP headers for API requests.
    """
    
    gemini_api_key: str
    max_queries: int = 5
    max_citations_per_query: int = 10
    arxiv_base_url: str = 'http://export.arxiv.org/api/query?'
    default_headers: dict = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })


class ArxivXmlParser:
    """Parser for ArXiv API XML responses."""

    NS = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }
    
    def parse_papers(self, data: str) -> List[Dict]:
        try:
            root = ET.fromstring(data)
            return [paper for entry in root.findall('atom:entry', self.NS) 
                   if (paper := self.parse_entry(entry))]
        except Exception as e:
            logger.error(f"Error parsing ArXiv response: {str(e)}")
            return []

    def parse_entry(self, entry) -> Optional[dict]:
        try:
            title = entry.find('atom:title', self.NS).text.strip()
            authors = [self._format_author_name(author.find('atom:name', self.NS).text)
                      for author in entry.findall('atom:author', self.NS)]
            arxiv_id = entry.find('atom:id', self.NS).text.split('/')[-1]
            year = entry.find('atom:published', self.NS).text[:4]
            abstract = entry.find('atom:summary', self.NS).text.strip()
            
            bibtex_key = f"{authors[0].split(',')[0]}{arxiv_id.replace('.', '')}"
            bibtex_entry = self._generate_bibtex(bibtex_key, title, authors, arxiv_id, year)
            
            return {
                'title': title,
                'authors': authors,
                'arxiv_id': arxiv_id,
                'published': year,
                'abstract': abstract,
                'bibtex_key': bibtex_key,
                'bibtex_entry': bibtex_entry
            }
        except Exception as e:
            logger.warning(f"Error processing paper entry: {str(e)}")
            return None

    @staticmethod
    def _format_author_name(author: str) -> str:
        names = author.split()
        return f"{names[-1]}, {' '.join(names[:-1])}" if len(names) > 1 else author

    @staticmethod
    def _generate_bibtex(key: str, title: str, authors: List[str], arxiv_id: str, year: str) -> str:
        return f"""@article{{{key},
            title={{{title}}},
            author={{{' and '.join(authors)}}},
            journal={{arXiv preprint arXiv:{arxiv_id}}},
            year={{{year}}}
        }}"""

class AsyncContextManager:
    """A context manager for handling async HTTP sessions.
    
    This class provides an async context manager interface for creating and
    cleaning up aiohttp ClientSession objects.
    """
    
    async def __aenter__(self):
        """Creates and returns a new aiohttp ClientSession.
        
        Returns:
            aiohttp.ClientSession: A new HTTP client session.
        """
        self._session = aiohttp.ClientSession()
        return self._session
        
    async def __aexit__(self, *_):
        """Closes the HTTP client session if it exists."""
        if self._session:
            await self._session.close()

class CitationGenerator:
    """Main class for generating citations from text using ArXiv papers.
    
    This class handles the end-to-end process of:
    1. Generating search queries from input text
    2. Searching ArXiv for relevant papers
    3. Generating citations and BibTeX entries
    
    Attributes:
        config: Configuration object containing API keys and limits
        xml_parser: Parser for ArXiv API responses
        async_context: Context manager for HTTP sessions
        llm: Language model for generating queries and citations
        query_chain: Chain for generating search queries
        citation_chain: Chain for generating citations
    """
    
    def __init__(self, config: Config):
        """Initializes the citation generator.
        
        Args:
            config: Configuration object containing necessary parameters
        """
        self.config = config
        self.xml_parser = ArxivXmlParser()
        self.async_context = AsyncContextManager()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=config.gemini_api_key,
            streaming=True
        )
        self.query_chain = self._create_query_chain()
        self.citation_chain = self._create_citation_chain()
    
    def _create_query_chain(self):
        """Creates a chain for generating search queries from input text.
        
        Returns:
            Chain: A LangChain chain that generates search queries.
        """
        query_prompt = PromptTemplate.from_template(
            """Generate {num_queries} precise academic search queries for papers 
            related to this text. Focus on different aspects of the content.
            
            Text to analyze: {text}
            
            Return the queries as a JSON array of strings.
            Format example: ["query1", "query2"]
            
            Make each query specific and targeted to find relevant academic papers."""
        )
        
        return (
            {"text": RunnablePassthrough(), "num_queries": RunnablePassthrough()}
            | query_prompt
            | self.llm
            | StrOutputParser()
        )
    
    def _create_citation_chain(self):
        """Creates a chain for generating citations from papers.
        
        Returns:
            Chain: A LangChain chain that generates LaTeX citations.
        """
        citation_prompt = PromptTemplate.from_template(
            """Insert citations into the text using LaTeX \\cite{{key}} commands.

            Do not change the original text. The only thing to change is the citation commands.

            Input text:
            {text}

            Available papers (cite all at least once):
            {papers}""")
        
        return (
            {"text": RunnablePassthrough(), "papers": RunnablePassthrough()}
            | citation_prompt
            | self.llm
            | StrOutputParser()
        )

    async def generate_queries(self, text: str, num_queries: int) -> List[str]:
        """Generates search queries from input text.
        
        Args:
            text: Input text to generate queries from.
            num_queries: Number of queries to generate.
            
        Returns:
            List[str]: List of generated search queries.
        """
        try:
            response = await self.query_chain.ainvoke({"text": text, "num_queries": num_queries})
            start, end = response.find("["), response.rfind("]") + 1
            if start >= 0 and end > start:
                queries = json.loads(response[start:end].replace("'", '"'))
                return [q.strip() for q in queries if isinstance(q, str)][:num_queries]
            return [q.strip() for q in response.split("\n") if q.strip()][:num_queries]
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}")
            return ["deep learning neural networks"]

    async def search_arxiv(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict]:
        """Searches ArXiv for papers matching a query.
        
        Args:
            session: HTTP client session.
            query: Search query string.
            max_results: Maximum number of results to return.
            
        Returns:
            List[Dict]: List of paper metadata dictionaries.
        """
        try:
            params = {
                'search_query': f'all:{urllib.parse.quote(query)}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            async with session.get(
                self.config.arxiv_base_url + urllib.parse.urlencode(params),
                headers=self.config.default_headers,
                timeout=30
            ) as response:
                return self.xml_parser.parse_papers(await response.text())
        except Exception as e:
            logger.error(f"ArXiv search error: {str(e)}")
            return []

    async def process_text(self, text: str, num_queries: int, citations_per_query: int) -> tuple[str, str]:
        """Processes input text to generate citations.
        
        This method:
        1. Generates search queries from the input text
        2. Searches ArXiv for relevant papers
        3. Generates citations and BibTeX entries
        
        Args:
            text: Input text to process.
            num_queries: Number of search queries to generate.
            citations_per_query: Number of citations to generate per query.
            
        Returns:
            tuple[str, str]: A tuple containing:
                - The input text with added citations
                - BibTeX entries for the cited papers
        """
        num_queries = min(max(1, num_queries), self.config.max_queries)
        citations_per_query = min(max(1, citations_per_query), self.config.max_citations_per_query)
        
        if not (queries := await self.generate_queries(text, num_queries)):
            return text, ""
        
        async with self.async_context as session:
            papers = []
            results = await asyncio.gather(
                *[self.search_arxiv(session, query, citations_per_query) for query in queries],
                return_exceptions=True
            )
            papers = [p for r in results if not isinstance(r, Exception) for p in r]
            
        if not papers:
            return text, ""
        
        try:
            cited_text = await self.citation_chain.ainvoke({
                "text": text,
                "papers": json.dumps(papers, indent=2)
            })
            bibtex_entries = "\n\n".join(p['bibtex_entry'] for p in papers if 'bibtex_entry' in p)
            return cited_text, bibtex_entries
        except Exception as e:
            logger.error(f"Citation generation error: {str(e)}")
            return text, ""

def create_gradio_interface(config: Config) -> gr.Interface:
    """Creates a Gradio web interface for the citation generator.
    
    This function creates a web interface with:
    - A text input area for the user's content
    - Controls for number of queries and citations
    - Output areas for cited text and BibTeX entries
    
    Args:
        config: Configuration object for the citation generator
        
    Returns:
        gr.Interface: A configured Gradio interface object
    """
    citation_gen = CitationGenerator(config)
    
    async def process(text: str, num_queries: int, citations_per_query: int) -> tuple[str, str]:
        """Processes user input and generates citations.
        
        Args:
            text: User's input text
            num_queries: Number of search queries to generate
            citations_per_query: Number of citations per query
            
        Returns:
            tuple[str, str]: Tuple of (cited text, BibTeX entries)
        """
        if not text.strip():
            return "Please enter text to process", ""
        try:
            return await citation_gen.process_text(text, num_queries, citations_per_query)
        except ValueError as e:
            return f"Input validation error: {str(e)}", ""
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            return f"Error: {str(e)}", ""

    css = """
        :root {
            --primary: #6A7E76;
            --primary-hover: #566961;
            --bg: #FFFFFF;
            --text: #454442;
            --border: #B4B0AC;
            --control-bg: #F5F3F0;
        }
        
        .container {
            max-width: 100%;
            padding: 0.75rem;
            background: var(--bg);
        }
        
        .header {
            text-align: center;
            margin-bottom: 1rem;
            padding: 0.75rem;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
        }
        
        .header h1 {
            font-size: 1.5rem;
            color: var(--primary);
            font-weight: 500;
            margin-bottom: 0.25rem;
        }
        
        .header p {
            font-size: 0.9rem;
            color: var(--text);
        }
        
        .input-group {
            padding: 1rem;
            border-radius: 4px;
            border: 1px solid var(--border);
            margin-bottom: 0.75rem;
        }
        
        .controls-row {
            gap: 0.75rem !important;
            margin-top: 0.5rem;
        }
        
        input[type="number"] {
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            padding: 0.5rem !important;
            background: var(--control-bg) !important;
            color: var(--text) !important;
        }
        
        textarea {
            border: 1px solid var(--border) !important;
            border-radius: 4px !important;
            padding: 0.75rem !important;
            background: var(--control-bg) !important;
            color: var(--text) !important;
            font-size: 0.95rem !important;
        }
        
        .generate-btn {
            background: var(--primary) !important;
            color: white !important;
            padding: 0.5rem 1.5rem !important;
            border-radius: 4px !important;
            border: none !important;
            font-size: 0.9rem !important;
            transition: background 0.2s !important;
            width: 100% !important;
        }
        
        .generate-btn:hover {
            background: var(--primary-hover) !important;
        }
        
        .output-container {
            display: flex;
            gap: 0.75rem;
        }
        
        label span {
            color: var(--text) !important;
            font-size: 0.9rem !important;
        }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
        gr.HTML("""<div class="header">
            <h1>📚 AutoCitation</h1>
            <p>Make your academic writing easier with AI-powered academic citations</p>
        </div>""")
        
        with gr.Group(elem_classes="input-group"):
            input_text = gr.Textbox(
                label="Input Text",
                placeholder="Paste or type your text here...",
                lines=8
            )
            with gr.Row(elem_classes="controls-row"):
                with gr.Column(scale=1):
                    num_queries = gr.Number(
                        label="Search Queries",
                        value=3,
                        minimum=1,
                        maximum=config.max_queries,
                        step=1
                    )
                with gr.Column(scale=1):
                    citations_per_query = gr.Number(
                        label="Citations per Query",
                        value=1,
                        minimum=1,
                        maximum=config.max_citations_per_query,
                        step=1
                    )
                with gr.Column(scale=2):
                    process_btn = gr.Button(
                        "Generate",
                        elem_classes="generate-btn"
                    )
        
        with gr.Group(elem_classes="output-group"):
            with gr.Row():
                with gr.Column(scale=1):
                    cited_text = gr.Textbox(
                        label="Generated Text",
                        lines=10,
                        show_copy_button=True
                    )
                with gr.Column(scale=1):
                    bibtex = gr.Textbox(
                        label="BibTeX References",
                        lines=10,
                        show_copy_button=True
                    )
        
        process_btn.click(
            fn=process,
            inputs=[input_text, num_queries, citations_per_query],
            outputs=[cited_text, bibtex]
        )
    
    return demo

if __name__ == "__main__":
    config = Config(
        gemini_api_key=os.getenv("GEMINI_API_KEY")
    )
    if not config.gemini_api_key:
        raise EnvironmentError("GEMINI_API_KEY not found in environment variables")
        
    demo = create_gradio_interface(config)
    try:
        demo.launch(
            server_port=7860,
            share=True,
        )
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
    except Exception as e:
        print(f"Error starting server: {str(e)}")