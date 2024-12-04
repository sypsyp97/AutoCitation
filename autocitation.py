import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from dotenv import load_dotenv
import gradio as gr
import asyncio
import aiohttp
import json
import urllib.parse
import xml.etree.ElementTree as ET
from loguru import logger

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()

@dataclass
class Config:
    GEMINI_API_KEY: str
    MAX_QUERIES: int = 5
    MAX_CITATIONS_PER_QUERY: int = 10
    ARXIV_BASE_URL: str = 'http://export.arxiv.org/api/query?'
    DEFAULT_HEADERS: dict = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

class ArxivXMLParser:
    def __init__(self):
        self.ns = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
    
    def parse_papers(self, data: str) -> List[Dict]:
        papers = []
        try:
            root = ET.fromstring(data)
            entries = root.findall('atom:entry', self.ns)
            used_keys = set()
            
            for entry in entries:
                paper = self.parse_entry(entry, used_keys)
                if paper:
                    papers.append(paper)
                    
        except Exception as e:
            logger.error(f"Error parsing ArXiv response: {str(e)}")
            
        return papers

    def parse_entry(self, entry, used_keys: set) -> Optional[dict]:
        try:
            title = entry.find('atom:title', self.ns).text.strip()
            authors = self._parse_authors(entry)
            arxiv_id = entry.find('atom:id', self.ns).text.split('/')[-1]
            published = entry.find('atom:published', self.ns).text
            abstract = entry.find('atom:summary', self.ns).text.strip()
            
            bibtex_key = self._generate_unique_key(authors[0], arxiv_id, used_keys)
            bibtex_entry = self._generate_bibtex(bibtex_key, title, authors, arxiv_id, published[:4])
            
            return {
                'title': title,
                'authors': authors,
                'arxiv_id': arxiv_id,
                'published': published,
                'abstract': abstract,
                'bibtex_key': bibtex_key,
                'bibtex_entry': bibtex_entry
            }
        except Exception as e:
            logger.warning(f"Error processing paper entry: {str(e)}")
            return None

    def _parse_authors(self, entry) -> List[str]:
        return [
            self._format_author_name(author.find('atom:name', self.ns).text)
            for author in entry.findall('atom:author', self.ns)
        ]

    @staticmethod
    def _format_author_name(author: str) -> str:
        names = author.split()
        return f"{names[-1]}, {' '.join(names[:-1])}" if len(names) > 1 else author

    def _generate_unique_key(self, first_author: str, arxiv_id: str, used_keys: set) -> str:
        base_key = f"{first_author.split()[-1]}{arxiv_id.replace('.', '')}"
        key = base_key
        counter = 1
        while key in used_keys:
            key = f"{base_key}_{counter}"
            counter += 1
        used_keys.add(key)
        return key

    def _generate_bibtex(self, key: str, title: str, authors: List[str], arxiv_id: str, year: str) -> str:
        return f"""@article{{{key},
                title={{{title}}},
                author={{{' and '.join(authors)}}},
                journal={{arXiv preprint arXiv:{arxiv_id}}},
                year={{{year}}}
            }}"""

class AsyncContextManager:
    def __init__(self):
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self._session
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._session:
            await self._session.close()

class CitationGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.xml_parser = ArxivXMLParser()
        self.async_context = AsyncContextManager()
        
        self.llm = self._initialize_llm()
        self.query_chain = self._create_query_chain()
        self.citation_chain = self._create_citation_chain()
    
    def _initialize_llm(self):
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.7,
            google_api_key=self.config.GEMINI_API_KEY,
            streaming=True
        )
    
    def _create_query_chain(self):
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
        try:
            system_prompt = """Generate multiple distinct search queries. Return JSON array of strings.
            Example: ["query1", "query2", "query3"]"""
            
            response = await self.query_chain.ainvoke({
                "text": f"{system_prompt}\n\nText: {text}\nNumber of queries: {num_queries}"
            })
            
            try:
                # Extract JSON array if present
                start = response.find("[")
                end = response.rfind("]") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end].replace("'", '"')
                    queries = json.loads(json_str)
                    if isinstance(queries, list):
                        return [q.strip() for q in queries if isinstance(q, str)][:num_queries]
                
                # Fallback: split by newlines
                queries = [q.strip() for q in response.split("\n") if q.strip()]
                return queries[:num_queries]
            
            except json.JSONDecodeError:
                return [response.strip()][:num_queries]
            
        except Exception as e:
            logger.error(f"Error generating queries: {str(e)}")
            return ["deep learning neural networks"]

    async def search_arxiv(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict]:
        cleaned_query = urllib.parse.quote(query)
        params = {
            'search_query': f'all:{cleaned_query}',
            'start': 0,
            'max_results': max_results,
            'sortBy': 'relevance',
            'sortOrder': 'descending'
        }
        
        query_url = self.config.ARXIV_BASE_URL + urllib.parse.urlencode(params)
        
        try:
            async with session.get(query_url, headers=self.config.DEFAULT_HEADERS, timeout=30) as response:
                data = await response.text()
                return self.xml_parser.parse_papers(data)
        except Exception as e:
            logger.error(f"ArXiv search error: {str(e)}")
            return []

    async def _gather_papers(self, session: aiohttp.ClientSession, queries: List[str], citations_per_query: int) -> List[Dict]:
        papers = []
        tasks = [
            self.search_arxiv(session, query, citations_per_query)
            for query in queries
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Search failed: {str(result)}")
                continue
            papers.extend(result)
        
        return papers

    async def _generate_citations(self, text: str, papers: List[Dict]) -> str:
        return await self.citation_chain.ainvoke({
            "text": text,
            "papers": json.dumps(papers, indent=2)
        })

    @staticmethod
    def _format_bibtex_entries(papers: List[Dict]) -> str:
        return "\n\n".join(
            p['bibtex_entry'] for p in papers
            if 'bibtex_entry' in p
        )

    async def process_text(self, text: str, num_queries: int, citations_per_query: int) -> tuple[str, str]:
        num_queries = min(max(1, num_queries), self.config.MAX_QUERIES)
        citations_per_query = min(max(1, citations_per_query), self.config.MAX_CITATIONS_PER_QUERY)
        
        queries = await self.generate_queries(text, num_queries)
        if not queries:
            return text, ""
        
        async with self.async_context as session:
            papers = await self._gather_papers(session, queries, citations_per_query)
            
        if not papers:
            return text, ""
        
        try:
            cited_text = await self._generate_citations(text, papers)
            bibtex_entries = self._format_bibtex_entries(papers)
            return cited_text, bibtex_entries
        except Exception as e:
            logger.error(f"Citation generation error: {str(e)}")
            return text, ""

def create_gradio_interface(config: Config) -> gr.Interface:
    citation_gen = CitationGenerator(config)
    
    async def process(text: str, num_queries: int, citations_per_query: int) -> tuple[str, str]:
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
        gr.HTML("""
            <div class="header">
                <h1>📚 AutoCitation</h1>
                <p>Make your academic writing easier with AI-powered academic citations</p>
            </div>
        """)
        
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
                        maximum=config.MAX_QUERIES,
                        step=1
                    )
                with gr.Column(scale=1):
                    citations_per_query = gr.Number(
                        label="Citations per Query",
                        value=1,
                        minimum=1,
                        maximum=config.MAX_CITATIONS_PER_QUERY,
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
        GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
    )
    if not config.GEMINI_API_KEY:
        raise EnvironmentError("GEMINI_API_KEY not found in environment variables")
        
    demo = create_gradio_interface(config)
    try:
        demo.launch(
            server_port=7860,
            share=False,
        )
    except KeyboardInterrupt:
        print("\nShutting down server gracefully...")
    except Exception as e:
        print(f"Error starting server: {str(e)}")