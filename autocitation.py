import asyncio
import json
import urllib.parse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, TypedDict
import sys
from loguru import logger

import aiohttp
import gradio as gr

from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

from langgraph.graph import StateGraph, END

# --- Helper Functions and Classes ---

def get_bibtex_writer() -> BibTexWriter:
    """Create and return a configured BibTexWriter instance."""
    writer = BibTexWriter()
    writer.indent = '    '
    writer.comma_first = False
    return writer

@dataclass
class Config:
    gemini_api_key: str
    max_retries: int = 3
    base_delay: int = 1
    max_queries: int = 5
    max_citations_per_query: int = 10
    arxiv_base_url: str = 'http://export.arxiv.org/api/query?'
    crossref_base_url: str = 'https://api.crossref.org/works'
    default_headers: Dict[str, str] = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    log_level: str = 'DEBUG'

class ArxivXmlParser:
    """Class to parse ArXiv XML responses."""
    NS = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    def parse_papers(self, data: str) -> List[Dict[str, Any]]:
        """Parse ArXiv XML data and return a list of paper dictionaries."""
        try:
            root = ET.fromstring(data)
            papers = []
            for entry in root.findall('atom:entry', self.NS):
                paper = self.parse_entry(entry)
                if paper:
                    papers.append(paper)
            return papers
        except Exception as e:
            logger.error(f"Error parsing ArXiv XML: {e}")
            return []

    def parse_entry(self, entry: ET.Element) -> Optional[Dict[str, Any]]:
        """Parse a single ArXiv entry element and return a dictionary with paper details."""
        try:
            title_node = entry.find('atom:title', self.NS)
            if title_node is None:
                return None
            title = title_node.text.strip() if title_node.text else ""

            authors = []
            for author in entry.findall('atom:author', self.NS):
                author_name_node = author.find('atom:name', self.NS)
                if author_name_node is not None and author_name_node.text:
                    authors.append(self._format_author_name(author_name_node.text.strip()))

            arxiv_id_node = entry.find('atom:id', self.NS)
            if arxiv_id_node is None or not arxiv_id_node.text:
                return None
            arxiv_id = arxiv_id_node.text.split('/')[-1]

            published_node = entry.find('atom:published', self.NS)
            year = published_node.text[:4] if (published_node is not None and published_node.text) else "Unknown"

            abstract_node = entry.find('atom:summary', self.NS)
            abstract = abstract_node.text.strip() if (abstract_node is not None and abstract_node.text) else ""

            bibtex_key = f"{authors[0].split(',')[0]}{arxiv_id.replace('.', '')}" if authors else f"unknown{arxiv_id.replace('.', '')}"
            bibtex_entry = self._generate_bibtex_entry(bibtex_key, title, authors, arxiv_id, year)

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
            logger.error(f"Error parsing ArXiv entry: {e}")
            return None

    @staticmethod
    def _format_author_name(author: str) -> str:
        """Format an author name as 'Lastname, Firstname'."""
        names = author.split()
        if len(names) > 1:
            return f"{names[-1]}, {' '.join(names[:-1])}"
        return author

    def _generate_bibtex_entry(self, key: str, title: str, authors: List[str], arxiv_id: str, year: str) -> str:
        """Generate a BibTeX entry for a paper."""
        db = BibDatabase()
        db.entries = [{
            'ENTRYTYPE': 'article',
            'ID': key,
            'title': title,
            'author': ' and '.join(authors),
            'journal': f'arXiv preprint arXiv:{arxiv_id}',
            'year': year
        }]
        writer = get_bibtex_writer()
        return writer.write(db).strip()

class CitationGenerator:
    """Class that provides methods for generating queries, searching papers, and managing LLM interactions."""
    def __init__(self, config: Config) -> None:
        self.config = config
        self.xml_parser = ArxivXmlParser()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0,
            google_api_key=config.gemini_api_key,
            streaming=True
        )
        self.citation_prompt = PromptTemplate.from_template(
            """Insert citations into the provided text using LaTeX \\cite{{key}} commands.
            
            You must not alter the original wording or structure of the text beyond adding citations.
            You must include all provided references at least once. Place citations at suitable points.

            Input text:
            {text}

            Available papers (cite each at least once):
            {papers}
            """
        )
        self.generate_queries_prompt = PromptTemplate.from_template(
            """Generate {num_queries} diverse academic search queries based on the given text.
            The queries should be concise and relevant.

            Requirements:
            1. Return ONLY a valid JSON array of strings.
            2. No additional text or formatting beyond JSON.
            3. Ensure uniqueness.

            Text: {text}
            """
        )
        self.fix_author_prompt = PromptTemplate.from_template(
            """Fix this author name that contains corrupted characters (�):

            Name: {author}

            Requirements:
            1. Return ONLY the fixed author name
            2. Use proper diacritical marks for names
            3. Consider common name patterns and languages
            4. If unsure, use the most likely letter
            5. Maintain the format: "Lastname, Firstname"
            """
        )
        logger.remove()
        logger.add(sys.stderr, level=config.log_level)

    async def generate_queries(self, text: str, num_queries: int) -> List[str]:
        """Generate a list of academic search queries from the input text."""
        input_map = {"text": text, "num_queries": num_queries}
        try:
            prompt = self.generate_queries_prompt.format(**input_map)
            response = await self.llm.ainvoke(prompt)
            content = response.content.strip()
            if not content.startswith('['):
                start = content.find('[')
                end = content.rfind(']') + 1
                if start >= 0 and end > start:
                    content = content[start:end]
            try:
                queries = json.loads(content)
                if isinstance(queries, list):
                    return [q.strip() for q in queries if isinstance(q, str)][:num_queries]
            except json.JSONDecodeError:
                lines = [line.strip() for line in content.split('\n')
                        if line.strip() and not line.strip().startswith(('[', ']'))]
                return lines[:num_queries]
            return ["deep learning neural networks"]
        except Exception as e:
            logger.error(f"Error generating queries: {e}")
            return ["deep learning neural networks"]

    async def search_arxiv(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search ArXiv for papers matching the query."""
        try:
            params = {
                'search_query': f'all:{urllib.parse.quote(query)}',
                'start': 0,
                'max_results': max_results,
                'sortBy': 'relevance',
                'sortOrder': 'descending'
            }
            url = self.config.arxiv_base_url + urllib.parse.urlencode(params)
            async with session.get(url, headers=self.config.default_headers, timeout=30) as response:
                text_data = await response.text()
                papers = self.xml_parser.parse_papers(text_data)
                return papers
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []

    async def search_crossref(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Search CrossRef for papers matching the query."""
        try:
            cleaned_query = query.replace("'", "").replace('"', "")
            if ' ' in cleaned_query:
                cleaned_query = f'"{cleaned_query}"'
            params = {
                'query.bibliographic': cleaned_query,
                'rows': max_results,
                'select': 'DOI,title,author,published-print,container-title',
                'sort': 'relevance',
                'order': 'desc'
            }
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; CitationBot/1.0; mailto:example@domain.com)',
                'Accept': 'application/json'
            }
            for attempt in range(self.config.max_retries):
                try:
                    async with session.get(
                        self.config.crossref_base_url,
                        params=params,
                        headers=headers,
                        timeout=30
                    ) as response:
                        if response.status == 429:
                            delay = self.config.base_delay * (2 ** attempt)
                            logger.warning(f"Rate limited by CrossRef. Retrying in {delay} seconds...")
                            await asyncio.sleep(delay)
                            continue
                        response.raise_for_status()
                        search_data = await response.json()
                        items = search_data.get('message', {}).get('items', [])
                        if not items:
                            return []
                        papers = []
                        existing_keys = set()
                        for item in items:
                            doi = item.get('DOI')
                            if not doi:
                                continue
                            try:
                                bibtex_url = f"https://doi.org/{doi}"
                                async with session.get(
                                    bibtex_url,
                                    headers={'Accept': 'application/x-bibtex', 'User-Agent': headers['User-Agent']},
                                    timeout=30
                                ) as bibtex_response:
                                    if bibtex_response.status != 200:
                                        continue
                                    bibtex_text = await bibtex_response.text()
                                    bib_database = bibtexparser.loads(bibtex_text)
                                    if not bib_database.entries:
                                        continue
                                    entry = bib_database.entries[0]
                                    if 'title' not in entry and 'booktitle' not in entry:
                                        continue
                                    if 'author' not in entry:
                                        continue
                                    title = entry.get('title', 'No Title').replace('{', '').replace('}', '')
                                    authors = entry.get('author', 'Unknown').replace('\n', ' ').replace('\t', ' ').strip()
                                    year = entry.get('year', 'Unknown')
                                    key = self._generate_unique_bibtex_key(entry, existing_keys)
                                    entry['ID'] = key
                                    existing_keys.add(key)
                                    writer = get_bibtex_writer()
                                    formatted_bibtex = writer.write(bib_database).strip()
                                    papers.append({
                                        'title': title,
                                        'authors': authors,
                                        'year': year,
                                        'bibtex_key': key,
                                        'bibtex_entry': formatted_bibtex
                                    })
                            except Exception as e:
                                logger.error(f"Error processing CrossRef item: {e}")
                        return papers
                except aiohttp.ClientError as e:
                    if attempt == self.config.max_retries - 1:
                        logger.error(f"Max retries reached for CrossRef search. Error: {e}")
                        raise
                    delay = self.config.base_delay * (2 ** attempt)
                    logger.warning(f"Client error during CrossRef search: {e}. Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)
        except Exception as e:
            logger.error(f"Error searching CrossRef: {e}")
            return []

    def _generate_unique_bibtex_key(self, entry: Dict[str, Any], existing_keys: set) -> str:
        """Generate a unique BibTeX key for an entry."""
        entry_type = entry.get('ENTRYTYPE', '').lower()
        author_field = entry.get('author', '')
        year = entry.get('year', '')
        authors = [a.strip() for a in author_field.split(' and ')]
        first_author_last_name = authors[0].split(',')[0] if authors else 'unknown'
        title_field = entry.get('booktitle' if entry_type == 'inbook' else 'title', '')
        title_word = re.sub(r'\W+', '', title_field.split()[0]) if title_field.split() else 'untitled'
        base_key = f"{first_author_last_name}{year}{title_word}"
        key = base_key
        index = 1
        while key in existing_keys:
            key = f"{base_key}{index}"
            index += 1
        return key

    async def fix_author_name(self, author: str) -> str:
        """Correct an author name that contains corrupted characters."""
        if not re.search(r'[�]', author):
            return author
        try:
            prompt = self.fix_author_prompt.format(author=author)
            response = await self.llm.ainvoke(prompt)
            fixed_name = response.content.strip()
            return fixed_name if fixed_name else author
        except Exception as e:
            logger.error(f"Error fixing author name: {e}")
            return author

    async def format_bibtex_author_names(self, papers: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Clean and format author names in BibTeX entries of papers."""
        updated_papers = []
        for paper in papers:
            try:
                bib_database = bibtexparser.loads(paper['bibtex_entry'])
                for entry in bib_database.entries:
                    if 'author' in entry:
                        authors = entry['author'].split(' and ')
                        cleaned_authors = []
                        for author in authors:
                            fixed_author = await self.fix_author_name(author.strip())
                            cleaned_authors.append(fixed_author)
                        entry['author'] = ' and '.join(cleaned_authors)
                writer = get_bibtex_writer()
                paper['bibtex_entry'] = writer.write(bib_database).strip()
                updated_papers.append(paper)
            except Exception as e:
                logger.error(f"Error cleaning BibTeX author names for paper {paper['bibtex_key']}: {e}")
                updated_papers.append(paper)  # Append unchanged if error occurs
        return updated_papers

# --- LangGraph State Definition ---

class State(TypedDict):
    input_text: str
    num_queries: int
    citations_per_query: int
    use_arxiv: bool
    use_crossref: bool
    generated_queries: List[str]
    papers: List[Dict[str, Any]]
    cited_text: str
    bibtex_entries: str

# --- Gradio Interface with LangGraph ---

def create_gradio_interface() -> gr.Interface:
    """Create and return a Gradio interface for the citation generator using LangGraph."""
    async def process(api_key: str, text: str, num_queries: int, citations_per_query: int,
                      use_arxiv: bool, use_crossref: bool) -> Tuple[str, str, str]:
        if not api_key.strip():
            return "Please enter your Gemini API Key.", "", ""
        if not text.strip():
            return "Please enter text to process", "", ""
        if not (use_arxiv or use_crossref):
            return "Please select at least one source (ArXiv or CrossRef)", "", ""
        try:
            config = Config(gemini_api_key=api_key)
            citation_gen = CitationGenerator(config)
            async with aiohttp.ClientSession() as session:
                # Define nodes
                async def generate_queries_node(state: State) -> dict:
                    queries = await citation_gen.generate_queries(state['input_text'], state['num_queries'])
                    return {'generated_queries': queries}

                async def search_papers_node(state: State) -> dict:
                    queries = state['generated_queries']
                    papers = []
                    search_tasks = []
                    for q in queries:
                        if state['use_arxiv']:
                            search_tasks.append(citation_gen.search_arxiv(session, q, state['citations_per_query']))
                        if state['use_crossref']:
                            search_tasks.append(citation_gen.search_crossref(session, q, state['citations_per_query']))
                    results = await asyncio.gather(*search_tasks, return_exceptions=True)
                    for r in results:
                        if not isinstance(r, Exception):
                            papers.extend(r)
                    # Remove duplicates
                    unique_papers = []
                    seen_keys = set()
                    for p in papers:
                        if p['bibtex_key'] not in seen_keys:
                            seen_keys.add(p['bibtex_key'])
                            unique_papers.append(p)
                    return {'papers': unique_papers}

                async def correct_author_names_node(state: State) -> dict:
                    if not state['papers']:
                        return {'papers': state['papers']}
                    corrected_papers = await citation_gen.format_bibtex_author_names(state['papers'])
                    return {'papers': corrected_papers}

                async def insert_citations_node(state: State) -> dict:
                    if not state['papers']:
                        return {'cited_text': state['input_text'], 'bibtex_entries': ""}
                    citation_input = {
                        "text": state['input_text'],
                        "papers": json.dumps(state['papers'], indent=2)
                    }
                    prompt = citation_gen.citation_prompt.format(**citation_input)
                    response = await citation_gen.llm.ainvoke(prompt)
                    cited_text = response.content.strip()
                    bib_database = BibDatabase()
                    for p in state['papers']:
                        if 'bibtex_entry' in p:
                            bib_db = bibtexparser.loads(p['bibtex_entry'])
                            if bib_db.entries:
                                bib_database.entries.append(bib_db.entries[0])
                    writer = get_bibtex_writer()
                    bibtex_entries = writer.write(bib_database).strip()
                    return {'cited_text': cited_text, 'bibtex_entries': bibtex_entries}

                # Build and compile the graph
                graph = StateGraph(State)
                graph.add_node("generate_queries", generate_queries_node)
                graph.add_node("search_papers", search_papers_node)
                graph.add_node("correct_author_names", correct_author_names_node)
                graph.add_node("insert_citations", insert_citations_node)
                graph.set_entry_point("generate_queries")
                graph.add_edge("generate_queries", "search_papers")
                graph.add_edge("search_papers", "correct_author_names")
                graph.add_edge("correct_author_names", "insert_citations")
                graph.add_edge("insert_citations", END)
                compiled_graph = graph.compile()

                # Initialize state
                initial_state = State(
                    input_text=text,
                    num_queries=min(max(1, int(num_queries)), config.max_queries),
                    citations_per_query=min(max(1, int(citations_per_query)), config.max_citations_per_query),
                    use_arxiv=use_arxiv,
                    use_crossref=use_crossref,
                    generated_queries=[],
                    papers=[],
                    cited_text="",
                    bibtex_entries=""
                )

                # Execute the graph
                final_state = await compiled_graph.ainvoke(initial_state)

                return (
                    final_state['cited_text'],
                    final_state['bibtex_entries'],
                    "\n".join(final_state['generated_queries'])
                )
        except Exception as e:
            return f"Error: {str(e)}", "", ""

    # CSS styling (unchanged from original)
    css = """
        :root {
            --primary-bg: #F8F9FA;
            --secondary-bg: #FFFFFF;
            --accent-1: #4A90E2;
            --accent-2: #50C878;
            --accent-3: #F5B041;
            --text-primary: #2C3E50;
            --text-secondary: #566573;
            --border: #E5E7E9;
            --shadow: rgba(0, 0, 0, 0.1);
        }
        body {
            background-color: var(--primary-bg);
            color: var(--text-primary);
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
            line-height: 1.6;
        }
        .header { text-align: center; margin-bottom: 2rem; padding: 1.5rem; background-color: var(--secondary-bg); box-shadow: 0 2px 4px var(--shadow); border-bottom: none; }
        .header h1 { font-size: 2.25rem; color: var(--accent-1); margin-bottom: 0.75rem; font-weight: 600; }
        .header p { font-size: 1.1rem; color: var(--text-secondary); }
        .input-group, .controls-row, .source-controls, .output-group { padding: 1rem; margin-bottom: 1.5rem; border: 1px solid var(--border); border-radius: 12px; background-color: var(--secondary-bg); box-shadow: 0 1px 3px var(--shadow); }
        .input-group label, .controls-row label, .source-controls label { color: var(--text-primary); font-weight: 500; margin-bottom: 0.5rem; display: block; }
        input[type="number"], textarea, .gradio-input, .gradio-output { border: 1px solid var(--border); border-radius: 8px; padding: 0.75rem; background-color: var(--primary-bg); color: var(--text-primary); font-size: 1rem; width: 100%; transition: border-color 0.3s, box-shadow 0.3s; }
        input[type="number"]:focus, textarea:focus { border-color: var(--accent-1); box-shadow: 0 0 0 2px rgba(74, 144, 226, 0.1); outline: none; }
        .generate-btn { background-color: var(--accent-1); color: white; padding: 1rem 2rem; border: none; border-radius: 8px; font-size: 1.1rem; font-weight: 500; cursor: pointer; transition: all 0.3s ease; width: 100%; }
        .generate-btn:hover { background-color: #357ABD; transform: translateY(-1px); box-shadow: 0 4px 6px var(--shadow); }
        .gradio-button { background-color: var(--accent-2) !important; border-radius: 8px !important; transition: all 0.3s ease !important; }
        .gradio-button:hover { background-color: #45B76C !important; transform: translateY(-1px); }
        .gradio-copy-button { background-color: var(--accent-3) !important; color: var(--text-primary) !important; border: none !important; border-radius: 6px !important; padding: 0.4rem 0.8rem !important; cursor: pointer !important; font-size: 0.9rem !important; font-weight: 500 !important; transition: all 0.3s ease !important; }
        .gradio-copy-button:hover { background-color: #F39C12 !important; transform: translateY(-1px); box-shadow: 0 2px 4px var(--shadow); }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
        gr.HTML("""
            <div class="header">
                <h1>📚 AutoCitation</h1>
                <p>An AI agent that automatically adds citations into your academic text</p>
            </div>
        """)
        api_key = gr.Textbox(label="Gemini API Key", placeholder="Enter your Gemini API key...", type="password")
        input_text = gr.Textbox(label="Input Text", placeholder="Paste or type your text here...", lines=8)
        with gr.Row():
            num_queries = gr.Number(label="Search Queries", value=3, minimum=1, maximum=Config.max_queries, step=1)
            citations_per_query = gr.Number(label="Citations per Query", value=1, minimum=1, maximum=Config.max_citations_per_query, step=1)
        with gr.Row():
            use_arxiv = gr.Checkbox(label="Search arXiv", value=True)
            use_crossref = gr.Checkbox(label="Search Crossref", value=True)
        process_btn = gr.Button("Generate", elem_classes="generate-btn")
        with gr.Row():
            cited_text = gr.Textbox(label="Generated Text", lines=10, show_copy_button=True)
            bibtex = gr.Textbox(label="BibTeX References", lines=10, show_copy_button=True)
            queries_text = gr.Textbox(label="Generated Queries", lines=10, show_copy_button=True)

        process_btn.click(
            fn=process,
            inputs=[api_key, input_text, num_queries, citations_per_query, use_arxiv, use_crossref],
            outputs=[cited_text, bibtex, queries_text]
        )

    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    try:
        demo.launch(server_port=7860, share=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {str(e)}")