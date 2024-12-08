import asyncio
import json
import os
import urllib.parse
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import sys
from loguru import logger

import aiohttp
import gradio as gr

from langchain.prompts import PromptTemplate

from langchain_google_genai import ChatGoogleGenerativeAI

import bibtexparser
from bibtexparser.bwriter import BibTexWriter
from bibtexparser.bibdatabase import BibDatabase

@dataclass
class Config:
    gemini_api_key: str
    max_retries: int = 3
    base_delay: int = 1
    max_queries: int = 5
    max_citations_per_query: int = 10
    arxiv_base_url: str = 'http://export.arxiv.org/api/query?'
    crossref_base_url: str = 'https://api.crossref.org/works'
    default_headers: dict = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })
    log_level: str = 'DEBUG'

class ArxivXmlParser:
    NS = {
        'atom': 'http://www.w3.org/2005/Atom',
        'arxiv': 'http://arxiv.org/schemas/atom'
    }

    def parse_papers(self, data: str) -> List[Dict]:
        try:
            root = ET.fromstring(data)
            papers = []
            for entry in root.findall('atom:entry', self.NS):
                paper = self.parse_entry(entry)
                if paper:
                    papers.append(paper)
            return papers
        except Exception as e:
            print(f"Error parsing ArXiv XML: {e}")
            return []

    def parse_entry(self, entry) -> Optional[dict]:
        try:
            title_node = entry.find('atom:title', self.NS)
            if title_node is None:
                return None
            title = title_node.text.strip()

            authors = []
            for author in entry.findall('atom:author', self.NS):
                author_name_node = author.find('atom:name', self.NS)
                if author_name_node is not None and author_name_node.text:
                    authors.append(self._format_author_name(author_name_node.text.strip()))

            arxiv_id_node = entry.find('atom:id', self.NS)
            if arxiv_id_node is None:
                return None
            arxiv_id = arxiv_id_node.text.split('/')[-1]

            published_node = entry.find('atom:published', self.NS)
            year = published_node.text[:4] if published_node is not None else "Unknown"

            abstract_node = entry.find('atom:summary', self.NS)
            abstract = abstract_node.text.strip() if abstract_node is not None else ""

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
            print(f"Error parsing ArXiv entry: {e}")
            return None

    @staticmethod
    def _format_author_name(author: str) -> str:
        names = author.split()
        if len(names) > 1:
            return f"{names[-1]}, {' '.join(names[:-1])}"
        return author

    def _generate_bibtex_entry(self, key: str, title: str, authors: List[str], arxiv_id: str, year: str) -> str:
        db = BibDatabase()
        db.entries = [{
            'ENTRYTYPE': 'article',
            'ID': key,
            'title': title,
            'author': ' and '.join(authors),
            'journal': f'arXiv preprint arXiv:{arxiv_id}',
            'year': year
        }]
        writer = BibTexWriter()
        writer.indent = '    '
        writer.comma_first = False
        return writer.write(db).strip()

class AsyncContextManager:
    async def __aenter__(self):
        self._session = aiohttp.ClientSession()
        return self._session

    async def __aexit__(self, *_):
        if self._session:
            await self._session.close()

class CitationGenerator:
    def __init__(self, config: Config):
        self.config = config
        self.xml_parser = ArxivXmlParser()
        self.async_context = AsyncContextManager()
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.3,
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

        logger.remove()
        logger.add(sys.stderr, level=config.log_level)

    async def generate_queries(self, text: str, num_queries: int) -> List[str]:
        input_map = {
            "text": text,
            "num_queries": num_queries
        }
        try:
            prompt = self.generate_queries_prompt.format(**input_map)
            response = await self.llm.apredict(prompt)
            content = response.strip()
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

    async def search_arxiv(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict]:
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
                text_data = await response.text()
                papers = self.xml_parser.parse_papers(text_data)
                return papers
        except Exception as e:
            logger.error(f"Error searching ArXiv: {e}")
            return []

    async def fix_author_name(self, author: str) -> str:
        if not re.search(r'[�]', author):
            return author
        try:
            prompt = f"""Fix this author name that contains corrupted characters (�):

                    Name: {author}

                    Requirements:
                    1. Return ONLY the fixed author name
                    2. Use proper diacritical marks for names
                    3. Consider common name patterns and languages
                    4. If unsure, use the most likely letter
                    5. Maintain the format: "Lastname, Firstname"
                    """
            response = await self.llm.apredict(prompt)
            fixed_name = response.strip()
            return fixed_name if fixed_name else author
        except Exception as e:
            logger.error(f"Error fixing author name: {e}")
            return author

    async def format_bibtex_author_names(self, text: str) -> str:
        try:
            bib_database = bibtexparser.loads(text)
            for entry in bib_database.entries:
                if 'author' in entry:
                    authors = entry['author'].split(' and ')
                    cleaned_authors = []
                    for author in authors:
                        fixed_author = await self.fix_author_name(author)
                        cleaned_authors.append(fixed_author)
                    entry['author'] = ' and '.join(cleaned_authors)
            writer = BibTexWriter()
            writer.indent = '    '
            writer.comma_first = False
            return writer.write(bib_database).strip()
        except Exception as e:
            logger.error(f"Error cleaning BibTeX special characters: {e}")
            return text

    async def search_crossref(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict]:
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
                                    headers={
                                        'Accept': 'application/x-bibtex',
                                        'User-Agent': 'Mozilla/5.0 (compatible; CitationBot/1.0; mailto:example@domain.com)'
                                    },
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

                                    writer = BibTexWriter()
                                    writer.indent = '    '
                                    writer.comma_first = False
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

    def _generate_unique_bibtex_key(self, entry: Dict, existing_keys: set) -> str:
        entry_type = entry.get('ENTRYTYPE', '').lower()
        author_field = entry.get('author', '')
        year = entry.get('year', '')
        authors = [a.strip() for a in author_field.split(' and ')]
        first_author_last_name = authors[0].split(',')[0] if authors else 'unknown'

        if entry_type == 'inbook':
            booktitle = entry.get('booktitle', '')
            title_word = re.sub(r'\W+', '', booktitle.split()[0]) if booktitle else 'untitled'
        else:
            title = entry.get('title', '')
            title_word = re.sub(r'\W+', '', title.split()[0]) if title else 'untitled'

        base_key = f"{first_author_last_name}{year}{title_word}"
        key = base_key
        index = 1
        while key in existing_keys:
            key = f"{base_key}{index}"
            index += 1
        return key

    async def process_text(self, text: str, num_queries: int, citations_per_query: int,
                           use_arxiv: bool = True, use_crossref: bool = True) -> tuple[str, str]:
        if not (use_arxiv or use_crossref):
            return "Please select at least one source (ArXiv or CrossRef)", ""

        num_queries = min(max(1, num_queries), self.config.max_queries)
        citations_per_query = min(max(1, citations_per_query), self.config.max_citations_per_query)

        async def generate_queries_tool(input_data: dict):
            return await self.generate_queries(input_data["text"], input_data["num_queries"])

        async def search_papers_tool(input_data: dict):
            queries = input_data["queries"]
            papers = []
            async with self.async_context as session:
                search_tasks = []
                for q in queries:
                    if input_data["use_arxiv"]:
                        search_tasks.append(self.search_arxiv(session, q, input_data["citations_per_query"]))
                    if input_data["use_crossref"]:
                        search_tasks.append(self.search_crossref(session, q, input_data["citations_per_query"]))
                results = await asyncio.gather(*search_tasks, return_exceptions=True)
            for r in results:
                if not isinstance(r, Exception):
                    papers.extend(r)
            # Deduplicate
            unique_papers = []
            seen_keys = set()
            for p in papers:
                if p['bibtex_key'] not in seen_keys:
                    seen_keys.add(p['bibtex_key'])
                    unique_papers.append(p)
            return unique_papers

        async def cite_text_tool(input_data: dict):
            try:
                citation_input = {
                    "text": input_data["text"],
                    "papers": json.dumps(input_data["papers"], indent=2)
                }
                prompt = self.citation_prompt.format(**citation_input)
                response = await self.llm.apredict(prompt)
                cited_text = response.strip()

                # Aggregate BibTeX entries
                bib_database = BibDatabase()
                for p in input_data["papers"]:
                    if 'bibtex_entry' in p:
                        bib_db = bibtexparser.loads(p['bibtex_entry'])
                        if bib_db.entries:
                            bib_database.entries.append(bib_db.entries[0])
                        else:
                            logger.warning(f"Empty BibTeX entry for key: {p['bibtex_key']}")
                writer = BibTexWriter()
                writer.indent = '    '
                writer.comma_first = False
                bibtex_entries = writer.write(bib_database).strip()
                return cited_text, bibtex_entries
            except Exception as e:
                logger.error(f"Error inserting citations: {e}")
                return input_data["text"], ""

        async def agent_run(input_data: dict):
            queries = await generate_queries_tool(input_data)
            papers = await search_papers_tool({
                "queries": queries,
                "citations_per_query": input_data["citations_per_query"],
                "use_arxiv": input_data["use_arxiv"],
                "use_crossref": input_data["use_crossref"]
            })
            if not papers:
                return input_data["text"], ""
            cited_text, final_bibtex = await cite_text_tool({
                "text": input_data["text"],
                "papers": papers
            })
            return cited_text, final_bibtex

        final_text, final_bibtex = await agent_run({
            "text": text,
            "num_queries": num_queries,
            "citations_per_query": citations_per_query,
            "use_arxiv": use_arxiv,
            "use_crossref": use_crossref
        })
        return final_text, final_bibtex

def create_gradio_interface() -> gr.Interface:
    async def process(api_key: str, text: str, num_queries: int, citations_per_query: int,
                     use_arxiv: bool, use_crossref: bool) -> tuple[str, str]:
        if not api_key.strip():
            return "Please enter your Gemini API Key.", ""
        if not text.strip():
            return "Please enter text to process", ""
        try:
            config = Config(
                gemini_api_key=api_key
            )
            citation_gen = CitationGenerator(config)
            return await citation_gen.process_text(
                text, num_queries, citations_per_query,
                use_arxiv=use_arxiv, use_crossref=use_crossref
            )
        except ValueError as e:
            return f"Input validation error: {str(e)}", ""
        except Exception as e:
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

        .container, .header, .input-group, .controls-row {
            padding: 0.75rem;
        }

        .container {
            max-width: 100%;
            background: var(--bg);
        }

        .header {
            text-align: center;
            margin-bottom: 1rem;
            background: var(--bg);
            border-bottom: 1px solid var(--border);
        }

        .header h1 {
            font-size: 1.5rem;
            color: var(--primary);
            font-weight: 500;
            margin-bottom: 0.25rem;
        }

        .header p, label span {
            font-size: 0.9rem;
            color: var(--text);
        }

        .input-group {
            border-radius: 4px;
            border: 1px solid var(--border);
            margin-bottom: 0.75rem;
        }

        .controls-row {
            display: flex !important;
            gap: 0.75rem;
            margin-top: 0.5rem;
        }

        .source-controls {
            display: flex;
            gap: 0.75rem;
            margin-top: 0.5rem;
        }
        
        .checkbox-group {
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        input[type="number"], textarea {
            border: 1px solid var(--border);
            border-radius: 4px;
            padding: 0.5rem;
            background: var(--control-bg);
            color: var(--text);
            font-size: 0.95rem;
        }

        .generate-btn {
            background: var(--primary);
            color: white;
            padding: 0.5rem 1.5rem;
            border-radius: 4px;
            border: none;
            font-size: 0.9rem;
            transition: background 0.2s;
            width: 100%;
        }

        .generate-btn:hover {
            background: var(--primary-hover);
        }

        .output-container {
            display: flex;
            gap: 0.75rem;
        }
    """

    with gr.Blocks(css=css, theme=gr.themes.Default()) as demo:
        gr.HTML("""<div class="header">
                <h1>📚 AutoCitation</h1>
                <p>Insert citations into your academic text</p>
                </div>""")

        with gr.Group(elem_classes="input-group"):
            api_key = gr.Textbox(
                label="Gemini API Key",
                placeholder="Enter your Gemini API key...",
                type="password",
                interactive=True
            )
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
                        maximum=Config.max_queries,
                        step=1
                    )
                with gr.Column(scale=1):
                    citations_per_query = gr.Number(
                        label="Citations per Query",
                        value=1,
                        minimum=1,
                        maximum=Config.max_citations_per_query,
                        step=1
                    )
            
            with gr.Row(elem_classes="source-controls"):
                with gr.Column(scale=1):
                    use_arxiv = gr.Checkbox(
                        label="Search ArXiv",
                        value=True,
                        elem_classes="checkbox-group"
                    )
                with gr.Column(scale=1):
                    use_crossref = gr.Checkbox(
                        label="Search CrossRef (Experimental)",
                        value=True,
                        elem_classes="checkbox-group"
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
            inputs=[api_key, input_text, num_queries, citations_per_query, use_arxiv, use_crossref],
            outputs=[cited_text, bibtex]
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
