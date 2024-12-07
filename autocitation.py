import asyncio
import json
import os
import urllib.parse
import unicodedata
import html
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import gradio as gr
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI
from zhipuai import ZhipuAI

load_dotenv()

@dataclass
class Config:
    gemini_api_key: str
    zhipu_api_key: str
    max_retries: int = 3
    base_delay: int = 1
    max_queries: int = 20
    max_citations_per_query: int = 10
    arxiv_base_url: str = 'http://export.arxiv.org/api/query?'
    crossref_base_url: str = 'https://api.crossref.org/works'
    default_headers: dict = field(default_factory=lambda: {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    })

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
        except Exception:
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
        except Exception:
            return None

    @staticmethod
    def _format_author_name(author: str) -> str:
        names = author.split()
        if len(names) > 1:
            return f"{names[-1]}, {' '.join(names[:-1])}"
        return author

    @staticmethod
    def _generate_bibtex(key: str, title: str, authors: List[str], arxiv_id: str, year: str) -> str:
        return f"""@article{{{key},
            title={{{title}}},
            author={{{' and '.join(authors)}}},
            journal={{arXiv preprint arXiv:{arxiv_id}}},
            year={{{year}}}
        }}"""

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
        self.zhipu_client = ZhipuAI(api_key=config.zhipu_api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.1,
            google_api_key=config.gemini_api_key,
            streaming=True
        )
        self.citation_chain = self._create_citation_chain()

    def _create_citation_chain(self):
        citation_prompt = PromptTemplate.from_template(
            """Insert citations into the provided text using LaTeX \\cite{{key}} commands.
            
            You must not alter the original wording or structure of the text beyond adding citations.
            You must include all provided references at least once. Place citations at suitable points.
            
            Input text:
            {text}
            
            Available papers (cite each at least once):
            {papers}
            """
        )
        return (
            {"text": RunnablePassthrough(), "papers": RunnablePassthrough()}
            | citation_prompt
            | self.llm
            | StrOutputParser()
        )

    async def generate_queries(self, text: str, num_queries: int) -> List[str]:
        try:
            response = self.zhipu_client.chat.asyncCompletions.create(
                model="glm-4-flash",
                messages=[
                    {"role": "system", "content": "You are a research assistant. Always respond with a valid JSON array of search queries."},
                    {"role": "user", "content": f"""
                    Generate {num_queries} diverse academic search queries based on the given text.
                    The queries should be concise and relevant.

                    Requirements:
                    1. Return ONLY a valid JSON array of strings.
                    2. No additional text or formatting beyond JSON.
                    3. Ensure uniqueness.

                    Text: {text}
                    """}
                ],
                temperature=0.0,
                max_tokens=1024
            )

            task_id = response.id
            task_status = response.task_status
            max_retries = 10
            retry_count = 0

            while task_status == 'PROCESSING' and retry_count < max_retries:
                await asyncio.sleep(1)
                result = self.zhipu_client.chat.asyncCompletions.retrieve_completion_result(id=task_id)
                task_status = result.task_status
                retry_count += 1

                if task_status == 'SUCCESS':
                    content = result.choices[0].message.content.strip()
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

        except Exception:
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
                return self.xml_parser.parse_papers(text_data)
        except Exception:
            return []

    async def fix_corrupted_chars(self, match, context_text: str) -> str:
        char = match.group(0)
        if char != '�':
            return char
            
        # Extract context window around corrupted character
        start = max(0, match.start() - 50)
        end = min(len(context_text), match.end() + 50)
        context = context[start:end]
        
        try:
            prompt = f"""Analyze this text with a corrupted character (�) and determine the most likely correct character:
            
            Text: {context}
            
            Requirements:
            1. Return ONLY the single replacement character
            2. Focus on names and common character patterns
            3. Consider language context (Spanish, etc.)
            4. If uncertain, default to 'a'
            """
            
            response = await self.llm.ainvoke([{
                "role": "user",
                "content": prompt
            }])
            
            suggested_char = response.content.strip()
            if len(suggested_char) == 1:
                return suggested_char
                
        except Exception as e:
            print(f"Error using LLM for character fix: {str(e)}")
        
        # Fallback to basic rules if LLM fails
        if 'González' in context_text:
            return 'á'
        elif 'Cristián' in context_text:
            return 'á'
        else:
            return 'a'

    async def fix_all_corrupted_chars(self, text: str) -> str:
        """
        Process all corrupted characters in a text string asynchronously.
        """
        matches = list(re.finditer(r'�', text))
        if not matches:
            return text
            
        # Store the original string and keep track of offset changes
        result = text
        offset = 0
        
        # Process each match one by one
        for match in matches:
            # Adjust match position for previous replacements
            pos = match.start() + offset
            fixed_char = await self.fix_corrupted_chars(match, text)
            
            # Replace the character and update the offset
            result = result[:pos] + fixed_char + result[pos + 1:]
            offset += len(fixed_char) - 1  # -1 because we're replacing one character
            
        return result

    async def clean_bibtex_special_chars(self, text: str) -> str:
        def process_bibtex_entry(entry):
            if 'author = {' in entry:
                author_start = entry.find('author = {') + len('author = {')
                author_end = entry.find('}', author_start)
                if author_start > -1 and author_end > -1:
                    authors = entry[author_start:author_end]
                    author_list = authors.split(' and ')
                    processed_authors = []
                    for author in author_list:
                        processed_author = author
                        for char, latex_cmd in latex_chars.items():
                            processed_author = processed_author.replace(char, latex_cmd)
                        processed_authors.append(processed_author)
                    
                    new_authors = ' and '.join(processed_authors)
                    entry = entry[:author_start] + new_authors + entry[author_end:]
            return entry

        latex_chars = {
            'á': '{\\\'a}', 'é': '{\\\'e}', 'í': '{\\\'i}', 'ó': '{\\\'o}', 'ú': '{\\\'u}',
            'Á': '{\\\'A}', 'É': '{\\\'E}', 'Í': '{\\\'I}', 'Ó': '{\\\'O}', 'Ú': '{\\\'U}',
            'ñ': '{\\~n}', 'Ñ': '{\\~N}',
            'ü': '{\\"u}', 'Ü': '{\\"U}',
            'ï': '{\\"i}', 'Ï': '{\\"I}'
        }
        
        entries = text.split('@')[1:]
        processed_entries = []
        for entry in entries:
            if entry.strip():
                processed_entry = process_bibtex_entry('@' + entry)
                processed_entries.append(processed_entry)
        
        result = ''.join(processed_entries)
        
        # Handle corrupted characters using LLM
        result = await self.fix_all_corrupted_chars(result)
        
        # Convert remaining special characters
        result = ''.join(c if c.isascii() or c.isspace() or c == '-' else 
                      latex_chars.get(c, c) if c in latex_chars else 
                      c if c.isspace() else c 
                      for c in result)
        
        return result.strip()

    async def search_crossref(self, session: aiohttp.ClientSession, query: str, max_results: int) -> List[Dict]:
        try:
            cleaned_query = query.replace("'", "").replace('"', "")
            if ' ' in cleaned_query:
                cleaned_query = f'"{cleaned_query}"'
            encoded_query = urllib.parse.quote(cleaned_query)
            
            params = {
                'query.bibliographic': encoded_query,
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
                            await asyncio.sleep(delay)
                            continue
                        
                        response.raise_for_status()
                        search_data = await response.json()
                        items = search_data.get('message', {}).get('items', [])
                        
                        if not items:
                            return []
                        
                        papers = []
                        for item in items:
                            doi = item.get('DOI')
                            if not doi:
                                continue
                            
                            try:
                                bibtex_url = f"https://doi.org/{doi}"
                                async with session.get(
                                    bibtex_url,
                                    headers={
                                        'Accept': 'text/bibliography; style=bibtex',
                                        'User-Agent': 'Mozilla/5.0 (compatible; CitationBot/1.0; mailto:example@domain.com)'
                                    },
                                    timeout=30
                                ) as bibtex_response:
                                    if bibtex_response.status != 200:
                                        continue

                                    bibtex_bytes = await bibtex_response.read()
                                    try:
                                        bibtex_text = bibtex_bytes.decode('utf-8')
                                    except UnicodeDecodeError:
                                        try:
                                            bibtex_text = bibtex_bytes.decode('latin1')
                                        except UnicodeDecodeError:
                                            bibtex_text = bibtex_bytes.decode('utf-8', errors='replace')

                                    if not bibtex_text.strip():
                                        continue

                                    bibtex_text = urllib.parse.unquote(bibtex_text)
                                    bibtex_text = html.unescape(bibtex_text)
                                    bibtex_text = unicodedata.normalize('NFKC', bibtex_text)
                                    bibtex_text = await self.clean_bibtex_special_chars(bibtex_text)
                                    bibtex_text = bibtex_text.strip()
                                    bibtex_text = re.sub(r'\s+', ' ', bibtex_text)
                                    bibtex_text = bibtex_text.replace(' @', '@')
                                    
                                    match = re.match(r'@(\w+)\s*{\s*([^,\s]+)\s*,', bibtex_text)
                                    if match:
                                        entry_type, key = match.groups()
                                        key = re.sub(r'[^\w-]', '', key)
                                        
                                        bibtex_text = re.sub(r'@(\w+)\s*{\s*([^,]+),', f'@{entry_type}{{{key},', bibtex_text)
                                        bibtex_text = re.sub(r',\s+([a-zA-Z]+)\s*=', r',\n    \1 = ', bibtex_text)
                                        bibtex_text = re.sub(r'}\s*$', '\n}', bibtex_text)
                                        
                                        papers.append({
                                            'bibtex_key': key,
                                            'bibtex_entry': bibtex_text
                                        })
                            except Exception:
                                continue
                        
                        return papers
                        
                except aiohttp.ClientError as e:
                    if attempt == self.config.max_retries - 1:
                        raise
                    delay = self.config.base_delay * (2 ** attempt)
                    await asyncio.sleep(delay)
                    
        except Exception:
            return []

    async def process_text(self, text: str, num_queries: int, citations_per_query: int,
                          use_arxiv: bool = True, use_crossref: bool = True) -> tuple[str, str]:
        if not (use_arxiv or use_crossref):
            return "Please select at least one source (ArXiv or CrossRef)", ""

        num_queries = min(max(1, num_queries), self.config.max_queries)
        citations_per_query = min(max(1, citations_per_query), self.config.max_citations_per_query)

        queries = await self.generate_queries(text, num_queries)
        if not queries:
            return text, ""

        async with self.async_context as session:
            search_tasks = []
            for query in queries:
                if use_arxiv:
                    search_tasks.append(self.search_arxiv(session, query, citations_per_query))
                if use_crossref:
                    search_tasks.append(self.search_crossref(session, query, citations_per_query))
            
            results = await asyncio.gather(*search_tasks, return_exceptions=True)

        papers = []
        for r in results:
            if not isinstance(r, Exception):
                papers.extend(r)

        unique_papers = []
        seen_keys = set()
        for p in papers:
            if p['bibtex_key'] not in seen_keys:
                seen_keys.add(p['bibtex_key'])
                unique_papers.append(p)
        papers = unique_papers

        if not papers:
            return text, ""

        try:
            cited_text = await self.citation_chain.ainvoke({
                "text": text,
                "papers": json.dumps(papers, indent=2)
            })
            bibtex_entries = "\n\n".join(p['bibtex_entry'] for p in papers if 'bibtex_entry' in p)
            return cited_text, bibtex_entries
        except Exception:
            return text, ""

def create_gradio_interface(config: Config) -> gr.Interface:
    citation_gen = CitationGenerator(config)

    async def process(text: str, num_queries: int, citations_per_query: int,
                     use_arxiv: bool, use_crossref: bool) -> tuple[str, str]:
        if not text.strip():
            return "Please enter text to process", ""
        try:
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
            inputs=[input_text, num_queries, citations_per_query, use_arxiv, use_crossref],
            outputs=[cited_text, bibtex]
        )

    return demo

if __name__ == "__main__":
    config = Config(
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        zhipu_api_key=os.getenv("ZHIPU_API_KEY")
    )
    if not config.gemini_api_key or not config.zhipu_api_key:
        raise EnvironmentError("GEMINI_API_KEY or ZHIPU_API_KEY not set.")

    demo = create_gradio_interface(config)
    try:
        demo.launch(server_port=7860, share=False)
    except KeyboardInterrupt:
        print("\nShutting down server...")
    except Exception as e:
        print(f"Error starting server: {str(e)}")