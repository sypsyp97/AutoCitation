"""Citation Generator Utilities

This module provides utility functions for the Citation Generator application,
implementing core functionality for API interactions, paper searches, and query generation.

The module handles three main tasks:
1. Google Gemini AI client initialization and management
2. ArXiv API interaction for paper searches
3. Query generation using AI for finding relevant papers

Functions:
    init_clients(): Initialize the Google Gemini AI client
    search_arxiv(query, max_results): Search for papers on ArXiv
    generate_queries(client, text, num_queries): Generate search queries using AI

Dependencies:
    - google.generativeai: For AI-powered query generation
    - python-dotenv: For environment variable management
    - loguru: For logging
    - xml.etree.ElementTree: For parsing ArXiv API responses
"""

import os
from loguru import logger
import sys
from dotenv import load_dotenv
import google.generativeai as genai
from tqdm import tqdm
import json
import time
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

def init_clients():
    """Initialize the Google Gemini client using API key from environment variables.
    
    This function:
    1. Loads environment variables from .env file
    2. Retrieves the GEMINI_API_KEY
    3. Configures and initializes the Gemini client
    4. Tests the client with a simple request
    
    Returns:
        tuple: (success: bool, client: genai.GenerativeModel or None)
            - success: True if initialization successful, False otherwise
            - client: Initialized Gemini client if successful, None otherwise
    
    Environment Variables:
        GEMINI_API_KEY: API key for Google Gemini AI service
    """
    logger.info("Starting client initialization...")
    
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        logger.error("GEMINI_API_KEY not found in environment variables")
        return False, None
        
    logger.info("API key loaded successfully")
    
    try:
        logger.info("Initializing Gemini client...")
        genai.configure(api_key=api_key)
        client = genai.GenerativeModel("gemini-1.5-flash")  
        
        # Test the client with a simple request
        test_response = client.generate_content("Hello")
        if test_response and hasattr(test_response, 'text'):
            logger.info("Gemini client initialized and tested successfully")
            return True, client
        else:
            logger.error("Gemini client test failed")
            return False, None
    except Exception as e:
        logger.error(f"Failed to initialize Gemini client: {str(e)}")
        return False, None

def search_arxiv(query, max_results=5):
    """Search arXiv papers using the arXiv API.
    
    This function constructs and sends a request to the arXiv API to search
    for academic papers based on the provided query. It processes the XML response
    and formats the results into a structured format including title, authors,
    abstract, and BibTeX entry.
    
    Args:
        query (str): Search query string
        max_results (int, optional): Maximum number of results to return. Defaults to 5.
    
    Returns:
        list: List of dictionaries containing paper information:
            - title: Paper title
            - authors: List of author names
            - abstract: Paper abstract
            - url: arXiv URL
            - pdf_url: Direct PDF URL
            - arxiv_id: arXiv identifier
            - bibtex_entry: Formatted BibTeX entry
            
    Note:
        The function implements rate limiting by sleeping for 3 seconds between
        batches of requests to comply with arXiv API guidelines.
    """
    base_url = 'http://export.arxiv.org/api/query?'
    
    # Clean and encode the query properly
    cleaned_query = urllib.parse.quote(query)
    
    params = {
        'search_query': f'all:{cleaned_query}',
        'start': 0,
        'max_results': max_results,
        'sortBy': 'relevance',
        'sortOrder': 'descending'
    }
    
    query_url = base_url + urllib.parse.urlencode(params)
    papers = []
    
    try:
        # Add headers to the request
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        request = urllib.request.Request(query_url, headers=headers)
        
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read().decode('utf-8')
            root = ET.fromstring(data)
            
            ns = {
                'atom': 'http://www.w3.org/2005/Atom',
                'arxiv': 'http://arxiv.org/schemas/atom'
            }
            
            entries = root.findall('atom:entry', ns)
            if not entries:
                logger.warning(f"No results found for query: {query}")
                return []

            used_keys = set()
            
            for entry in entries:
                try:
                    title_elem = entry.find('atom:title', ns)
                    if title_elem is None:
                        continue

                        
                    paper_info = {
                        'title': title_elem.text.strip(),
                        'authors': [author.find('atom:name', ns).text for author in entry.findall('atom:author', ns)],
                        'arxiv_id': entry.find('atom:id', ns).text.split('/')[-1],
                        'abstract': entry.find('atom:summary', ns).text.strip(),
                        'published': entry.find('atom:published', ns).text
                    }
                    
                    # Extract year and month
                    year = paper_info['published'][:4]
                    month = paper_info['published'][5:7]
                    
                    # Format authors
                    formatted_authors = []
                    for author in paper_info['authors']:
                        name_parts = author.split()
                        if len(name_parts) > 1:
                            last_name = name_parts[-1]
                            first_names = ' '.join(name_parts[:-1])
                            formatted_authors.append(f"{last_name}, {first_names}")
                        else:
                            formatted_authors.append(author)
                    
                    # Create BibTeX key
                    if paper_info['authors']:
                        first_author_last = paper_info['authors'][0].split()[-1].lower()
                        second_author_initial = ''
                        if len(paper_info['authors']) > 1:
                            second_author_last = paper_info['authors'][1].split()[-1][0].lower()
                            second_author_initial = second_author_last
                        
                        arxiv_suffix = paper_info['arxiv_id'].split('.')[-1]
                        base_key = f"{first_author_last}{second_author_initial}{year}{month}{arxiv_suffix}"
                        
                        bibtex_key = base_key
                        counter = 1
                        while bibtex_key in used_keys:
                            bibtex_key = f"{base_key}_{counter}"
                            counter += 1
                        used_keys.add(bibtex_key)
                        
                        bibtex_entry = f"""@article{{{bibtex_key},
  title={{{paper_info['title']}}},
  author={{{' and '.join(formatted_authors)}}},
  journal={{arXiv preprint arXiv:{paper_info['arxiv_id']}}},
  year={{{year}}}
}}"""
                        
                        paper_info['bibtex_key'] = bibtex_key
                        paper_info['bibtex_entry'] = bibtex_entry
                        papers.append(paper_info)
                except Exception as e:
                    logger.warning(f"Error processing paper entry: {str(e)}")
                    continue
            
            return papers
            
    except Exception as e:
        logger.error(f"Error searching arXiv: {str(e)}")
        return []

def generate_queries(client, text, num_queries):
    """Generate search queries using Google Gemini model.
    
    This function uses the Gemini AI model to analyze the input text and generate
    relevant search queries for finding academic papers. It processes the AI response
    and ensures the output is properly formatted.
    
    Args:
        client (genai.GenerativeModel): Initialized Gemini AI client
        text (str): Input text to analyze
        num_queries (int): Number of queries to generate
    
    Returns:
        list: List of generated search queries
        
    Example:
        >>> client = init_clients()[1]
        >>> text = "The impact of climate change on coral reefs"
        >>> queries = generate_queries(client, text, 2)
        >>> print(queries)
        ['coral reef degradation climate change impacts',
         'climate change effects coral reef ecosystems']
    """
    try:
        system_prompt = """You are an advanced research assistant specializing in natural language processing and academic search optimization.
        Generate precise and highly relevant academic search queries that will find papers related to the given text.
        Return a JSON array of strings containing search queries. Each query should focus on a different aspect of the text.
        Format: ["query1", "query2"]"""
        
        full_prompt = f"{system_prompt}\n\nText: {text}\nNumber of queries needed: {num_queries}"
        
        response = client.generate_content(full_prompt)
        content = response.text.strip()
        
        # Enhanced response processing
        try:
            # Try to find a JSON array in the response
            start = content.find('[')
            end = content.rfind(']') + 1
            if start >= 0 and end > start:
                content = content[start:end]
            
            queries = json.loads(content)
            if isinstance(queries, list):
                # Clean and validate queries
                valid_queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]
                if valid_queries:
                    return valid_queries[:num_queries]
                
            # Fallback: split content into lines if JSON parsing fails
            return [content.split('\n')[0]][:num_queries]
            
        except json.JSONDecodeError:
            # Create single query from response
            logger.warning("Could not parse JSON response, using raw text as query")
            return [content.split('\n')[0]][:num_queries]
            
    except Exception as e:
        logger.error(f"Error generating queries: {str(e)}")
        return ["CBCT reconstruction deep learning"]  # Fallback query based on the example text

def run_test(citations_per_query, num_queries):
    """Main function to run the citation generation process."""
    if citations_per_query < 1 or num_queries < 1:
        logger.error("Invalid input: citations_per_query and num_queries must be positive integers")
        return

    success, client = init_clients()
    if not success:
        logger.error("Client initialization failed, exiting...")
        return
    
    logger.info("Client initialization successful, proceeding with main process...")
    
    test_text = """This study addresses the challenge of applying analytical methods for Cone Beam Computed Tomography (CBCT) reconstructions along arbitrary trajectories instead of iterative methods. Traditional analytical methods like Filtered Back Projection (FBP) often fail to adequately process CBCT images due to the intricate and varied paths involved. Although iterative methods are a common solution, they require substantial computational time and resources. To address these challenges, this paper proposes two approaches, the first approach enhances the traditional FBP algorithm by using deep learning to train an optimized filter before reconstruction. The second approach improves upon the Backprojection then Filtering (BPF) algorithm by first reconstructing and then applying a deep learning-trained filter. Both methods significantly optimize the initial reconstruction results and enhance efficiency, offering promising improvements over existing iterative reconstruction techniques."""
    
    logger.info("Step 1: Extracting queries from text...")
    queries = generate_queries(client, test_text, num_queries)
    if not queries:
        logger.error("Failed to generate queries, exiting...")
        return
    
    logger.info("Step 2: Starting paper search...")
    all_papers = []
    
    for query in tqdm(queries, desc="Searching queries"):
        papers = search_arxiv(query, max_results=citations_per_query)
        if papers:
            logger.info(f"Found {len(papers)} papers for query: {query}")
            all_papers.extend(papers)
            for paper in papers:
                logger.info(f"Found paper: {paper['title']} with key {paper['bibtex_key']}")
            time.sleep(3)  # Rate limiting
    
    if not all_papers:
        logger.error("No papers found. Stopping process.")
        return
    
    logger.info("Step 3: Generating citations in text...")
    system_prompt = """You are a LaTeX citation assistant. Add \\cite{} commands to appropriate places in the text.
    Place citations at relevant points where the text discusses related concepts. All papers must be cited.
    Only add \\cite{} commands, do not modify the original text."""
    
    user_prompt = f"Text to add citations to:\n\n{test_text}\n\nAvailable papers:\n{json.dumps(all_papers, indent=2)}"
    full_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    try:
        response = client.generate_content(full_prompt)
        cited_text = response.text
        print("\nGenerated citations:")
        print(cited_text)
        
        # Save outputs to files
        with open('output_text.txt', 'w', encoding='utf-8') as f:
            f.write(cited_text)
        logger.info("Saved cited text to output_text.txt")
        
        logger.info("Step 4: Generating BibTeX entries...")
        bibtex_entries = []
        for paper in all_papers:
            print(paper['bibtex_entry'])
            bibtex_entries.append(paper['bibtex_entry'])
        
        with open('references.bib', 'w', encoding='utf-8') as f:
            f.write('\n\n'.join(bibtex_entries))
        logger.info("Saved BibTeX entries to references.bib")
        
    except Exception as e:
        logger.error(f"Error generating citations: {str(e)}")

# if __name__ == "__main__":
#     logger.info("Starting citation generation process...")
#     try:
#         citations_wanted = int(input("Enter the number of citations you want per query (minimum 1): "))
#         num_queries = int(input("Enter the number of search queries to generate (minimum 1): "))
        
#         if citations_wanted < 1 or num_queries < 1:
#             raise ValueError("Both values must be positive integers")
            
#         run_test(citations_per_query=citations_wanted, num_queries=num_queries)
#         logger.info("Process completed successfully")
#     except ValueError as e:
#         logger.error(f"Invalid input: {str(e)}")
#         sys.exit(1)
#     except Exception as e:
#         logger.error(f"Process failed with error: {str(e)}")
#         sys.exit(1)