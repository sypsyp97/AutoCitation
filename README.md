# 📚 AutoCitation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> AI-powered academic citation generator

AutoCitation automatically analyzes your text content and generates relevant academic citations from [arXiv](https://arxiv.org/) and [CrossRef](https://www.crossref.org/) databases using Google `gemini-1.5-flash`. It then integrates the citations into your text using LaTeX `\cite{}` citation commands and return the cooresponding BibTeX entries.

## 🎉 Demo

![AutoCitation Demo](example.png)

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- [Google Gemini API key](https://ai.google.dev/)

### Installation

1. Clone the repository
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root and add your API keys:

    ```bash
    GEMINI_API_KEY=your_gemini_api_key_here
    ```

## Usage

```bash
python autocitation.py
```

This will start a Gradio web interface.

## Acknowledgements

AI can be a powerful tool for finding citations, but it is important to verify the results and ensure that the citations are accurate and appropriate. This tool is intended to assist with the citation process, but it is not a substitute for careful research and review of the sources.

## License

This project is open source and available under the [GNU General Public License v3.0 (GPLv3) License](LICENSE).
