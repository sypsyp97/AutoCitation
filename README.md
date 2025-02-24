# 📚 AutoCitation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Hugging Face](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-yellow.svg)](https://huggingface.co/spaces/yipengsun/AutoCitation)
[![cc-by-nc-shield](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
> AI-powered academic citation generator

AutoCitation is an agent that automatically analyzes your text content and generates relevant academic citations from [arXiv](https://arxiv.org/) and [Crossref](https://www.crossref.org/) databases using Google `gemini-2.0-flash`. It then integrates the citations into your text using LaTeX `\cite{}` citation commands and return the cooresponding BibTeX entries.

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

## Usage

Use the following command to run the tool:

```bash
python autocitation.py
```

This will start a Gradio web interface.

## Acknowledgements

AI can be a powerful tool for finding citations, but it is important to verify the results and ensure that the citations are accurate and appropriate. This tool is intended to assist with the citation process, but it is not a substitute for careful research and review of the sources.

## Contributing

Contributions are welcome through pull requests. Feel free to open an issue for bug reports or feature suggestions.

## License

This project is open source and available under the [Creative Commons Attribution-NonCommercial 4.0 International License](LICENSE).
