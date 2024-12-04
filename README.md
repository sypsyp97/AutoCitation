# 📚 AutoCitation

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

> AI-powered academic citation generator using arXiv and Gemini

AutoCitation automatically analyzes your text content and generates relevant academic citations from arXiv using Google's `gemini-1.5-flash` model.

![AutoCitation Demo](example.gif)

## ✨ Features

- 🤖 Powered by Google's Gemini 1.5 Flash model
- 📄 Automatic paper discovery on arXiv
- 📝 BibTeX formatted citations
- 🌐 Easy-to-use web interface
- ⚡ Real-time citation generation

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Google Gemini API key (free)

### Installation

1. Clone the repository
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Create a `.env` file in the project root and add your Gemini API key:

    ```bash
    GEMINI_API_KEY=your_api_key_here
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
