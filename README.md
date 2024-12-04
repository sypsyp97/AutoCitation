# Citation Generator

An intelligent citation assistant that helps you find relevant academic papers and generate citations in LaTeX for your text using Google's Gemini AI and arXiv integration.
![Citation Generator](example.png)


## Prerequisites

- Python 3.9 or higher
- Google Gemini API key

## Installation

1. Clone this repository
2. Install the required dependencies:
   ```bash
   pip install customtkinter pyperclip google-generativeai python-dotenv loguru tqdm
   ```
3. Create a `.env` file in the root directory and add your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

## Usage

1. Run the application:
   ```bash
   python main.py
   ```
2. Enter your text in the input area
3. Set the desired number of queries and citations
4. Click "process" to process your text
5. Use the copy buttons to copy the cited text or BibTeX entries


## License

This project is open source and available under the MIT License.
