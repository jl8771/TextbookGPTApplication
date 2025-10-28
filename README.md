# TextbookGPT

This repository contains a Python application that serves an API using Flask. The API enables users to interact with a large language model (LLM) augmented with Retrieval-Augmented Generation (RAG). It is designed to retrieve and process documents, specifically pages of textbooks, to provide context-aware responses.

## Features

- **Flask API**: A lightweight API for interacting with the LLM.
- **RAG Integration**: Augments the LLM with document retrieval for context-based responses.
- **Customizable Data**: The application does not include preloaded data; users can provide their own documents for retrieval.
- **Local Web Interface**: Intended for use with a local web interface for seamless interaction.
- **Testing with Jupyter Notebook**: Small-scale testing can be performed using the `chat.ipynb` notebook.

## Requirements

- Python 3.8 or higher
- Flask
- Dependencies listed in `requirements.txt`

## Usage

1. **Prepare Your Documents**: Add your own textbook pages or other documents for retrieval. Ensure they are formatted correctly for the application.
2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Run the API**: Access the Web Interface: Open the local web interface in your browser to interact with the LLM.
4. **Test with Jupyter Notebook**: Use chat.ipynb for small-scale testing and experimentation.

## Notes
This repository does not include any preloaded data. Users must provide their own documents for retrieval.

The application is designed for local use and is not optimized for large-scale deployment.

## Contributing
Contributions are welcome! Feel free to open issues or submit pull requests.