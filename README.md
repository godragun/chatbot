
# PDF Document Agent - Setup Instructions

This guide will help you set up and run the PDF Document Agent application.

## Prerequisites

Before starting, ensure you have the following:

1. Python 3.8+ installed
2. [OLLAMA](https://ollama.ai/) installed and running
3. At least one language model pulled in OLLAMA (e.g., llama3, mistral)

## Installation Steps

### 1. Create a Virtual Environment

```bash
# Create a virtual environment
python -m venv pdf_agent_env

# Activate the virtual environment
# On Windows
pdf_agent_env\Scripts\activate
# On macOS/Linux
source pdf_agent_env/bin/activate
```

### 2. Clone or Create the Project

Create a new directory for your project and save the provided Python script as `app.py`.

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Ensure OLLAMA is Running

Make sure OLLAMA is running with at least one model available:

```bash
# Check available models
ollama list

# If needed, pull a model
ollama pull llama3
```

### 5. Run the Application

```bash
streamlit run app.py
```

The application should open in your default web browser at http://localhost:8501.

## Using the Application

1. **Upload Page**: Select and process your PDF document
2. **Summary Page**: View an AI-generated summary of the document
3. **Questions Page**: Explore AI-generated questions and answers
4. **Multiple Choice Page**: Test your knowledge with MCQs based on the document
5. **Chat Page**: Have a conversation with your document

## Troubleshooting

- **Error connecting to OLLAMA**: Ensure OLLAMA is running and accessible
- **Memory issues with large PDFs**: Try processing smaller documents or increase your system's available memory
- **Slow processing**: Complex documents may take longer to process, especially with larger models

## Advanced Configuration

To customize the application further:

- Edit the `app.py` file to modify UI elements or processing logic
- Change the default model in the application settings
- Adjust chunk sizes for better question answering performance
