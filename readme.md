# LLM Suite

LLM Suite allows you compare responses of various local LLMs.

## Features

- Automatically detects installed Ollama models
- Customize temperature and system prompt
- View responses side-by-side or stacked
- Export results as CSV or JSON

## Setup Instructions

### Prerequisites

You'll need:
- Python 3.7+
- Ollama installed and running locally

## Installation

1. Install required Python packages:

```bash
pip install -r requirements.txt
```

2. Make sure you have Ollama installed and running:
   - Visit [Ollama's website](https://ollama.com/) to download and install if needed
   - Run the Ollama app

3. Pull some models (if you haven't already):

```bash
ollama pull llama3.1
ollama pull mistral
# And any other models you want to compare
```

### Running the App

1. Run the following in a terminal of your choice:

```bash
streamlit run main.py
```

2. The app should open in your browser (if not, navigate to the Local URL shown in the terminal)

## Usage Tips

- Click "Refresh Available Models" in the sidebar if you pull new models while the app is running
- Adjust the temperature slider to control randomness in responses
- Use the system prompt to set the context or personality for the models
- Switch between "Side by Side" and "Stacked" tabs to view responses in different layouts
- Use the export options to save your comparisons

## Troubleshooting

- If no models appear, make sure Ollama is running (`ollama serve` in terminal)
- If responses take too long, try using fewer models at once or reduce the complexity of your prompt. This is largely hardware based.
- If you get errors, check that your Ollama API is accessible at http://localhost:11434