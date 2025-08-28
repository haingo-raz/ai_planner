# Meal Planner Chatbot (LangGraph + LangChain)

A simple console chatbot that asks dietary questions and then generates a multi-day meal plan using an OpenAI model via LangChain.

## Prerequisites
- Python 3.10+
- OpenAI API key

## Setup


# 1) Create & activate a virtual environment
```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
# .venv\Scripts\Activate.ps1
```

# 2) Install dependencies
```
pip install -r requirements.txt
```

# 3) Set your OpenAI key (same shell you're running in)
```
# macOS/Linux:
export OPENAI_API_KEY=your_key_here
# Windows PowerShell:
# $Env:OPENAI_API_KEY="your_key_here"

# (Optional) Choose a model & temperature
export MODEL_NAME=gpt-4o-mini
export MODEL_TEMPERATURE=0.4
```

## Alternatively, create a .env file in the project root:
```
OPENAI_API_KEY=your_key_here
MODEL_NAME=gpt-4o-mini
MODEL_TEMPERATURE=0.4
```

## Run the chatbot
```
python main.py
```