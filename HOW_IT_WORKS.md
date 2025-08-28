# HOW_IT_WORKS.md

This document explains the inner workings of the Meal Planner chatbot built with **LangGraph** and **LangChain**.

---

## 1) Core Components

- **LangGraph** — Manages the *state machine* (conversation flow).
- **LangChain** — Builds prompts and calls the OpenAI LLM.
- **Rich** — Nicer tables and colored console output.
- **python-dotenv** — Loads API keys and settings from `.env`.

---

## 2) Conversation Flow (State Machine)

The chatbot follows a deterministic sequence of states:

1. **ask** → prints the current question and waits for input.  
2. **store** → validates & saves the answer, advances to next question (or re-asks on invalid).  
3. **plan** → after all questions, calls the OpenAI model and prints a multi-day meal plan + shopping list.

Transitions:

- `ask → store`  
- `store → ask` (if more questions remain)  
- `store → plan` (when all questions are answered)  
- `plan → END`

ASCII diagram:

~~~text
[ask] --(capture input)--> [store] --(more questions?)--> [ask]
                                    \--(none left)-------> [plan] --> [END]
~~~

---

## 3) Data Handling

- **Questions**: Defined in `QUESTIONS` inside `main.py`.

    ~~~python
    QUESTIONS = {
      "name": "What's your name?",
      "age": "How old are you? (in years)",
      ...
    }
    ~~~

- **Answers**: Stored in a dictionary (`state["answers"]`) keyed by the question name.

    ~~~json
    {
      "name": "Alex",
      "age": "30",
      "liked_foods": "chicken, pasta",
      "disliked_foods": "broccoli"
    }
    ~~~

- **State Machine**: The state tracks:
  - `idx` — which question you’re on  
  - `answers` — collected inputs  
  - `stage` — `"ask"`, `"store"`, or `"plan"`  
  - `last_input` — most recent input

---

## 4) LLM Integration

- **System Prompt** — tells the model to act as a nutrition-aware meal planner (budget-friendly, respect allergies/dislikes, concise justifications).
- **Human Prompt** — injects the collected answers (age, height, likes/dislikes, target meals, duration, etc.) in a structured way.
- **Model Call** — uses `ChatOpenAI` with your `OPENAI_API_KEY`.

**Defaults**
- **Model**: `gpt-4o-mini`  
- **Temperature**: `0.4` (balanced creativity)

---

## 5) Customization

- **Change questions** → edit the `QUESTIONS` dictionary in `main.py`.  
- **Swap the model** → update `get_llm()` to use another LangChain chat model (e.g., Anthropic, Ollama).  
- **Tweak the output** → adjust the system/human prompts in `node_plan` (tone, detail, grouping, etc.).
- **Validation hints** → customize the hint messages in node_store for clearer re-prompts.

---

## 6) Why LangGraph?

LangGraph gives explicit control of the conversation via a **graph of states and transitions**, instead of ad-hoc `if/else` logic.  
It’s easier to extend, debug, and reason about where the conversation is at any time.

- Entry point: `ask`  
- Edges: `ask → store`, `store → ask`, `store → plan`, `plan → END`  
- Deterministic order: follows the keys of `QUESTIONS`.

---

## 7) Example Flow

~~~text
Welcome to the Meal Planner!
What's your name?
> Alex

How old are you? (in years)
> 30

...

How long should the meal plan be? (number of days)
> 5

[Meal plan generated here]

Collected Answers:
name → Alex
age  → 30
...
~~~

---

## 8) Next Steps

- Add **persistence** (save answers/plan to a file or database).  
- Wrap in a **web UI** (FastAPI, Streamlit, Gradio).  
- Expand the **nutrition model** (calories, macros, portion sizes).  

---

## Summary

The chatbot cycles through a LangGraph-driven **state machine** that asks questions, saves responses, and then uses **LangChain + OpenAI** to generate a personalized, multi-day meal plan with shopping lists.
