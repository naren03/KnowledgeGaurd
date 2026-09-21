# KnowledgeGaurd

Simple PDF RAG example using LangGraph for the workflow and Groq for the chat model.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
```

Add your Groq API key to `.env`:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## Ask The PDF A Question

The default PDF is:

```text
docs/Lumetra_HR_Leave_Vacation_Health_Wellbeing_Policy.pdf
```

Run the app:

```powershell
python rag_app.py
```

Then type a question like:

```text
How many vacation days do employees get?
```
