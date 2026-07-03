# \# AI-Assisted Engineering Drawing Intelligence

# 

# A prototype system for extracting structured, SAP-ready data from customer engineering drawings using multimodal AI, with deterministic validation and prompt-injection defence.

# 

# \## Problem

# 

# Sales engineers receive customer engineering drawings via email as PDFs. Manually interpreting each drawing to extract dimensions, materials, tolerances, and metadata for SAP material-master entry is time-consuming, error-prone, and does not scale with order volume.

# 

# \## Approach

# 

# A hybrid pipeline that uses AI only where ambiguity requires it (visual interpretation of drawings, semantic normalisation of fields), and deterministic code for everything else (PDF loading, validation rules, unit checks, JSON structuring, SAP mapping).

# 

# \## Architecture





\## Key design principles



\- \*\*AI for judgement, deterministic code for rules.\*\* Vision-language models handle drawing interpretation; Python handles validation, ranges, and business logic.

\- \*\*Prompt-injection defence.\*\* Every extracted text element is treated as data, never as instruction. Suspicious content is flagged for human review before the LLM is invoked.

\- \*\*Four-layer validation.\*\* Syntax (Pydantic), business rules (engineering), confidence thresholds, and security (injection, corrupt files).

\- \*\*Provider abstraction.\*\* LLM logic sits behind an interface — Gemini today, Claude / GPT-4 / Azure OpenAI tomorrow.



\## Getting started



```bash

python -m venv .venv

.venv\\Scripts\\activate           # Windows

pip install -r requirements.txt

cp .env.example .env             # then edit .env with your Gemini API key

```



\## Status



Prototype / interview case study. Not production.



\## Author



Raja Sekhar Kaipa — Applied AI Specialist

