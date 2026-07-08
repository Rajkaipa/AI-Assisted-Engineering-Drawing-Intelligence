# SCHOTT Engineering Drawing Intelligence

**AI-assisted extraction of engineering drawings into validated, SAP-ready material data.**

A production-oriented prototype demonstrating how Vision AI and deterministic engineering rules
can be combined to safely automate SAP material master creation from customer emails containing
CAD drawings.

Built as a Round 2 case-study for the Junior Digital Innovation Manager role at SCHOTT AG (July 2026).

---

## What it does

The system reads customer emails containing engineering drawings, extracts specifications using
Vision AI, validates them with deterministic engineering rules, and generates SAP-ready structured
output — but only when every validation layer agrees the data is safe to send.

- 📧 Reads customer emails (`.eml`) with PDF drawing attachments
- 🧠 Extracts drawing number, material, dimensions, and cooking-zone geometry using Gemini 2.5 Flash
- ✅ Validates extraction with mandatory field checks, dimensional ranges, geometric plausibility, and cross-source reconciliation
- 🛡️ Detects prompt-injection attempts in emails and PDFs (including hidden white-text attacks)
- 📸 Refuses degraded inputs before wasting AI budget (image sharpness gate)
- ⚖️ Routes every ambiguous case to human review with a specific actionable reason
- 📤 Generates SAP-ready output — **only when every validation layer passes**

## Architecture

```
Customer Email
    ↓
Ingestion (email parsing, PDF identification)
    ↓
Security & Quality Gates  ← deterministic, runs BEFORE AI
    ↓
Vision AI Extraction  ← Gemini 2.5 Flash + Pydantic schema
    ↓
Engineering Validation  ← deterministic, runs AFTER AI
    ↓
Cross-source Reconciliation  ← email claims vs drawing content
    ↓
Business Decision  →  🟢 SAP  |  🟡 Human Review  |  🔴 Customer Action
```

**Design principle:** AI where it adds unique value (visual interpretation of varied CAD layouts).
Deterministic code everywhere else (security, validation, routing, gates).

## Demo scenarios

The prototype ships with 8 verified scenarios covering the failure modes that matter for enterprise SAP integration:

| # | Scenario | What it demonstrates |
|---|---|---|
| 1 | Happy path | Full clean pipeline → GREEN → SAP JSON generated |
| 2 | Prompt injection in email | Deterministic detector catches suspicious instructions in email body |
| 3 | Prompt injection in PDF | Detector catches hidden white-text attack invisible to human eye |
| 4 | Corrupt PDF | Fail-fast rejection with actionable reason |
| 5 | Missing attachment | Business follow-up routing |
| 6 | Email vs drawing conflict | Cross-source validator catches per-field disagreement |
| 7 | Low-completeness drawing | Deterministic quality score overrides LLM's over-confident self-report |
| 8 | Blurred drawing | Image quality gate refuses input before spending AI budget |

## Tech stack

**Current MVP:** Python 3.11 · Streamlit · Gemini 2.5 Flash · PyMuPDF · Pydantic v2 · scipy

**Production roadmap:** Outlook / Microsoft Graph API email ingestion · SAP S/4HANA API · Azure deployment with enterprise SSO

## Running locally

**Prerequisites:** Python 3.11+, a Google AI Studio API key with Gemini access.

```bash
# Clone the repo
git clone https://github.com/Rajkaipa/AI-Assisted-Engineering-Drawing-Intelligence.git
cd AI-Assisted-Engineering-Drawing-Intelligence

# Set up virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\Activate.ps1  # Windows PowerShell

# Install dependencies
pip install -r requirements.txt

# Configure your Gemini API key
cp .env.example .env
# then edit .env and set GEMINI_API_KEY=your-key-here

# Run the Streamlit app
streamlit run app.py
```

The app opens at `http://localhost:8501`. Select any demo scenario from the sidebar to see the pipeline in action.

## Project structure

```
.
├── app.py                          # Streamlit UI (presentation layer)
├── src/
│   └── pipeline.py                 # Orchestrator: extraction, validation, routing
├── notebooks/
│   ├── 01_extraction.ipynb         # Vision AI extraction development
│   ├── 02_security_validation.ipynb # Injection detection + validation rules
│   └── 03_end_to_end.ipynb         # Full pipeline integration
├── data/
│   └── incoming_emails/
│       ├── 01_happy_path.eml       # ... 8 demo scenarios
│       └── attachments/            # Test PDF drawings
├── requirements.txt
├── .env.example
└── README.md
```

## Key design decisions

| Decision | Rationale |
|---|---|
| **Vision AI for extraction** | CAD drawings vary in layout across customers — rule-based parsers cannot generalise |
| **Deterministic validation** | LLMs are systematically overconfident on degraded inputs; validation must be testable and versioned |
| **Cheap gates before AI** | Refuse unsafe or unreadable inputs before spending AI budget |
| **Human-in-the-loop routing** | Enterprise safety requires ambiguous cases to reach a person, not silent auto-processing |
| **Modular architecture** | Middle layers reusable when Streamlit → production UI, or Gemini → in-house model |
| **SAP mapping as separate layer** | Extraction pipeline stays ERP-agnostic; SAP swap without pipeline change |

## Limitations & next phase

This is a 4-day prototype. What was intentionally deferred to a Phase 2 build:

- **Formal evaluation metrics** — precision/recall on a labelled test set of 50-100 real customer drawings. The prototype validates behaviour across designed scenarios; production requires quantified evaluation.
- **Reviewer feedback loop** — every human review decision should become training signal for prompt and rule refinement.
- **Multi-page CAD support** — currently handles single-page top-view drawings. Production requires section views and BOM tables.
- **CAD-native ingestion** — where customers can supply DXF or STEP files, vision extraction is unnecessary. Machine-defined geometry is more reliable.
- **Direct SAP S/4HANA integration** — current output is downloadable JSON. Production connects to SAP material master API with audit logging.
- **Multi-domain specialist agents** — SCHOTT product families (cookware, pharma, architectural glass, optics) each require distinct engineering rules; a classification agent routing to specialist validators is the scaling pattern.





