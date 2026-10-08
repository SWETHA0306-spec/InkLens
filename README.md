# InkLens — Extreme Handwriting Digitizer

**Track:** HNX26EPS04 · **Positioning:** uncertainty-first digitization for genuinely bad handwriting.

## What is included

InkLens is a jury-ready, self-contained web demo built around an explainable pipeline:

1. **Image quality analysis** — contrast, ink density, and edge energy.
2. **Recognition options** — an opt-in vision API for handwriting or local two-pass Tesseract OCR.
3. **Uncertainty policy** — low-consensus regions become `[REVIEW_REQUIRED]`; no hard-coded text is substituted for actual recognition.
5. **Human-in-the-loop review** — edit the transcription and save feedback to `feedback/corrections.jsonl`.

Recognition quality depends on the configured engine and the input. Model-reported confidence is only an estimate, not a calibrated probability; verify important output.

## Run on Windows

```powershell
cd "C:\path\to\extreme-digitizer"
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python generate_data.py
uvicorn app:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000**.

If PowerShell blocks environment activation, run `Set-ExecutionPolicy -Scope Process Bypass` in that terminal, then activate again.

## Recognition backends

The pipeline uses the first configured backend:

1. **Vision API (opt-in):** recommended for difficult handwriting, mixed scripts,
   and page layout. It uses an OpenAI-compatible chat-completions API supporting
   image input and JSON output. Set these environment variables before starting:

   ```powershell
   $env:INKLENS_VLM_API_KEY = "your-api-key"
   $env:INKLENS_VLM_MODEL = "gpt-4o-mini"
   # Optional for another compatible provider:
   # $env:INKLENS_VLM_BASE_URL = "https://provider.example/v1"
   uvicorn app:app --host 0.0.0.0 --port 8000
   ```

   **Privacy:** configuring this backend sends uploaded images to the selected
   provider. Without `INKLENS_VLM_API_KEY`, the app makes no remote OCR requests.

2. **Local Tesseract:** install the Tesseract OCR application separately, then
   install the Python dependencies above. Ensure `tesseract.exe` is on `PATH`, or
   set `TESSERACT_CMD` to its full path. The local backend compares two OCR passes
   and lowers confidence when their output disagrees. Tesseract is primarily a
   printed-text engine and may perform poorly on difficult handwriting.

3. **No backend:** the app reports that OCR is not configured and flags the image
   for review instead of returning invented text.

The UI shows the active engine and whether recognition is local or remote.

## Test the backend

```bash
curl http://localhost:8000/api/health
python -m unittest discover -s tests
python evaluate.py
```

The benchmark uses the configured real OCR backend and reports an actionable
setup message if no backend is available.

## Judge demo script

1. Click **Clinic note / mixed layout**.
2. Click **Run digitizer stack**.
3. Point out the **Evidence ledger**: region role, consensus score, reason codes, and review state.
4. Show that the crossed-out dosage is suppressed and the margin note is retained separately.
5. Edit the transcription, click **Save reviewed corrections**, and show the feedback buffer.
6. Explain the research contribution: the system is optimized to be **correctly uncertain**, not confidently wrong.

## API

- `GET /api/health` — liveness check
- `GET /api/samples` — seeded sample catalog
- `POST /api/digitize` — multipart image upload
- `POST /api/feedback` — append a human correction

## Project map

- `app.py` — FastAPI service and endpoints
- `pipeline.py` — quality analysis, OCR adapters, layout, and confidence policy
- `index.html` — professional review dashboard
- `generate_data.py` — reproducible hard samples
- `evaluate.py` — seeded CER/uncertainty benchmark for the configured engine
- `tests/` — offline recognition-policy tests
- `data/` — demo images and metadata
- `feedback/` — local review buffer

## Research extension path

For the full competition version, keep the same API and UI contracts while replacing the optional adapter with: (a) a line detector trained on crossed-out/margin annotations, (b) TrOCR logits for token entropy, (c) a VLM verifier for layout and mixed script, and (d) a domain LM constrained to never rewrite `[REVIEW_REQUIRED]` tokens without human approval. The ablation should compare OCR-only, OCR+layout, and the complete uncertainty-aware stack on a held-out hard set.
