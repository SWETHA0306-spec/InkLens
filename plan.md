# InkLens implementation plan

## Product outcome
A polished local web demo that digitizes difficult handwriting with an inspectable confidence ledger, explicit uncertainty flags, crossed-out suppression, margin-note separation, and a human correction buffer.

## Architecture
- `app.py`: FastAPI API and static file host.
- `pipeline.py`: dependency-light deterministic image features and layout region model; optional Tesseract adapter behind a feature flag.
- `index.html`: dark research-lab dashboard designed for a short jury walkthrough.
- `generate_data.py` / `data`: reproducible seeded hard samples.
- `evaluate.py`: repeatable CER and uncertainty-policy check.

## Design system
- **Movement:** editorial research lab / instrument panel.
- **Principles:** evidence over decoration; high-contrast hierarchy; calm warnings; visible provenance.
- **Color philosophy:** deep navy creates focus, cyan communicates verified signal, amber communicates review—not failure, and rose reserves emphasis for suppressed content.
- **Layout:** asymmetric two-column workspace: evidence intake on the left, transcription and ledger on the right.
- **Signature elements:** monospaced evidence labels, rounded instrument cards, cyan “InkLens” aperture mark.
- **Interaction:** every action reveals why the system decided; no silent transformations.
- **Typography:** Manrope for interface hierarchy, DM Mono for machine-readable evidence.

## Scope assumptions
The included offline adapter prioritizes repeatable judging and zero model downloads. It does not claim production accuracy. Real OCR can be enabled without changing the UI/API contract, and the README documents the adapter path for TrOCR/VLM integration.
