# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MedGemma X-ray Labeling System: automated medical X-ray image annotation using the MedGemma-1.5-4B-IT multimodal LLM, served via a remote OpenAI-compatible API. The system identifies body parts, laterality (left/right/bilateral), and projection views for X-ray images, then provides a Flask web UI for human review of low-confidence predictions.

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Full pipeline (labeling + review server)
python run_pipeline.py --workers 4

# Labeling only
python LLM_lable.py --workers 4
python LLM_lable.py --workers 4 --limit 10   # test with limited samples

# Review web server only (after labeling is done)
python run_pipeline.py --review-only --port 5000
python web/app.py --port 5000

# Test API connectivity
python medgamma_test.py

# Fix breast labels (wrong generic views -> mammography-specific views)
python fix_breast_labels.py --dry-run        # preview only
python fix_breast_labels.py --limit 10       # test run
```

## Architecture

The pipeline is a 3-stage sequential process per sample:

1. **Task A** (body part subdivision): For `脊柱` (spine) and `下肢` (lower limb) with >2 images, calls LLM to identify sub-parts
2. **Task B** (laterality): Determines left/right/bilateral from Excel metadata or LLM
3. **Task C** (projection view): Assigns radiographic projection using one of four routing branches:
   - **Branch 0 (breast fast-path)**: Filename pattern matching (`_L_CC`, `_R_MLO`, etc.) bypasses LLM entirely with confidence 0.95
   - **Branch A (fast-path)**: Single image + single Excel label -> direct assignment
   - **Branch B (global matching)**: Multiple images/labels -> greedy matching via `projection_matcher.py`
   - **Branch C (free classification)**: No Excel labels -> LLM-only classification

### Key Modules

- **`LLM_lable.py`** - Main processing engine. Reads `selected_samples.xlsx`, calls MedGemma API for each sample using `ThreadPoolExecutor`, writes results to `processed_labels_v3.xlsx`. Supports checkpoint-based resume.
- **`projection_matcher.py`** - Projection matching algorithm. Asks LLM to score each image against standard views (0-10), parses into a score matrix, then uses greedy assignment with confidence scoring. Uses `STANDARD_VIEWS` (frontal/lateral/oblique/axial) for general X-rays and `BREAST_VIEWS` (CC/MLO/spot) for mammography.
- **`web/app.py`** - Flask review server. Loads labeled Excel, serves images via `/api/image/<path>`, provides CRUD endpoints for modifying labels. Writes modifications back to the Excel file with thread-safe locking (`EXCEL_LOCK`).
- **`fix_breast_labels.py`** - Post-hoc repair script for breast samples that were incorrectly labeled with generic projection views.

### Data Flow

```
selected_samples.xlsx + data/{影像号}/*.png + part_exam_orientation.json
    -> LLM_lable.py (multi-threaded, calls MedGemma API)
    -> processed_labels_v3.xlsx
    -> web/app.py (Flask review UI)
    -> review_modifications.json (exported changes)
```

### API Integration

All LLM calls go through a single OpenAI-compatible endpoint configured at the top of `LLM_lable.py`:
- `BASE_URL` / `API_URL` / `MODEL_ID` constants
- Images are base64-encoded and sent as `image_url` content blocks
- Retry logic: 3 attempts with 5s delay between retries, 1.5s delay between normal requests
- 600s timeout per request

### Configuration

- `part_exam_orientation.json` maps Chinese body part names to allowed projection views and orientations
- `CN_EN_MAP` in `LLM_lable.py` is the comprehensive Chinese-to-English mapping dictionary for body parts and projection views
- Confidence threshold default: 0.6 (below this triggers `needs_review=True`)

## Language

The codebase uses Chinese for: log messages, variable comments, body part names in data, and user-facing strings. English is used for: code identifiers, API prompts to MedGemma, and standardized label values (frontal/lateral/etc.). Maintain this bilingual convention.

## Security

Per `.github/instructions/snyk_rules.instructions.md`: run Snyk code scan for new first-party code in supported languages and fix any issues found before committing.
