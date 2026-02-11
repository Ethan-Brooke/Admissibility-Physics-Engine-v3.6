# Version 3.8 — Crystal Auto-Extraction

**Release date:** 2026-02-11

## What Changed

### Crystal v3: Auto-Extracted Dependencies
The enforcement crystal analysis (enforcement_crystal_v3.py) now auto-extracts all dependencies
by running every `check_*()` function in the theorem bank and reading the `dependencies` field.
This eliminates the hardcoded `DEPENDENCY_MAP` that previously drifted from the actual code.

**Key corrections from v2 crystal:**
| Metric | v2 (stale) | v3 (auto-extracted) |
|--------|-----------|-------------------|
| Nodes | 64 | **61** |
| Edges | 198 | **189** |
| Paths to sin²θ_W | 5,137 | **3,181** |
| Attribution A1/A3/A4 | 40/40/21% | **46.7/42.1/11.1%** |
| Width-1 waists | depths 1, 8, 19 | **depths 8, 17, 19** |

### L_nc Vocabulary Migration
The theorem bank now speaks 3-axiom natively:
- **Before:** L_nc declared `dependencies=['A1', 'A2']` (5-axiom vocabulary)
- **After:** L_nc declares `dependencies=['A1', 'A3']` (3-axiom native)
- A2 and A5 are no longer referenced anywhere in the bank

### Triple-Waisted Topology
The crystal has three width-1 structural bottlenecks:
- **T_gauge** (depth 8): Gauge group selection — all information above is abstract constraint math
- **T9_grav** (depth 17): Einstein field equations — gravitational channel
- **T12E** (depth 19): Baryon fraction — terminal confluence of the full framework

### Dashboard v3.8
- Version header: "3 axioms (A1, A3, A4) + 2 derived lemmas"
- Crystal visualization: waist indicators at depths 8, 17, 19 (was 6, 12)
- Paper 14 description: 61-node, triple-waisted, 3,181 paths (was 56-node, double-waisted, 1,398)
- Audit item A51 added for crystal auto-extraction
- 12 solid results / 7 open issues

### CI/CD
- GitHub Actions workflow moved to `.github/workflows/update_dashboard.yml` (was at repo root)
- `run_dashboard_export.py` generates `dashboard_data.json` from theorem bank + crystal
- Auto-commits on push to main when source files change

## File Changes

| File | Status | Notes |
|------|--------|-------|
| fcf_theorem_bank.py | Updated | L_nc deps=['A1','A3'], cleaned L_irr/L_loc annotations |
| enforcement_crystal_v3.py | New | Auto-extracting crystal analysis |
| run_dashboard_export.py | New | Dashboard data generator |
| index.html | Updated | v3.8 crystal corrections |
| README.md | Updated | 3-axiom structure, crystal metrics |
| VERSION_3_8.md | New | This file |
| .github/workflows/update_dashboard.yml | Moved | Was at repo root (never fired) |

## Theorem Count
- **60 theorems**: 41 [P] + 19 [P_structural]
- **0 failures**, all pass
- **20 predictions** (0 free parameters)
