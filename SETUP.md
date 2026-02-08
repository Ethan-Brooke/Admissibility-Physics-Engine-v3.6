# Setup: admissibility-physics-engine

## Create the repo

```bash
mkdir admissibility-physics-engine
cd admissibility-physics-engine
git init
```

## Add all files (flat structure)

Drag-and-drop or copy all files into the folder. The structure should be:

```
admissibility-physics-engine/
├── Admissibility_Physics_Engine_V3_6.py      # Entry point
├── Admissibility_Physics_Theorems_V3_6.py    # Tiers 0-3
├── Admissibility_Physics_Gravity_V3_6.py     # Tiers 4-5
├── theorem_0_canonical_v4.py                 # T0 witnesses
├── run.py                                    # Convenience wrapper
├── dashboard_data.json                       # Dashboard export
├── Admissibility_Physics_Dashboard_V3_6.jsx  # Status dashboard
├── Admissibility_Physics_Energy_Budget_V3_6.jsx  # Energy budget viz
├── github_workflows_update_dashboard.yml     # CI (move to .github/workflows/)
├── README.md
├── VERSION_3_6.md
├── OBSERVATION_beta_cosmology.md
├── LICENSE
└── .gitignore
```

## Verify it works

```bash
python3 Admissibility_Physics_Engine_V3_6.py           # Should show 49/49 PASS
python3 Admissibility_Physics_Engine_V3_6.py --json     # JSON output
python3 Admissibility_Physics_Engine_V3_6.py --audit-gaps  # Gap report
python3 run.py                                          # Same as above via wrapper
```

No pip installs needed — pure Python 3.10+ stdlib.

## Push to GitHub

```bash
git add -A
git commit -m "v3.6: 49 theorems, 43 proven, 20 predictions, 5 cosmological params within 1σ"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/admissibility-physics-engine.git
git push -u origin main
```

## Optional: GitHub Actions CI

Move the workflow file into place:
```bash
mkdir -p .github/workflows
mv github_workflows_update_dashboard.yml .github/workflows/update_dashboard.yml
git add .github/
git commit -m "Add CI workflow"
git push
```

This auto-exports `dashboard_data.json` on every push.

## Requirements

- Python 3.10+
- No external dependencies
- Runs on Linux, macOS, Windows
