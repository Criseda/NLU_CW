# Authorship Verification: Academic Poster

This directory contains the source code and assets for the NLU Authorship Verification poster (Track C). The design follows a professional academic layout using the `beamerposter` framework and a custom minimalist grid layout. It includes programmatic vector diagrams (TikZ) and high-fidelity performance plots.

## Components

- `poster.tex`: The main LaTeX source file using the `beamerposter` class structure.
- `beamerposter.sty`: Configuration file mapping LaTeX font sizes effectively for an A0 canvas scale.
- `beamerthemezurichposter.sty`: The core UI theme dictating the purple header styling and margin allocations.
- `generate_plots.py`: A Python script for generating research-grade figures (`performance_comparison.pdf`, `confusion_matrix.pdf`, `feature_groups.pdf`).
- `Makefile`: Automation for the build process (plot generation and PDF compilation).
- `UOM_logo.png`: Official University of Manchester branding.

## Rendering Pipeline & Dependencies

To render the poster from source, you need both a LaTeX and a Python environment.

### 1. Python Environment Dependencies (for generating diagrams)
The `Makefile` will first generate the performance and feature group charts natively from python. Ensure you have the following installed:
- Python 3.8+
- Packages: `matplotlib`, `seaborn`, `pandas`, `numpy`

### 2. LaTeX Dependencies (for compiling the layout)
You will need a functional TeX distribution installed on your system.
- **Mac Users**: Install MacTeX via Homebrew: `brew install --cask mactex-no-gui`.
- **Linux Users**: Install TeX Live: `sudo apt install texlive-full`.
- **Windows Users**: Install MiKTeX or TeX Live.

## Build Instructions

Using the `make` utility is the simplest way to execute the full pipeline locally. 

**Compile final PDF from start-to-finish:**
```bash
# This automatically executes the python plot generators, then compiles the LaTeX twice to resolve internal references.
make
```
**Compile only the LaTeX (if charts are already generated):**
```bash
make pdf
```
**Regenerate only the charts:**
```bash
make plots
```
**Clean the directory of auxiliary build files:**
```bash
make clean
```

The final A0 presentation file will be exported as `poster.pdf`.
