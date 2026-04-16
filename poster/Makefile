# Makefile for NLU Poster

# Path to LaTeX binaries on Mac
TEXPATH=/Library/TeX/texbin/
PDFLATEX=$(TEXPATH)pdflatex
BIBTEX=$(TEXPATH)bibtex

# Target file
MAIN=poster

all: plots pdf

plots:
	python3 generate_plots.py

pdf: $(MAIN).tex
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex
	# Double pass for alignment and internal references
	$(PDFLATEX) -interaction=nonstopmode $(MAIN).tex

clean:
	rm -f *.aux *.log *.out *.nav *.snm *.toc *.pdf
	rm -f performance_comparison.pdf confusion_matrix.pdf

.PHONY: all pdf clean plots
