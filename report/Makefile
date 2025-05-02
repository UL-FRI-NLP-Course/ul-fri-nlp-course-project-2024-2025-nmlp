LATEX=pdflatex
BIBTEX=bibtex

.PHONY: %
%: %.tex
	$(LATEX) --shell-escape $< && \
	$(BIBTEX) $@ && \
	$(LATEX) --shell-escape $< && \
	$(LATEX) --shell-escape $<

.PHONY: clean
clean:
	rm -rf *.aux *.bbl *.bcf *.blg *.log *.out *.xml *.toc *.nav *.snm *.vrb
