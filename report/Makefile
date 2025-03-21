LATEX=pdflatex
BIBTEX=bibtex

.PHONY: %
%: %.tex
	$(LATEX) $< && \
	$(BIBTEX) $@ && \
	$(LATEX) $< && \
	$(LATEX) $<

.PHONY: clean
clean:
	rm -rf *.aux *.bbl *.bcf *.blg *.log *.out *.xml *.toc *.nav *.snm *.vrb
