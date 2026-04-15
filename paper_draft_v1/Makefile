OUTDIR := build
MAIN := main.tex

.PHONY: pdf clean

pdf:
	rm -f $(OUTDIR)/main.fdb_latexmk $(OUTDIR)/main.fls
	LC_ALL=C LANG=C latexmk -f -pdf -outdir=$(OUTDIR) $(MAIN) || test -f $(OUTDIR)/main.pdf

clean:
	LC_ALL=C LANG=C latexmk -C -outdir=$(OUTDIR) $(MAIN)
