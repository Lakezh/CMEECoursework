#!/bin/bash

Rscript miniproject.R

pdflatex miniproject.tex
   bibtex miniproject
   pdflatex miniproject.tex
   pdflatex miniproject.tex
rm*.log*.aux