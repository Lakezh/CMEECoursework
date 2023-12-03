#!/bin/bash
if [ -f "miniproject.R" ]; then
   echo "Running miniproject.R"
   python3 Modified.py
else
   echo "File Not Found: miniproject.R does not exist."
   exit 1
fi
Rscript miniproject.R

pdflatex miniproject.tex
   bibtex miniproject
   pdflatex miniproject.tex
   pdflatex miniproject.tex
rm*.log*.aux
