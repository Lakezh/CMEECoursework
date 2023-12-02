#!/bin/bash
if [ -f "Modified.py" ]; then
   echo "Running Modified.py"
   python3 Modified.py
else
   echo "File Not Found: Modified.py does not exist."
   exit 1
fi

#!/bin/bash

if [ -f "miniproject.Rmd" ]; then
   echo "Running miniproject.Rmd"
   R -e "rmarkdown::render('miniproject.Rmd')"
else
   echo "File Not Found: miniproject.Rmd does not exist."
   exit 1
fi
#!/bin/bash

if [ -f "miniproject.tex" ]; then
    echo "Compiling miniproject.tex"

    pdflatex miniproject.tex
    bibtex miniproject
    pdflatex miniproject.tex
    pdflatex miniproject.tex

    # Remove temporary files
    rm *.aux *.log *.bbl *.blg *.bcf *.run.xml

    # Check if LaTeX compilation was successful
    if [ $? -eq 0 ]; then
        echo "miniproject.tex compiled successfully."
    else
        echo "Error: Failed to compile miniproject.tex."
        exit 1
    fi
else
    echo "Error: miniproject.tex does not exist."
    exit 1
fi


