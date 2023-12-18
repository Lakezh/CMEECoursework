week1 content:
code:1.boilerplate.sh: 
A simple boilerplate for shell scripts, which can output a short message.
2.CompileLaTeX.sh:
Compiles a LaTeX file into a PDF. It runs pdflatex and bibtex on the provided LaTeX file, then opens the resulting PDF.
3.ConcatenateTwoFiles.sh:
Concatenates two input files into a single output file. It checks if there is correct number of input arguments.There should be two input files and one file for output.
4.CountLines.sh:
Counts the number of lines in a provided file. It checks if there is correct number which is one of input arguments.
5.csvtospace.sh:
Converts a CSV (comma-separated values) file to a space-separated format. It checks for two input arguments and verifies the existence of the input file. There should be one input file and one file for ouput.
6.MyExampleScript.sh:
A script demonstrating the use of variables and displays a message.
7.tiff2png.sh:
Converts all TIFF files in the current directory to PNG format using the convert command.
8.UnixPrac1.txt:
Contains Unix shell commands for processing FASTA formatted files. Includes line counting, character counting, specific sequence extraction, nucleotide counting, and ratio calculations.
9.variables.sh
Illustrates the use of variables in shell scripts. It includes examples of special variables and their usage.
10.tabtocsv.sh
Substitutes tabs in the provided file with commas, effectively converting a tab-delimited file into a CSV format.

data: 
I used three FASTA data file which are 407228326.fasta, 407228412.fasta and E.coli.fasta representing DNA squences.

results:
There are no results from the scripts. However, some scripts might produce output if the scripts are used. These output should be put in results directory.

writeup:
I put FirstExample.tex in a seperate directory which is called writeup. I used FirstExample.tex to test CompileLaTeX.sh. Remember to use relative path when inputs the file. The pdf outputs are also in writeup directory.