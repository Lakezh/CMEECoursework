#!/bin/bash
if [ $# -ne 1 ]; then
#when the input files are not 1, the message comes out using echo
    echo "Error. Format: $0 <InputFile1>"
    exit
fi

if [ ! -f "$1" ]; then
#test if the files exist
    echo "Input file does not exist"
    exit
fi

NumLines=`wc -l < $1`
echo "The file $1 has $NumLines lines"
echo done!
exit

