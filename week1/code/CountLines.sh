#!/bin/bash
if [ $# -ne 1 ]; then
    echo "Error. Format: $0 <InputFile1>"
    exit
fi

if [ ! -f "$1" ]; then
    echo "Input file does not exist"
    exit
fi

NumLines=`wc -l < $1`
echo "The file $1 has $NumLines lines"
echo done!
exit

