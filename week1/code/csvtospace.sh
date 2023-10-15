if [ $# -ne 2 ]; then
#when the input files are not 2, the message comes out using echo
    echo "Error. Format: $0 <InputFile1> <OutputFile>"
    exit
fi

if [ ! -f "$1" ]; then
#test if the files exist
    echo "Input file does not exist"
    exit
fi

if [ ! -f "$2" ]; then
#test if the files exist
    echo "Output file does not exist."
    exit
cat $1 | tr -s "," " " >> $2
echo done!
exit