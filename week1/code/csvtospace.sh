if [ $# -ne 2 ]; then
    echo "Error. Format: $0 <InputFile1> <OutputFile>"
    exit
fi

if [ ! -f "$1" ]; then
    echo "Input file does not exist"
    exit
fi

if [ ! -f "$2" ]; then
    echo "Output file does not exist."
    exit
cat $1 | tr -s "," " " >> $2
exit
echo done!