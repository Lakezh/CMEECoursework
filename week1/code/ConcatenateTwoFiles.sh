
if [ $# -ne 3 ]; then
    echo "Error. Format: $0 <InputFile1> <InuputFile2> <OutputFile>"
    exit
fi

InputFiles=("$1" "$2")
for InputFiles in "{InputFiles[@]}";do
    if [ ! -f "$InputFiles" ]; then
        echo "Input file does not exist."
        exit
    fi

if [ ! -f "$3" ]; then
    echo "Output file does not exist."
    exit
fi

cat $1 > $3
cat $2 >> $3
echo "Merged File is"
cat $3
echo done!
exit
