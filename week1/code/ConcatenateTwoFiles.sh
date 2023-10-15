#when the input files are not 3, the message comes out using echo
if [ $# -ne 3 ]; then
    echo "Error. Format: $0 <InputFile1> <InuputFile2> <OutputFile>"
    exit
fi


InputFiles=("$1" "$2")
for InputFiles in "{InputFiles[@]}";do
#using a for loop to include both input files
    if [ ! -f "$InputFiles" ]; then
    #test if the files exist
        echo "Input file does not exist."
        exit
    fi

if [ ! -f "$3" ]; then
#test if the files exist
    echo "Output file does not exist."
    exit
fi

cat $1 > $3
cat $2 >> $3
echo "Merged File is"
cat $3
echo done!
exit
