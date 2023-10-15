f [ $# -ne 1 ]; then
    echo "Error. Format: $0 <InputFile1>"
    exit
fi

if [ ! -f "$1" ]; then
    echo "Input file does not exist"
    exit
fi

echo "creating a comma delimited version of $1 ..."
cat $1 | tr -s "\t" "," >> $1.csv
echo "Done!"
exit