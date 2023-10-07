echo "This script was called with $# parameters"
echo "the first argument is $1"

MY_VAR='some string'
echo 'the current value of the variable'
echo 'please enter a new string'
read MY_VAR

echo 'Enter two numbers separated by space'
read a b
echo 'you entered' $a 'and' $b '; their sum is:'
MY_SUM=$(expr $a + $b)
echo $MY_SUM
