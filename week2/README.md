week2 content
code:
1.align_seqs.py
Aligns biological sequences. Reads sequence data from a CSV file and likely performs sequence alignment, outputting the results to another file.

2.basic_csv.py
Processes a CSV file containing fields like species and family. Reads the file, stores each row as a tuple in a list.

3.basic_io1.py
Showcases basic file reading operations. Opens a text file and prints each line, exemplifying how to read files line by line.
The test.txt file is in sandbox directory, whcih means it can not be used.

4.basic_io2.py
Demonstrates file writing operations. Writes a range of numbers to a file, with each number on a new line. 

5.basic_io3.py
Illustrates how to store objects using pickle. Serializes a dictionary and saves it to a binary file.

6.boilerplate.py
Provides a basic structure for a Python program, including standard metadata and a main function that prints a message.

7.cfexercises1.py
Contains functions for basic mathematical calculations.

8.cfexercises2.py
Includes additional mathematical functions.

9.control_flow.py
Demonstrates various control flow statements in Python.

10.debug_example.py
Contains a function buggyfunc with an interactive debugging setup using ipdb.

11.debugme.py
Similar to debug_example.py, features a buggyfunc designed for debugging practice.

12.dictionary.py
Creates a dictionary from a list of tuples, likely organizing species data.

13.lc1.py
Uses list comprehensions to process bird data, creating separate lists for Latin names, common names, and body masses.

14.lc2.py
Handles UK rainfall data using list comprehensions.

15.loops.py
Demonstrates examples of loop structures.

16.MyExampleScript.py
Demonstrates basic variable assignment and printing in Python, using string variables.

17.oaks.py
Processes a list of tree species to identify oak species using a defined function is_an_oak.

18.oaks_debugme.py
Focuses on oak tree data, includes debugging lines and doctests within is_an_oak function.

19.scopes.py
Illustrates the scope of variables in nested functions, showing impacts on global and local variables.

20.sysargv.py
Demonstrates the use of sys.argv for accessing command-line arguments by printing the script name, number of arguments, and the arguments.

21.test_control_flow.py
Set up to demonstrate control flow statements and includes doctest modules for inline testing.

22.tuple.py
Involves operations with a tuple of bird data, instructing to print this data, focusing on tuple handling and output.

data:
1.sequences.csv
Two DNA sequences. And I used squences in align_seqs.py to find the best alignment.
2.TestOaksData.csv
A set data of species of oaks.
3.JustOaksData.csv
A new output data after applying oaks_debugme.py to TestOaksData.csv.
4.testcsv.csv
It is a data set of species, containing species names,infraorder,family names ,distribution and body mass
5.bodymass.csv
Extract the data of species names and body mass from testcsv.csv.

results:
The best alignment sequence from align_seqs.py will be output and should be put in results directory.
