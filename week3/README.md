week3 content
code:
1. apply1.R
Demonstrates the use of the apply function on a matrix. It calculates the mean and variance for each row, and the mean for each column of a randomly generated 10x10 matrix.

2. apply2.R
Showcases the use of a custom function with the apply function. It applies a conditional operation to each row of a matrix, multiplying elements by 100 if their sum is positive.

3. basic_io.R
Introduces basic input/output operations in R. It includes reading a CSV file, writing and appending data to a file, and handling headers and row names.

4. boilerplate.R
A template script for creating and testing a simple function. It demonstrates basic function definition, argument handling, and output in R.

5. break.R
A simple demonstration of using a break statement in a loop. The script uses a while loop and exits it when i equals 10.

6. browse.R
Implements a function simulating exponential growth and demonstrates debugging with the browser() function. 

7. control_flow.R
An instructional script covering basic control flow structures, including if/else statements and for loops, with practical examples.

8. DataWrang.R
Focuses on loading and initial processing of the Pound Hill dataset, including data conversion and basic transformations.

9. DataWrangTidy.R
Demonstrates data wrangling using tidyverse. It transposes a dataset, replaces missing values, and reshapes it from wide to long format.

10. Florida.R
Performs statistical analysis on temperature data, including a permutation test to assess the significance of an observed correlation coefficient.

11. Girko.R
Centers on mathematical operations involving eigenvalues and ellipses, including a custom function for building ellipses.

12. GPDD_Data.R
Visualizes geographic data by superimposing data points on a world map. And discuissed about the biases that might occured in analysis based on the given data.

13. MyBars.R
Uses ggplot2 to create visualizations with lineranges, demonstrating complex plotting techniques.

14. next.R
A script demonstrating the use of the next statement in loops. It skip iterations when the number is even. Therefore, it can only prints odd numbers.

15. plotLin.R
Demonstrates linear regression analysis and data visualization.

16. PP_Dists.R
Focuses on analyzing a predator-prey dataset, including transforming data into logarithmic form and calculating statistical summaries by feeding type.

17. PP_Regress.R
Performs grouped linear regression analysis on predator-prey data, summarizing key regression statistics like slope, intercept, and R-squared.

18. preallocate.R
Illustrates the importance of preallocating memory in loops by comparing execution times of functions with and without preallocation.

19. R_conditionals.R
Showcases the use of conditional statements in R through functions that perform checks like evenness, power of 2, and primality.

20. Ricker.R
Simulates the Ricker model for population dynamics.

21. sample.R
Includes functions for sampling from a population and calculating means, emphasizing the impact of memory preallocation in loops.

22. SQLinR.R
Demonstrates the integration of SQL database operations within R.

23. TreeHeight.R
Calculates tree heights using trigonometry based on distance and angle measurements. Reads tree data from a CSV file and applies a mathematical formula for height calculation.

24. TreeHeight_edit.R
This funtion is the modified TreeHeight.R, whcih is the script for Tree heights practals. Names it TreeHeight_edit.R to distinguish it from TreeHeight.R.

25. try.R
Showcases error handling and conditional execution in R using a custom sampling function, demonstrating the use of try for managing errors in loops.

26. Vectorize1.R
Illustrates the efficiency difference between loop-based and vectorized operations in R.

27. Vectorize2.R
Implements a stochastic version of the Ricker model using both loop-based and vectorized approaches.

data:
1. EcolArchives-E089-51-D1.csv
represents the traits of birds of diffrent species.
2. GPDDFiltered.RData
Map data from Global Population Dynamics Database
3. KeyWestAnnualMeanTemperature.RData
Data of annual mean temperature from Key West in Florida, USA for the 20th century.
4. PoundHillData.csv
Field data for wrangling.
5. PoundHillMetaData.csv
The metadata of PoundHillData.csv.
6. Results.txt
A set of data used in MyBars.R for producing plots.
7. trees.csv
Data of trees including the species, distance, angle degree and tree heights. 

results:
All the output results should be put in this directory.

writeup:
Get the results first, and then run Florida.tex. Since this file needs the output figures.I put Florida.tex in a seperate directory which is called writeup. The pdf outputs are also in writeup directory.