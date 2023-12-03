The project utilizes R for data analysis and modeling, and LaTeX for documenting the research findings
Repository Structure
    code/: Contains R script and Latex document.
    data/: Contains the datasets used in the analysis.
    results/: Stores the output of the analysis, including model comparison results and figures.
    
Tools and Packages Used
Here is an overview of the key tools used and their purposes:
R Packages
    readr
        Purpose: Used for efficiently reading and writing data in various formats.
        Note: Ensure data formats are compatible with the functions used for reading (e.g., CSV, TSV).
    minpack.lm
        Purpose: Provides a robust method for non-linear least squares fitting, essential for fitting logistic and Gompertz models.
        Note: Requires initial parameter values for fitting; sensitive to these starting values.
    dplyr
        Purpose: Facilitates data manipulation and cleaning, such as filtering, selecting, and mutating data frames.
        Note: Be familiar with dplyr's syntax for efficient data manipulation.
    ggplot2
        Purpose: Utilized for data visualization and creating complex plots.
        Note: Familiarity with ggplot2 layers and aesthetics is crucial for creating custom plots.

LaTeX for Documentation
    Purpose: LaTeX is used for preparing the research document, ensuring a professional layout and structure.

Running the Analysis
    Execute the R scripts to perform data processing, analysis, and model fitting.
    Compile the LaTeX document to produce the final formatted report.
