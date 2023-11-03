# This function calculates heights of trees given distance of each tree 
# from its base and angle to its top, using  the trigonometric formula 
#
# height = distance * tan(radians)
#
# ARGUMENTS
# degrees:   The angle of elevation of tree
# distance:  The distance from base of tree (e.g., meters)
#
# OUTPUT
# The heights of the tree, same units as "distance"

TreeData <- read.csv("../data/trees.csv")
head(TreeData)
TreeLength <- length(TreeData$Distance.m)
TreeData$Distance.m[2]

TreeHeight <- function(degrees, distance) {
    radians <- degrees * pi / 180
    height <- distance * tan(radians)
    print(paste("Tree height is:", height))
  
    return (height)
}

#Load required libraries
library(dplyr)
#Load the data
data <- read.csv("../data/trees.csv")
#Calculate tree heights
data <- data %>%
  mutate(Tree.Height.m = TreeHeight(Angle.degrees, Distance.m))
#Save the results
write.csv(data, file = "../results/TreeHts.csv", row.names = FALSE)

