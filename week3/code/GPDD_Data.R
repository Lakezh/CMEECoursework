#Load necessary libraries
library(maps)

# Load GPDD data
load("../data/GPDDFiltered.RData")

#Create a world map
map("world")

#Superimpose locations from GPDD dataframe
points(gpdd$long, gpdd$lat, col = "red", pch = 1)

#Potential biases in the GPDD dataset
#As observed in the map, biases may include:
#1. Spatial bias: Most of data might from the areas with higher research efforts and more advanced techniques. leading to an uneven geographical representation.
#2. bias caused by location limitation: data collection might be easier in easily accessible areas. The data might be less in regions hard to access.
#3. Bias towards certain species: Some species might be more heavily studied and easy to collect the data.


