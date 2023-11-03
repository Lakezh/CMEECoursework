#load the library
library(tidyverse)

#read the dataset
MyData <- read.csv("../data/PoundHillData.csv", header = FALSE)
MyMetaData <- read.csv("../data/PoundHillMetaData.csv", header = TRUE, sep = ";")

#transpose the dataset
MyData <- as.data.frame(t(MyData))
MyData[MyData == ""] <- 0

#convert from wide to long format
TempData <- MyData[-1,] #remove the first row
colnames(TempData) <- MyData[1,] #set column names from the original data

MyWrangledData <- TempData%>%
  gather(key = "Species", value = "Count", -Cultivation, -Block, -Plot, -Quadrat) %>%
  mutate(
    Cultivation = as.factor(Cultivation),
    Block = as.factor(Block),
    Plot = as.factor(Plot),
    Quadrat = as.factor(Quadrat),
    Count = as.integer(Count)
  )


str(MyWrangledData)
head(MyWrangledData)
dim(MyWrangledData)