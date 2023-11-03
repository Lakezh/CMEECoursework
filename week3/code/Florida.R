rm(list=ls())

load("../data/KeyWestAnnualMeanTemperature.RData")
ls()
class(ats)
head(ats)
plot(ats)

#calculate the observed correlation coefficient
observed_cor <- cor(ats$Year, ats$Temp)
#set the number of permutations
n_permutations <- 10000
#initialize a vector to store random correlation coefficients
rand_cor <- numeric(n_permutations)

#perform the permutation test
for (i in 1:n_permutations) {
  #shuffle the temperature data
  shuffled_temp <- sample(ats$Temp)
  
  #calculate the correlation coefficient for the shuffled data
  rand_cor[i] <- cor(ats$Year, shuffled_temp)
}

library(ggplot2)

#create a density plot of random correlation coefficients
ggplot(data.frame(rand_cor = rand_cor), aes(x = rand_cor)) +
  geom_density(fill = "skyblue", color = "black") +
  labs(title = "Density Plot of Random Correlation Coefficients",
       x = "Correlation Coefficients",
       y = "Density")+
  geom_vline(xintercept = observed_cor, linetype = "dashed", color = "red", size = 1)
ggsave("../results/Florida_plot.pdf", device = "pdf")

observed_cor
#caculate the p-value
p_value <- sum(rand_cor >= observed_cor) / n_permutations

p_value

