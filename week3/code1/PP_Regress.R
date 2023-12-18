#load necessary libraries
library(ggplot2)
library(dplyr)

#set the path to the dataset and read it
data <- read.csv("../data/EcolArchives-E089-51-D1.csv", header = TRUE, sep = ",", stringsAsFactors = TRUE)

regression_results <- data %>%
  group_by(Type.of.feeding.interaction, Predator.lifestage) %>%
  do(model = lm(Prey.mass ~ Predator.mass, data = .)) %>%
  summarize(
    slope = coef(model)[[2]],
    intercept = coef(model)[[1]],
    R = summary(model)$r.squared,
    F_statistic = summary(model)$fstatistic[1],
    p_value = summary(model)$fstatistic[2]
  )

write.csv(regression_results, "../results/PP_Regress_Results.csv", row.names = FALSE)

 #save plot as a PDF in the Results directory
ggplot(data, aes(x = log(Predator.mass), y = log(Prey.mass), color = Predator.lifestage)) +
  geom_point() +
  geom_smooth(method = "lm", se = FALSE) +
  facet_wrap(.~Type.of.feeding.interaction) +
  labs(title = "Linear Regression on Predator-Prey Interactions") +
  theme_minimal()
ggsave("../results/PP_Regress_plot.pdf", device = "pdf")