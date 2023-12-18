#load necessary libraries
library(ggplot2)
library(dplyr)

#read the predator-prey dataset
data <- read.csv(paste0("../data/EcolArchives-E089-51-D1.csv"))

#convert masses and size ratios to logarithms
data$log_predator_mass <- log(data$Predator.mass)
data$log_prey_mass <- log(data$Prey.mass)
data$log_size_ratio <- log(data$Prey.mass / data$Predator.mass)

#calculate mean and median values by feeding type
summary_stats <- data %>%
  group_by(Type.of.feeding.interaction) %>%
  summarize(
    Mean_log_predator_mass = mean(log_predator_mass),
    Median_log_predator_mass = median(log_predator_mass),
    Mean_log_prey_mass = mean(log_prey_mass),
    Median_log_prey_mass = median(log_prey_mass),
    Mean_log_size_ratio = mean(log_size_ratio),
    Median_log_size_ratio = median(log_size_ratio)
  )

#save statistics to a CSV file
write.csv(summary_stats, file = paste0("../results/PP_Results.csv"), row.names = FALSE)

#create subplots for predator mass, prey mass, and size ratio by feeding type
pdf("../results/Pred_Subplots.pdf")
par(mfrow = c(2, 2))

for (feed_type in unique(data$Type.of.feeding.interaction)) {
  subset_data <- subset(data, Type.of.feeding.interaction == feed_type)
  plot(density(subset_data$log_predator_mass), main = paste("Predator Mass -", feed_type), xlab = "Log Predator Mass")
}
dev.off()

pdf("../results/Prey_Subplots.pdf")
par(mfrow = c(2, 2))
for (feed_type in unique(data$Type.of.feeding.interaction)) {
  subset_data <- subset(data, Type.of.feeding.interaction == feed_type)
  plot(density(subset_data$log_prey_mass), main = paste("Prey Mass -", feed_type), xlab = "Log Prey Mass")
}
dev.off()

pdf("../results/SizeRatio_Subplots.pdf")
par(mfrow = c(2, 2))
for (feed_type in unique(data$Type.of.feeding.interaction)) {
  subset_data <- subset(data, Type.of.feeding.interaction == feed_type)
  plot(density(subset_data$log_size_ratio), main = paste("Size Ratio -", feed_type), xlab = "Log Size Ratio")
}
dev.off()


#another way to do it
ggplot(data, aes(x = log_predator_mass)) +
  geom_density() +
  facet_wrap(~ Type.of.feeding.interaction) +
  labs(title = "Predator mass")
#ggsave("../results/Pred_Subplots_dw.pdf", device = "pdf")