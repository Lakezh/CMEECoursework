# CMEE 2022 HPC exercises R code pro forma
# For stochastic demographic model cluster run

# Clear the workspace and turn off graphics
rm(list=ls())
graphics.off()
# Load all the functions you need by sourcing your main.R file
source("/rds/general/user/zh1323/home/zh1323/zh1323_HPC_2023_main.R")

# Read in the job number from the cluster
iter <- as.numeric(Sys.getenv("PBS_ARRAY_INDEX"))
# iter <- 1
# Control the random number seeds so that each parallel simulation takes place with a different seed
set.seed(iter)

# Set initial conditions
simulation_length = 120
simulation_times = 150

# Ensure that 25 of the parallel simulations are allocated to each of the initial conditions
if (iter >= 1 && iter <= 25) {
    initial_state <- state_initialise_adult(4, 100)  # a large population of 100 adults
} else if (iter <= 50) {
    initial_state <- state_initialise_adult(4, 10) # a small population of 10 adults
} else if (iter <= 75) {
    initial_state <- state_initialise_spread(4, 100) # a large population of 100 individuals spread across the life stages
} else if (iter <= 100) {
    initial_state <- state_initialise_spread(4, 10) # a small population of 10 individuals spread across the life stages
}

growth_matrix <- matrix(c(0.1, 0.0, 0.0, 0.0,
                        0.5, 0.4, 0.0, 0.0,
                        0.0, 0.4, 0.7, 0.0,
                        0.0, 0.0, 0.25, 0.4), nrow=4, ncol=4, byrow=T)
reproduction_matrix <- matrix(c(0.0, 0.0, 0.0, 2.6,
                                0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0,
                                0.0, 0.0, 0.0, 0.0), nrow=4, ncol=4, byrow=T)
clutch_distribution <- c(0.06,0.08,0.13,0.15,0.16,0.18,0.15,0.06,0.03)

output_file_name <- paste("neu_sim_results_", iter, ".rda", sep="")

# Initialise a list which will contain the results of your 150 simulations
simu_results <- list()
for(i in 1:simulation_times){
    state_list <- stochastic_simulation(initial_state, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length)
    simu_results <- c(simu_results, list(state_list))
}
save(simu_results, file = output_file_name)







