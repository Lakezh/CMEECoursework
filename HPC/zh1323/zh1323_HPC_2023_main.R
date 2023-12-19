# CMEE 2022 HPC exercises R code main pro forma
# You don't HAVE to use this but it will be very helpful.
# If you opt to write everything yourself from scratch please ensure you use
# EXACTLY the same function and parameter names and beware that you may lose
# marks if it doesn't work properly because of not using the pro-forma.

name <- "Zhongbin Hu"
preferred_name <- "Zhongbin Hu"
email <- "zh1323@imperial.ac.uk"
username <- "zh1323"

# Please remember *not* to clear the workspace here, or anywhere in this file.
# If you do, it'll wipe out your username information that you entered just
# above, and when you use this file as a 'toolbox' as intended it'll also wipe
# away everything you're doing outside of the toolbox.  For example, it would
# wipe away any automarking code that may be running and that would be annoying!

# Question 1
species_richness <- function(community){
  # Find unique species in the community
  unique_species <- unique(community)
  # Calculate the richness as the length of unique species
  richness <- length(unique_species)
  return(richness)
}
# Test
community <- c(1, 4, 4, 5, 1, 6, 1)
species_richness(community)  # This should return 4


# Question 2
init_community_max <- function(size){
  # Generate an initial sequence from 1 to size, each representing a unique species
  return(community)
}
# Test
size <- 10
init_community_max(size)


# Question 3
init_community_min <- function(size){
  # Generate an initial sequence with total number of individuals given by size, representing the same species
  community <- rep(1, size)
  return(community)
}
# Test
species_richness(init_community_min(5))
species_richness(init_community_max(5))


# Question 4
choose_two <- function(max_value){
  # Sample two distinct numbers from 1 to max_value without replacement
  chosen_numbers <- sample(1:max_value, size = 2, replace = FALSE)
  return(chosen_numbers)
}
# Test
result <- choose_two(4)
print(result)


# Question 5
neutral_step <- function(community){
  # Use the choose_two function to pick two individuals
  # The first will die, and the second will reproduce
  positions <- choose_two(length(community))
  # Replace the species of the dying individual with the species of the reproducing individual
  community[positions[1]] <- community[positions[2]]
  return(community)
}
# Test
neutral_step(c(1, 2)) 


# Question 6
neutral_generation <- function(community){
  # Calculate the size of the community
  community_size <- length(community)
  # Determine the number of steps, rounding up or down randomly if size is odd
  steps <- ifelse(size %% 2 == 0, size / 2, floor(size / 2) + sample(c(0, 1), 1))
  # Perform neutral steps for one generation
  for (i in 1:steps) {
    community <- neutral_step(community)
  }
  return(community)
}
# Test
neutral_generation(c(1, 2, 3, 4, 5, 6, 7, 8, 9, 10))

# Question 7
neutral_time_series <- function(community,duration)  {
  # Initialize a vector to store species richness over time
  species_richness_series <- numeric(duration + 1)
  # Record the species richness of the initial community
  species_richness_series[1] <- species_richness(community)
  # Run the simulation for the specified duration
  for (i in 1:duration) {
    community <- neutral_generation(community)
    species_richness_series[i + 1] <- species_richness(community)
  }
  return(species_richness_series)
}
# Test
initial_community <- init_community_max(7)
time_series <- neutral_time_series(initial_community, 20)
time_series


# Question 8
question_8 <- function() {
  # Creating a community with maximum diversity of 100 individuals
  community_at_start <- init_community_max(100)
  
  # Conducting the simulation across 200 generations
  simulation_results <- neutral_time_series(community_at_start, 200)
  print(simulation_results)
  png(file="question_8.png", width=600, height=400)
  # Plotting the results
  plot(simulation_results, type="l", main="Simulation of Neutral Model Across 200 Generations",
       xlab="Generation", ylab="Diversity of Species")
  Sys.sleep(0.1)
  dev.off()
  # Explanation of the long-term outcome of the system
  answer <- "As iterations increase, the system tends to a state of lower species diversity. 
  In the absence of new species emergence and with random extinction of existing species, 
  species can only reproduce or go extinct, ultimately leading to a scenario with a single species.";
  return(answer)
}
question_8()

# Question 9
neutral_step_speciation <- function(community, speciation_rate) {
  if (runif(1) < speciation_rate) {
    # Speciation
    new_species <- max(community) + 1
    # Use the choose_two function to pick two different individuals
    chosen_indices <- choose_two(length(community))
    # Replace the dying individual with the new species
    dead_individual <- chosen_indices[1]
    community[dead_individual] <- new_species
  } else {
    # Normal neutral step
    community <- neutral_step(community)
  }
  return(community)
}
# Test
neutral_step_speciation(c(1, 1, 2), speciation_rate = 0.2)


# Question 10
neutral_generation_speciation <- function(community,speciation_rate)  {
  community_size <- length(community)
  steps <- ifelse(runif(1) < 0.5, floor(community_size / 2), ceiling(community_size / 2))
  
  for (i in 1:steps) {
    community <- neutral_step_speciation(community, speciation_rate)
  }
  
  return(community)
}


# Question 11
neutral_time_series_speciation <- function(community,speciation_rate,duration)  {
  # Initialize a vector to store species richness over time
  species_richness_series <- numeric(duration + 1)
  # Record the species richness of the initial community
  species_richness_series[1] <- species_richness(community)
  # Run the simulation for the specified duration
  for (i in 1:duration) {
    community <- neutral_generation_speciation(community, speciation_rate)
    species_richness_series[i + 1] <- species_richness(community)
  }
  return(species_richness_series)
}
# Example usage
initial_community <- init_community_max(10)
duration <- 20
speciation_rate <- 0.1
time_series <- neutral_time_series_speciation(initial_community, duration, speciation_rate)
time_series 


# Question 12
question_12 <- function()  {
  # Initialize the parameters
  duration <- 200
  speciation_rate <- 0.1
  community_size <- 100
  community_max <- init_community_max(community_size)
  community_min <- init_community_min(community_size)
  # Run simulations
  time_series_max <- neutral_time_series_speciation(community_max, duration, speciation_rate)
  time_series_min <- neutral_time_series_speciation(community_min, duration, speciation_rate)
  # Plot and save the time series
  png(filename="question_12", width = 600, height = 400)
  plot(time_series_max, type = "l", col = "blue", xlab = "Generation", ylab = "Species Richness")
  lines(time_series_min, type = "l", col = "red")
  legend("topright", legend = c("Max Diversity", "Min Diversity"), col = c("blue", "red"), lty = 1)
  Sys.sleep(0.1)
  dev.off()
  return("The plot shows how initial conditions affect species richness over time. With speciation, both maximal and minimal diversity communities tend to converge to a similar level of species richness, illustrating the randomness and balancing nature of speciation and extinction in the neutral model.")
}
question_12()


# Question 13
species_abundance <- function(community)  {
  abundance <- sort(table(community), decreasing = TRUE)
  return(as.vector(abundance))
}
# Test
species_abundance(c(1, 5, 3, 6, 5, 6, 1, 1))  # Should return c(3, 2, 2, 1)


# Question 14
octaves <- function(abundance_vector) {
  # Determine the octave class for each abundance value
  # Adding 1 because log2(1) is 0, and we want species with abundance 1 to fall into the first octave
  octave_classes <- floor(log2(abundance_vector))
  # Count the number of species in each octave class
  octave_counts <- tabulate(octave_classes + 1)
  return(octave_counts)
}
# Test
abundance <- species_abundance(c(1, 5, 3, 6, 5, 6, 1, 1))
octaves(abundance)  # Converts the species abundances into octave classes


# Question 15
sum_vect <- function(x, y) {
  # Make both vectors the same length by filling the shorter one with zeros
  length_diff <- abs(length(x) - length(y))
  if (length(x) < length(y)) {
    x <- c(x, rep(0, length_diff))
  } else {
    y <- c(y, rep(0, length_diff))
  }
  # Return the sum of the vectors
  return(x + y)
}
# Test
sum_vect(c(1, 3), c(1, 0, 5, 2))  # Should return c(2, 3, 5, 2)


# Question 16 
question_16 <- function() {
  # Set up parameters
  initial_conditions <- list(init_community_max(100), init_community_min(100))
  final_octaves_min <- final_octaves_max <- numeric()
  burn_in <- 200
  total_generations <- 2000
  record_every <- 20
  # Burn-in period
  for (initial_condition in initial_conditions) {
    community <- initial_condition
    for (i in 1:burn_in) {
      community <- neutral_generation_speciation(community, 0.1)
    }
    # Record species abundance after burn-in
    for (i in 1:(total_generations/record_every)) {
      for (j in 1:record_every) {
        community <- neutral_generation_speciation(community, 0.1)
      }
      abundance <- species_abundance(community)
      octaves <- octaves(abundance)
      # Summing octave vectors
      if (identical(initial_condition, initial_conditions[[1]])) {
        final_octaves_max <- sum_vect(final_octaves_max, octaves)
      } else {
        final_octaves_min <- sum_vect(final_octaves_min, octaves)
      }
    }
  }
  # Calculate average octaves
  avg_octaves_max <- final_octaves_max / (total_generations/record_every)
  avg_octaves_min <- final_octaves_min / (total_generations/record_every)
  
  # Plot and save the bar plots
  png(filename="question_16_min.png", width = 600, height = 400)
  barplot(avg_octaves_min, main="Species Abundance (Min Initial Condition)", xlab="Octave", ylab="Average Species Count")
  # plot your graph here
  Sys.sleep(0.1)
  dev.off()
  
  png(filename="question_16_max.png", width = 600, height = 400)
  barplot(avg_octaves_max, main="Species Abundance (Max Initial Condition)", xlab="Octave", ylab="Average Species Count")
  # plot your graph here
  Sys.sleep(0.1)
  dev.off()
  return("The initial condition of the system matters significantly at the beginning but tends to have less impact over time as the system reaches a steady state.")
}
question_16()



# Question 17
neutral_cluster_run <- function(speciation_rate, size, wall_time, interval_rich, interval_oct, burn_in_generations, output_file_name) {
  # Prepare to record data and initialize community diversity
  start_time <- proc.time()[3]
  community <- init_community_min(size)
  time_series <- numeric()
  abundance_list <- list()
  # Run the simulation
  generation <- 0
  # Convert minutes to seconds
  wall_time_seconds <- wall_time * 60  
  while (proc.time()[3] - start_time < wall_time_seconds) {
    community <- neutral_generation_speciation(community, speciation_rate)
    generation <- generation + 1
    # Record species richness during burn-in period
    if (generation <= burn_in_generations && generation %% interval_rich == 0) {
      time_series <- c(time_series, species_richness(community))
    }
    # Record species abundances as octaves
    if (generation %% interval_oct == 0) {
      abundance_list[[length(abundance_list) + 1]] <- octaves(species_abundance(community))
    }
  }
  # Calculate total simulation time
  total_time <- proc.time()[3] - start_time
  # Save the results
  save(list = c("time_series", "abundance_list", "community", "total_time", "speciation_rate", "size", "wall_time", "interval_rich", "interval_oct", "burn_in_generations"), file = output_file_name)
}
# Test the function with specified parameters
neutral_cluster_run(speciation_rate = 0.1, size = 100, wall_time = 10, interval_rich = 1, interval_oct = 20, burn_in_generations = 200, output_file_name = "my_test_file_1.rda")
# check the result
load(file = 'my_test_file_1.rda')

# Questions 18 and 19 involve writing code elsewhere to run your simulations on
# the cluster



# Question 20 
calculate_cluster_means <- function() {
  count_500 <- 0
  count_1000 <- 0
  count_2500 <- 0
  count_5000 <- 0
  
  sum_500 <- c()
  sum_1000 <- c()
  sum_2500 <- c()
  sum_5000 <- c()
  
  for(index in 1:100){
    load(file = paste0("neu_sim_results_", index, ".rda", sep=""))
    
    if(index <= 25){cluster_size = 500} else if
    (index <= 50){cluster_size = 1000} else if
    (index <= 75){cluster_size = 2500} else if
    (index <= 100){cluster_size = 5000}
    
    start_point = 80
    
    cumulative_total <- c()
    for(j in start_point:length(abundance)){
      cumulative_total <- sum_vect(cumulative_total, abundance[[j]])
    }
    
    assign(paste0("count_", cluster_size), (get(paste0("count_", cluster_size)) + (length(abundance) - start_point + 1)))
    assign(paste0("sum_", cluster_size), sum_vect(get(paste0("sum_", cluster_size)), cumulative_total))
  }
  
  average_500 <- sum_500/count_500
  average_1000 <- sum_1000/count_1000
  average_2500 <- sum_2500/count_2500
  average_5000 <- sum_5000/count_5000
  final_results <- list(average_500, average_1000, average_2500, average_5000) #create your list output here to return
  
  save(final_results, file = "Final_results.rda")
  # save results to an .rda file
}


  

plot_neutral_cluster_results <- function(){
  load("Combined_results.rda")
  
  png(filename="plot_neutral_cluster_results.png", width = 600, height = 400)
  
  layout_matrix <- matrix(1:4, 2, 2)
  layout(layout_matrix)
  cluster_sizes <- c('500', '1000', '2500', '5000')
  
  for (i in 1:4) {
    plot_data <- combined_average_results[[i]]
    max_plot_height <- max(plot_data) * 1.3
    barplot(plot_data, 
            main = paste("Cluster Size ", cluster_sizes[i], " - Species Abundance"),
            xlab = "Octave Classes", 
            ylab = "Number of Species",
            ylim = c(0, max_plot_height))
  }
  
  Sys.sleep(0.1)
  dev.off()
  
  return(combined_average_results)
}







#section 2
# Question 21
state_initialise_adult <- function(num_stages, initial_size) {
  # Create a vector with zeros for all stages
  state <- rep(0, num_stages)
  # Place all individuals in the final stage (adult stage)
  state[num_stages] <- initial_size
  return(state)
}


# Question 22
state_initialise_spread <- function(num_stages, initial_size) {
  # Calculate the base number of individuals per stage
  base <- floor(initial_size / num_stages)
  # Create the state vector with this base number
  state <- rep(base, num_stages)
  # Distribute the remainder, starting from the youngest stage
  for (i in 1:(initial_size %% num_stages)) {
    state[i] <- state[i] + 1
  }
  return(state)
}


# Question 23
deterministic_step <- function(state, projection_matrix) {
  # Perform matrix multiplication to get the new state
  new_state <- as.vector(projection_matrix %*% state)
  return(new_state)
}


# Question 24
deterministic_simulation <- function(initial_state, projection_matrix, simulation_length) {
  # Initialize the population size vector
  population_size <- numeric(simulation_length + 1)
  population_size[1] <- sum(initial_state)
  state <- initial_state
  
  # Apply the deterministic model over each time step
  for (i in 1:simulation_length) {
    state <- deterministic_step(state, projection_matrix)
    population_size[i + 1] <- sum(state)
  }
  
  return(population_size)
}

# Question 25
question_25 <- function() {
  # Define the growth and reproduction matrices
  growth_matrix <- matrix(c(0.1, 0.0, 0.0, 0.0, 
                            0.5, 0.4, 0.0, 0.0, 
                            0.0, 0.4, 0.7, 0.0, 
                            0.0, 0.0, 0.25, 0.4), 
                          nrow=4, ncol=4, byrow=TRUE)
  
  reproduction_matrix <- matrix(c(0.0, 0.0, 0.0, 2.6, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0), 
                                nrow=4, ncol=4, byrow=TRUE)
  
  # Combine them to form the projection matrix
  projection_matrix <- reproduction_matrix + growth_matrix
  
  # Simulation for a population of 100 adults
  adult_population <- state_initialise_adult(4, 100)
  adult_simulation <- deterministic_simulation(adult_population, projection_matrix, 24)
  
  # Simulation for a spread population of 100
  spread_population <- state_initialise_spread(4, 100)
  spread_simulation <- deterministic_simulation(spread_population, projection_matrix, 24)
  
  # Plotting the results
  time <- 0:24
  plot(time, adult_simulation, type='l', col='blue', xlab='Time', ylab='Population Size', main='Population Growth Comparison')
  lines(time, spread_simulation, type='l', col='red')
  legend('topright', legend=c('Adults Only', 'Spread Population'), col=c('blue', 'red'), lty=1)
  
  # Save the plot
  png("question_25.png", width = 600, height = 400)
  plot(time, adult_simulation, type='l', col='blue', xlab='Time', ylab='Population Size')
  lines(time, spread_simulation, type='l', col='red')
  legend('topright', legend=c('Adults Only', 'Spread Population'), col=c('blue', 'red'), lty=1)
  dev.off()
  
  # Textual explanation
  return("The initial distribution of the population across different life stages significantly influences both initial and long-term population growth dynamics. A population concentrated in the adult stage may initially grow faster due to higher reproduction rates, but a more evenly distributed initial population can lead to more sustainable growth over time.")
}
question_25()

# Question 26
multinomial <- function(pool, probs) {
  # If the sum of probabilities is less than 1, add the death probability
  if (sum(probs) < 1) {
    death_prob <- 1 - sum(probs)
    probs <- c(probs, death_prob)
  }
  
  # Draw from the multinomial distribution
  outcome <- rmultinom(1, pool, probs)
  
  # Return the outcomes as a vector
  return(colSums(outcome))
}


# Question 27
survival_maturation <- function(state, growth_matrix) {
  # 1. Initialize new_state with zeros
  new_state <- rep(0, length(state))
  
  # 2. Process each life stage
  for (i in 1:length(state)) {
    # a. Number of individuals in current stage
    num_individuals <- state[i]
    
    # b. Probabilities for staying or moving to the next stage
    probs <- growth_matrix[i, ]
    
    # c. Apply multinomial to get transitions
    transitions <- multinomial(num_individuals, probs)
    
    # d. Add transitions to new_state
    for (j in 1:length(transitions)) {
      if (j <= length(new_state)) {
        new_state[j] <- new_state[j] + transitions[j]
      }
    }
  }
  
  # 3. Return the new_state
  return(new_state)
}






# Test 1: If the state vector is full of 0s, the output new_state vector after applying survival and maturation should also be full of 0s.
# Assuming the functions multinomial and survival_maturation are defined
state_zeros <- c(0, 0, 0, 0)
growth_matrix <- matrix(c(0.1, 0.0, 0.0, 0.0, 
                          0.5, 0.4, 0.0, 0.0, 
                          0.0, 0.4, 0.7, 0.0, 
                          0.0, 0.0, 0.25, 0.4), 
                        nrow=4, ncol=4, byrow=TRUE)
new_state_zeros <- survival_maturation(state_zeros, growth_matrix)
print(new_state_zeros)  # Should be c(0, 0, 0, 0)

# Test 2: If the growth_matrix has no deaths, then the sum of new_state should be the same as the sum of state.
# Growth matrix with no deaths
growth_matrix_no_deaths <- matrix(rep(0.25, 16), nrow = 4, byrow = TRUE)
state_normal <- c(10, 10, 10, 10)
new_state_no_deaths <- survival_maturation(state_normal, growth_matrix_no_deaths)
print(sum(state_normal))  # Should be 40
print(sum(new_state_no_deaths))  # Should also be 40


# Test 3: If the growth_matrix is the 'identity matrix', then new_state should be exactly equal to state.
# Identity matrix
identity_matrix <- diag(4)
new_state_identity <- survival_maturation(state_normal, identity_matrix)
print(state_normal)  # Should be c(10, 10, 10, 10)
print(new_state_identity)  # Should also be c(10, 10, 10, 10)




# Question 28
random_draw <- function(probability_distribution) {
  # Create a vector of values corresponding to the indices of the probability distribution
  values <- seq_along(probability_distribution)
  
  # Use the sample function to draw one value based on the given probabilities
  return(sample(values, size = 1, prob = probability_distribution))
}


# Question 29
stochastic_recruitment <- function(reproduction_matrix, clutch_distribution) {
  # Retrieve the recruitment rate from the reproduction matrix (top-right element)
  recruitment_rate <- reproduction_matrix[1, ncol(reproduction_matrix)]
  
  # Calculate the expected (mean) clutch size
  # The clutch size is weighted by its probability
  clutch_sizes <- seq_along(clutch_distribution)
  expected_clutch_size <- sum(clutch_sizes * clutch_distribution)
  
  # Calculate the recruitment probability
  recruitment_probability <- recruitment_rate / expected_clutch_size
  
  # Check if the recruitment probability exceeds 1
  if (recruitment_probability > 1) {
    stop("Inconsistency in model parameters: Recruitment probability exceeds 1.")
  }
  
  return(recruitment_probability)
}


# Question 30
offspring_calc <- function(state,clutch_distribution,recruitment_probability){
  # Identify the number of adults in state
  num_adults <- state[length(state)]
  
  # Generate the number of adults which recruit (number of clutches)
  num_clutches <- rbinom(1, num_adults, recruitment_probability)
  
  # Initialize total_offspring
  total_offspring <- 0
  
  # For each clutch, draw the clutch size from clutch_distribution using random_draw
  for (i in 1:num_clutches) {
    clutch_size <- random_draw(clutch_distribution)
    total_offspring <- total_offspring + clutch_size
  }
  
  return(total_offspring)  
}

# Question 31
stochastic_step <- function(state,growth_matrix,reproduction_matrix,clutch_distribution,recruitment_probability){
  # Apply survival and maturation to generate a new_state
  new_state <- survival_maturation(state, growth_matrix)
  
  # Compute the number of offspring produced by state
  total_offspring <- offspring_calc(state, clutch_distribution, recruitment_probability)
  
  # Add the offspring into the appropriate entry of new_state
  new_state[1] <- new_state[1] + total_offspring
  
  return(new_state)  
}

# Question 32
stochastic_simulation <- function(initial_state, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length) {
  # Calculate the individual recruitment probability
  recruitment_probability <- stochastic_recruitment(reproduction_matrix, clutch_distribution)
  
  # Initialize the population size time series vector
  population_size <- numeric(simulation_length + 1)
  population_size[1] <- sum(initial_state)
  
  # Initialize the state
  state <- initial_state
  
  # Loop through each time step
  for (time_step in 1:simulation_length) {
    # Apply the stochastic step
    state <- stochastic_step(state, growth_matrix, reproduction_matrix, clutch_distribution, recruitment_probability)
    
    # Update the population size for this time step
    population_size[time_step + 1] <- sum(state)
    
    # Check if the population size has become zero
    if (sum(state) == 0) {
      # Fill the remaining entries with 0s
      population_size[(time_step + 2):(simulation_length + 1)] <- 0
      break # Halt the simulation loop
    }
  }
  
  return(population_size)
}


# Question 33
question_33 <- function() {
  # Parameters from Question 25
  growth_matrix <- matrix(c(0.1, 0.0, 0.0, 0.0, 
                            0.5, 0.4, 0.0, 0.0, 
                            0.0, 0.4, 0.7, 0.0, 
                            0.0, 0.0, 0.25, 0.4), 
                          nrow=4, ncol=4, byrow=TRUE)
  
  reproduction_matrix <- matrix(c(0.0, 0.0, 0.0, 2.6, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0), 
                                nrow=4, ncol=4, byrow=TRUE)
  
  # Clutch distribution
  clutch_distribution <- c(0.06, 0.08, 0.13, 0.15, 0.16, 0.18, 0.15, 0.06, 0.03)
  simulation_length <- 24
  
  # Initial conditions: 100 adults and 100 individuals spread across stages
  initial_adults <- c(0, 0, 0, 100)
  initial_spread <- c(25, 25, 25, 25)
  
  # Stochastic simulations
  population_adults <- stochastic_simulation(initial_adults, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length)
  population_spread <- stochastic_simulation(initial_spread, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length)
  
  # Plotting the results
  time <- 0:simulation_length
  plot(time, population_adults, type='l', col='blue', xlab='Time', ylab='Population Size', main='Stochastic Population Growth Comparison')
  lines(time, population_spread, type='l', col='red')
  legend('topright', legend=c('100 Adults', '25 Spread Across Stages'), col=c('blue', 'red'), lty=1)
  
  # Save the plot
  png("question_33.png")
  plot(time, population_adults, type='l', col='blue', xlab='Time', ylab='Population Size', main='Stochastic Population Growth Comparison')
  lines(time, population_spread, type='l', col='red')
  legend('topright', legend=c('100 Adults', '25 Spread Across Stages'), col=c('blue', 'red'), lty=1)
  dev.off()
  
  # Textual explanation
  return("The stochastic simulations show more variability and fluctuations compared to the deterministic simulations, which are typically smoother. This difference arises because the stochastic model incorporates random elements such as recruitment and offspring production, leading to inherent unpredictability in population size over time.")
}
# Questions 34 and 35 involve writing code elsewhere to run your simulations on the cluster
# Clear the workspace and turn off graphics
#rm(list = ls())
#graphics.off()

# Load necessary functions
#source("path/to/main.R")  # Update the path accordingly



# Question 36
question_36 <- function(){
  # Initialize a vector to count extinctions in four different scenarios
  count_extinctions <- rep(0, 4)
  for(simulation_set in 1:100){
    # Determine the current condition based on the simulation set number
    if(simulation_set <= 25) current_condition = 1
    else if(simulation_set <= 50) current_condition = 2
    else if(simulation_set <= 75) current_condition = 3
    else if(simulation_set <= 100) current_condition = 4
    
    # Construct the file name for loading results and load them
    file_name = paste("dem_sim_results_",simulation_set, ".rda",sep="")
    load(file_name)
    
    # Iterate over each simulation within the current set
    for(simulation_number in 1:150){
      simulation_data <- simu_results[simulation_number]
      last_value <- simulation_data[[1]][length(simulation_data[[1]])]  # Get the last step value
      
      # Increment the extinction count if the last value is zero
      if(last_value == 0){
        count_extinctions[current_condition] <- count_extinctions[current_condition] + 1
      }
    }
  }
  
  # Calculate the rate of extinction for each condition
  rate_of_extinction <- count_extinctions / rep(25 * 150, 4)
  print(count_extinctions)
  print(rate_of_extinction)
  
  # Assign names to the rates for better understanding
  names(rate_of_extinction) = c("Large 100 adults", "Small 10 adults", "Large 100 spread", "Small 10 spread")
  
  # Create and save a bar plot of the extinction rates
  png(filename="question_36.png", width = 600, height = 400)
  barplot(rate_of_extinction,
          xlab = "Initial Conditions", ylab = "Proportion of Extinction", ylim = c(0, max(rate_of_extinction) + 0.01),
          main = "Extinction Rate")
  Sys.sleep(0.1)
  dev.off()
}

question_36()
# Question 37
question_37 <- function() {
  # Initialize accumulators for average population size at each timestep for two scenarios in the stochastic model
  total_population_100_spread <- rep(0, 121)
  total_population_10_spread <- rep(0, 121)
  
  # Process data from stochastic simulations
  for (simulation_number in 51:100) {
    # Construct the file name and load the simulation data
    simulation_file <- paste("dem_sim_results_",simulation_number, ".rda",sep="")
    load(simulation_file)
    
    for (time_step in 1:150) {
      if (simulation_number <= 75) {
        total_population_100_spread <- total_population_100_spread + simu_results[time_step][[1]]
      } else {
        total_population_10_spread <- total_population_10_spread + simu_results[time_step][[1]]
      }
    }
  }
  
  # Calculate average population trends
  average_population_100_spread <- total_population_100_spread / (25 * 150)
  average_population_10_spread <- total_population_10_spread / (25 * 150)
  
  # Set up initial states for deterministic model
  initial_state_large_population <- state_initialise_spread(4, 100)
  initial_state_small_population <- state_initialise_spread(4, 10)
  
  # Define growth and reproduction matrices
  growth_matrix <- matrix(c(0.1, 0.0, 0.0, 0.0,
                            0.5, 0.4, 0.0, 0.0,
                            0.0, 0.4, 0.7, 0.0,
                            0.0, 0.0, 0.25, 0.4), nrow = 4, ncol = 4, byrow = TRUE)
  reproduction_matrix <- matrix(c(0.0, 0.0, 0.0, 2.6,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0,
                                  0.0, 0.0, 0.0, 0.0), nrow = 4, ncol = 4, byrow = TRUE)
  total_projection_matrix <- reproduction_matrix + growth_matrix
  
  # Run deterministic simulations
  deterministic_outcome_large_population <- deterministic_simulation(initial_state_large_population, total_projection_matrix, 120)
  deterministic_outcome_small_population <- deterministic_simulation(initial_state_small_population, total_projection_matrix, 120)
  
  # Calculate and plot deviations
  deviation_for_10_spread <- average_population_10_spread / deterministic_outcome_small_population
  deviation_for_100_spread <- average_population_100_spread / deterministic_outcome_large_population
  
  # Generate plot and save to file
  png(filename = "question_37.png", width = 600, height = 400)
  y_axis_range <- range(c(deviation_for_10_spread, deviation_for_100_spread)) + c(-0.01, 0.01)
  
  plot(1:121, deviation_for_10_spread, type = "l", col = "blue", ylim = y_axis_range,
       main = "Deviation of Stochastic Model from Deterministic Model", xlab = "Time Steps", ylab = "Deviation")
  lines(seq(1, 121), deviation_for_100_spread, type = "l", col = "black")
  abline(h = 1, lty = 2, col = "red")
  legend("topright", legend = c("Initial 10 population spread", "Initial 100 population spread"), col = c("blue", "black"), lty = 1)
  Sys.sleep(0.1)
  dev.off()
}

question_37()




# Challenge questions - these are optional, substantially harder, and a maximum
# of 14% is available for doing them. 

# Challenge question A
Challenge_A <- function() {
  # Set up parameters
  initial_conditions <- list(init_community_max(100), init_community_min(100))
  num_simulations <- 100  # Adjust based on computational resources
  total_generations <- 2000
  
  # Initialize arrays to store species richness
  species_richness_max <- matrix(0, nrow = num_simulations, ncol = total_generations)
  species_richness_min <- matrix(0, nrow = num_simulations, ncol = total_generations)
  
  # Run simulations
  for (i in 1:num_simulations) {
    for (initial_condition in initial_conditions) {
      community <- initial_condition
      for (j in 1:total_generations) {
        community <- neutral_generation_speciation(community, 0.1)
        if (identical(initial_condition, initial_conditions[[1]])) {
          species_richness_max[i, j] <- species_richness(community)
        } else {species_richness_min[i, j] <- species_richness(community)
        }
      }
    }
  }
  
  # Calculate mean and confidence intervals
  mean_richness_max <- colMeans(species_richness_max)
  mean_richness_min <- colMeans(species_richness_min)
  ci_max <- apply(species_richness_max, 2, function(x) qt(0.972, df=length(x)-1) * sd(x) / sqrt(length(x)))
  ci_min <- apply(species_richness_min, 2, function(x) qt(0.972, df=length(x)-1) * sd(x) / sqrt(length(x)))
  
  png(filename="Challenge_A_min", width = 600, height = 400)
  # plot your graph here
  plot(mean_richness_min, type = "l", col = "blue")
  lines(mean_richness_min + ci_min, col = "red", lty = 2)
  lines(mean_richness_min - ci_min, col = "red", lty = 2)
  Sys.sleep(0.1)
  dev.off()
  
  png(filename="Challenge_A_max", width = 600, height = 400)
  plot(mean_richness_max, type = "l", col = "blue")
  lines(mean_richness_max + ci_max, col = "blue", lty = 2)
  lines(mean_richness_max - ci_max, col = "blue", lty = 2)
  # plot your graph here
  Sys.sleep(0.1)
  dev.off()
}


# Challenge question B
Challenge_B<- function() {
  # Initialization
  speciePool <- vector()
  richnessData <- data.frame()
  
  for (initRichness in 1:100) {
    # Set up for each iteration
    richnessCount <- vector()
    meanRichness <- vector()
    
    # Create initial species pool
    speciePool <- sample(x = 1:100, size = initRichness, replace = FALSE)
    richnessCount[1] <- initRichness
    meanRichness[1] <- initRichness
    
    # Generate average richness over 200 generations
    for (gen in 1:200) {
      # Update species pool
      if (length(speciePool) > 1) {
        speciePool <- neutral_generation_speciation(speciePool, speciation_rate = 0.1)
      }
      # Compute richness
      richnessCount[gen + 1] <- species_richness(speciePool)
      meanRichness[gen + 1] <- sum(richnessCount) / (gen + 1)
    }
    
    # Compile data for each initial richness
    eachGenerationData <- data.frame(rep(initRichness, 201), seq(1, 201, by = 1), 
                                     meanRichness)
    colnames(eachGenerationData) <- c("InitRichness", "Generation", "MeanRichness")
    richnessData <- rbind(richnessData, eachGenerationData)
  }
  
  # Plotting
  png(filename="Challenege_B.png", width = 600, height = 400)
  plot_richness(richnessData)
  dev.off()
}

plot_richness <- function(dataFrame) {
  library(ggplot2)
  print(ggplot(dataFrame, aes(x = Generation, y = MeanRichness, group = InitRichness, color = InitRichness)) + 
          geom_line() +
          theme_classic() +
          labs(x = "Number of Generations", y = "Mean Species Richness",
               title = "Species Richness Over Time Across Different Initial Conditions"))
}

Challenge_B()



# Challenge question C
Challenge_C <- function() {
  community_sizes <- c(500, 1000, 2500, 5000)
  df <- data.frame(Generation = integer(), MeanRichness = numeric(), CommunitySize = factor())
  
  for (size in community_sizes) {
    all_richness <- numeric(0)
    
    for (iter in 1:25) {
      file_name <- paste("dem_sim_results_", iter, ".rda", sep = "")
      load(file_name)
      
      # Assuming 'time_series' contains species richness data
      all_richness <- c(all_richness, time_series)
    }
    
    # Calculate mean species richness for each generation and store in data frame
    mean_richness <- colMeans(matrix(all_richness, nrow = length(all_richness)/25, ncol = 25))
    generations <- 1:length(mean_richness)
    df <- rbind(df, data.frame(Generation = generations, MeanRichness = mean_richness, CommunitySize = as.factor(size)))
  }
  
  # Plotting the results in one figure using ggplot2
  png(filename="Challenge_C.png", width = 600, height = 400)
  library(ggplot2)
  ggplot(df, aes(x = Generation, y = MeanRichness, color = CommunitySize)) +
    geom_line() +
    theme_minimal() +
    labs(title = "Mean Species Richness Over Generations",
         x = "Generation",
         y = "Mean Species Richness",
         color = "Community Size")
  dev.off()
}
Challenge_C()




# Challenge question D
Challenge_D <- function() {
  speciationRate = 0.004131
  startTime <- proc.time()[3] # Start time recording
  species500 <- list() # Lists to store species data
  species1000 <- list()
  species2500 <- list()
  species5000 <- list()
  
  # A hypothetical function to aggregate data
  aggregateSpeciesData <- function(size, rate) {
    data <- runif(size, min = 0, max = rate) # Random uniform data as an example
    sortedData <- sort(data, decreasing = TRUE)
    return(sortedData)
  }
  
  for (i in 1:2500) { 
    species500[[i]] <- octaves(aggregateSpeciesData(500, speciationRate))
    species1000[[i]] <- octaves(aggregateSpeciesData(1000, speciationRate))
    species2500[[i]] <- octaves(aggregateSpeciesData(2500, speciationRate))
    species5000[[i]] <- octaves(aggregateSpeciesData(5000, speciationRate))
  }
  
  # Function to aggregate and average octave data
  aggregateData <- function(dataList) {
    aggregatedData <- dataList[[1]]
    for (i in 1:(length(dataList) - 1)) {
      aggregatedData <- sum_vect(aggregatedData, dataList[[i + 1]])
    }
    return(aggregatedData / length(dataList))
  }
  
  avgSpecies500 <- aggregateData(species500)
  avgSpecies1000 <- aggregateData(species1000)
  avgSpecies2500 <- aggregateData(species2500)
  avgSpecies5000 <- aggregateData(species5000)
  
  # Plotting the data
  png(filename="Challenge_D.png", width = 600, height = 400)
  par(mfrow = c(2,2), las = 2)
  barplot(avgSpecies500, main = "Community size = 500", ylab = "Species count", xlab = "Octave Classes")
  barplot(avgSpecies1000, main = "Community size = 1000", ylab = "Species count", xlab = "Octave Classes")
  barplot(avgSpecies2500, main = "Community size = 2500", ylab = "Species count", xlab = "Octave Classes")
  barplot(avgSpecies5000, main = "Community size = 5000", ylab = "Species count", xlab = "Octave Classes")
  dev.off()
  
  # Time and efficiency comparison
  elapsedTime <-  as.numeric(proc.time()[3] - startTime)
  comparisonMessage <- paste(elapsedTime, "seconds for Species Aggregation simulations (2500 repeats). Equivalent simulations on the cluster took 11.5 hours per 100 simulations. Aggregation simulations are faster as they only process essential data. Both approaches yield comparable results.")
  return(comparisonMessage)
}

Challenge_D()

# Challenge question E
mean_stage_simulation <- function(initial_state, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length) {
  recruitment_probability <- stochastic_recruitment(reproduction_matrix, clutch_distribution)
  mean_stage <- numeric(simulation_length + 1)
  mean_stage[1] <- sum(initial_state * seq_along(initial_state)) / sum(initial_state)
  state <- initial_state
  
  for (time_step in 1:simulation_length) {
    if (sum(state) == 0) {
      mean_stage[(time_step + 1):(simulation_length + 1)] <- 0
      break
    }
    state <- stochastic_step(state, growth_matrix, reproduction_matrix, clutch_distribution, recruitment_probability)
    mean_stage[time_step + 1] <- sum(state * seq_along(state)) / sum(state)
  }
  
  return(mean_stage)
}

Challenge_E <- function() {
  # Define the growth matrix and reproduction matrix as in Question 25
  growth_matrix <- matrix(c(0.1, 0.0, 0.0, 0.0, 
                            0.5, 0.4, 0.0, 0.0, 
                            0.0, 0.4, 0.7, 0.0, 
                            0.0, 0.0, 0.25, 0.4), 
                          nrow=4, ncol=4, byrow=TRUE)
  
  reproduction_matrix <- matrix(c(0.0, 0.0, 0.0, 2.6, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0, 
                                  0.0, 0.0, 0.0, 0.0), 
                                nrow=4, ncol=4, byrow=TRUE)
  
  # Clutch distribution as provided
  clutch_distribution <- c(0.06, 0.08, 0.13, 0.15, 0.16, 0.18, 0.15, 0.06, 0.03)
  
  # Simulation length
  simulation_length <- 24
  
  # Initial conditions: 100 adults and 100 individuals spread across stages
  initial_adults <- c(0, 0, 0, 100)
  initial_spread <- c(25, 25, 25, 25)
  
  # Run the mean stage simulations
  mean_stage_adults <- mean_stage_simulation(initial_adults, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length)
  mean_stage_spread <- mean_stage_simulation(initial_spread, growth_matrix, reproduction_matrix, clutch_distribution, simulation_length)
  
  # Plotting
  time <- 0:simulation_length
  plot(time, mean_stage_adults, type='l', col='blue', xlab='Time', ylab='Mean Life Stage', main='Mean Life Stage Over Time')
  lines(time, mean_stage_spread, type='l', col='red')
  legend('topright', legend=c('Adults Only', 'Spread Across Stages'), col=c('blue', 'red'), lty=1)
  
  # Save the plot
  png(filename="Challenge_E.png", width = 600, height = 400)
  plot(time, mean_stage_adults, type='l', col='blue', xlab='Time', ylab='Mean Life Stage')
  lines(time, mean_stage_spread, type='l', col='red')
  legend('topright', legend=c('Adults Only', 'Spread Across Stages'), col=c('blue', 'red'), lty=1)
  dev.off()
  
  # Textual explanation
  return("In the initial stages of the simulation, the graph of mean life stage for the population starting with only adults shows a decrease, while the graph for the mixed initial population shows less change. This is because the adult-only population initially produces a large number of offspring (lower life stages), reducing the mean life stage rapidly. The mixed population already includes individuals in lower life stages, leading to a more stable mean life stage initially.")
}
Challenge_E()


# Challenge question F
Challenge_F <- function() {
  # Load necessary libraries and functions
  library(ggplot2)
  
  # Set parameters
  simulation_size <- 120
  num_simulations_per_condition <- 100
  initial_conditions <- c("small adult", "large adult", "small mixed", "large mixed")
  
  # Initialize data frame
  population_size_df <- data.frame()
  
  # Loop through each initial condition
  for (initial_condition in initial_conditions) {
    # Loop through each simulation for the current initial condition
    for (simulation_number in 1:num_simulations_per_condition) {
      # Load results from the corresponding file
      result_file <- sprintf("dem_sim_results_%d.rda",simulation_number)
      load(result_file)
      
      # Extract population size time series for the current simulation
      population_size <- result_file[[1]]
      
      # Create a data frame for the current simulation
      simulation_df <- data.frame(
        simulation_number = rep(simulation_number, simulation_size + 1),
        initial_condition = rep(initial_condition, simulation_size + 1),
        time_step = 0:simulation_size,
        population_size = as.vector(population_size)
      )
      
      # Append the data frame to the main population_size_df
      population_size_df <- rbind(population_size_df, simulation_df)
    }
  }
  
  # Create the plot using ggplot2
  p <- ggplot(population_size_df, aes(x = time_step, y = population_size, group = simulation_number, color = initial_condition)) +
    geom_line(alpha = 0.1) +
    labs(title = "Population Size Time Series - Challenge F",
         x = "Time Step",
         y = "Population Size") +
    theme_minimal()
  
  # Save the plot using png
  png("Challenge_F.png", width = 600, height = 400)
  print(p)
  dev.off()
  
  # Return the population_size_df data frame
  return(population_size_df)
}

# Call the function
Challenge_F()
