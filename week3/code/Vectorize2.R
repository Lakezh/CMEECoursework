# Runs the stochastic Ricker equation with gaussian fluctuations

rm(list = ls())

stochrick <- function(p0 = runif(1000, 0.5, 1.5), r = 1.2, K = 1, sigma = 0.2,numyears = 100)
{

  N <- matrix(NA, numyears, length(p0))  #initialize empty matrix

  N[1, ] <- p0

  for (pop in 1:length(p0)) { #loop through the populations

    for (yr in 2:numyears){ #for each pop, loop through the years

      N[yr, pop] <- N[yr-1, pop] * exp(r * (1 - N[yr - 1, pop] / K) + rnorm(1, 0, sigma)) # add one fluctuation from normal distribution
    
     }
  
  }
 return(N)

}
system.time(stochrick())

# Now write another function called stochrickvect that vectorizes the above to
# the extent possible, with improved performance: 

# print("Vectorized Stochastic Ricker takes:")
# print(system.time(res2<-stochrickvect()))

rm(list = ls())

stochrickvect <- function(p0 = runif(1000, 0.5, 1.5), r = 1.2, K = 1, sigma = 0.2,numyears = 100)
{

  N <- matrix(NA, numyears, length(p0))#initialize empty matrix

  N[1, ] <- p0
#generate random fluctuations for all years at once
  fluctuations <- matrix(rnorm(numyears * length(p0), mean = 0, sd = sigma), nrow = numyears)
    for (yr in 2:numyears){ #for each pop, loop through the years

      N[yr, ] <- N[yr-1, ] * exp(r * (1 - N[yr - 1, ] / K) + fluctuations[yr, ]) 
      #add one fluctuation from normal distribution
    
     }
  
 return(N)

}

print("Vectorized Stochastic Ricker takes:")
print(system.time(res2<-stochrickvect()))
