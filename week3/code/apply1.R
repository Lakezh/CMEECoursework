M <- matrix(rnorm(100), 10, 10)
RowMeans <- apply(M, 1, mean)
print(RowMeans)

ColMeans <- apply(M, 2, mean)

## Now the variance
RowVars <- apply(M, 1, var)
print (RowVars)

## By column
ColMeans <- apply(M, 2, mean)
print (ColMeans)