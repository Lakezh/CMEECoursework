someOperation <- function(v) {
    if (sum(v) > 0) {
        reutrn (v * 100)
    } else {
        reutrn(v)
    }
}

M <- matrix(rnorm(100), 10, 10)
print (apply(M, 1, someOperation))