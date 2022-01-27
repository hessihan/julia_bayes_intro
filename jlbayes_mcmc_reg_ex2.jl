# estimate with MCMC (Turing) for prior (not conjugate)
using Turing
using Distributions
using Random
using LinearAlgebra
using StatsPlots

# Generate moc data
n = 50
dim = 1
Random.seed!(99)
u = rand(Normal(0, 0.7), n) # true value σ² = 0.49
x = rand(Uniform(-sqrt(3.0), sqrt(3.0)), n, dim) # x generated from Uniform[-√3, √3]
X = [ones(n) x] # dependent variable matrix with ones
β = Vector(1:1:dim+1) # true coefficients
y = X*β + u # true data generating process

# set prior (normal for β, InverseGamma for σ²)
k = size(X)[2]
β0 = zeros(k)
τ0 = 0.2
A0 = τ0 * I # Identity matrix
ν0 = 5.0
λ0 = 7.0
# H0 = (λ0 / ν0) * inv(A0) # scale matrix for marginal prior of multivariate coefficient (mv Tdist)
# h0 = diag(sqrt(Matrix(H0, k, k))) # scale params for marginal prior of single coefficient (T dist) H0の対角成分
sd0 = diag(sqrt(Matrix(A0, k, k))) # the parameter for conditional prior for β|σ² ~ N_k(β0, σ²*inv(A0)), (inv(A0) part)

@model function linear_regression_normal_invgamma(X, y)    
    # Set variance priors
    σ² ~ InverseGamma(ν0/2, λ0/2)

    # Set the priors on our coefficients. (No conditions about σ²)
    nfeatures = size(X, 2)
    β ~ MvNormal(β0, inv(A0))

    # Write likelihood.
    y ~ MvNormal(X * β, σ² * I)
end

# Draw random number from posterior.
model = linear_regression_normal_invgamma(X, y)

n_draws = 5_000
n_chains = 4
n_tune = 1_000
chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains)
summarize(chn)
quantile(chn)

plot(chn)

# Analytical Solution
inv(X' * X) * X' * y