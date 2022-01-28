using Turing
using Distributions
using Random
using LinearAlgebra
using StatsPlots
using MCMCChains
using DataFrames
using Printf

# geberate moc data from logit model
n = 1_000
num_var = 2
Random.seed!(99)
X = rand(Uniform(-sqrt(3.0), 2.0*sqrt(3.0)), n, num_var) # case-specific variables Z_i Matrix
X = [ones(n) X]
β = Vector([0.0, 0.5, -0.5]) # true coefficients
# ポアソン分布の平均をモデル化
λ = exp.(X*β)
y = rand.(Poisson.(λ)) # observable choice 0 or 1

# set prior, β ~ MvNormal(β0, A0), no σ²
n, k = size(X)
β0 = zeros(k)
A0 = 0.01 * I

@model function poisson_regression_model(X, y)
    # Set the priors only for β.
    β ~ MvNormal(β0, inv(A0))

    # Write the likelihood, can't vectorize?
    n = size(X, 1)
    for i = 1:n
        y[i] ~ Poisson(exp(X[i, :]' * β))
    end
end

# Draw random number from posterior.
model = poisson_regression_model(X, y)

n_draws = 5_000
# n_chains = 4
n_chains = 1
n_tune = 1_000
chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains; discard_adapt=false)
plot(chn)
# chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains)[n_tune:n_draws, :, :]
β_chn = chn[n_tune+1:n_draws, :, :] # delete initial and exclude odd (hamiltonian bluh bluh) chain 
β_chn = group(β_chn, :β)

summarize(β_chn)
quantile(β_chn)
plot(β_chn)

# Marginal effect ... interpret β just like as ln(y) = βx model (equivalent with λ = exp(x))