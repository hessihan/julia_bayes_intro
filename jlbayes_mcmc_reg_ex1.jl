# estimate with MCMC (Turing) for conjugate prior
# compare with jlbayes_conjugate_regression.jl

using Turing
using Distributions
using Random
using LinearAlgebra
using StatsPlots
using Optim

# Generate moc data
n = 50
dim = 1
Random.seed!(99)
u = rand(Normal(0, 0.7), n) # true value σ² = 0.49
x = rand(Uniform(-sqrt(3.0), sqrt(3.0)), n, dim) # x generated from Uniform[-√3, √3]
X = [ones(n) x] # dependent variable matrix with ones
β = Vector(1:1:dim+1) # true coefficients
y = X*β + u # true data generating process

# set prior (conjugate prior for camparison with analytic result)
# p.78 eq (3.32)
k = size(X)[2]
β0 = zeros(k)
τ0 = 0.2
A0 = τ0 * I # Identity matrix
ν0 = 5.0
λ0 = 7.0
# H0 = (λ0 / ν0) * inv(A0) # scale matrix for marginal prior of multivariate coefficient (mv Tdist)
# h0 = diag(sqrt(Matrix(H0, k, k))) # scale params for marginal prior of single coefficient (T dist) H0の対角成分
sd0 = diag(sqrt(Matrix(A0, k, k))) # the parameter for conditional prior for β|σ² ~ N_k(β0, σ²*inv(A0)), (inv(A0) part)

@model function linear_regression(X, y)
    # conjugate priors
    # https://turing.ml/dev/tutorials/05-linear-regression/
    
    # Set variance priors
    σ² ~ InverseGamma(ν0/2, λ0/2)

    # Set the priors on our coefficients.
    nfeatures = size(X, 2)
    β ~ MvNormal(β0, σ² * inv(A0))

    # Write likelihood.
    y ~ MvNormal(X * β, σ² * I)
end

# Draw random number from posterior.
model = linear_regression(X, y)
n_draws = 5_000
n_chains = 4
n_tune = 1_000
# sample(model, sampler, parallel_type, n, n_chains)
# chain = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains, discard_initial=n_tune)
# How to implement burn-in / warmup? --> discard initial value automatically. ; discard_adapt=false to turn off
chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains)
chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains; discard_adapt=false)

# Chain info.
typeof(chn)
size(chn)
describe(chn)
summarize(chn)
quantile(chn)
plot(chn)

# Indexing a Chains object.
β1_chn = chn["β[1]"]
β2_chn = chn[Symbol("β[2]")]
σ²_chn = chn[:σ²]
all_β_chn = group(chn, :β) # get subset of all parameters include "β"

typeof(β1_chn)
β1_chn[iter = 1000:5000, chain = 1]
mean(β1_chn)
std(β1_chn)

# Monte Carlo Standard Error
mcse(θ_chn) = sqrt((1 / (length(θ_chn) * (length(θ_chn) - 1))) * sum((θ_chn .- mean(θ_chn)).^2))

[β1_chn, β2_chn, σ²_chn]

mcse.([β1_chn, β2_chn, σ²_chn])

# without burn-in
mean(β1_chn[iter = 1:5000])

histogram(β1_chn[:, 1])
histogram!(β1_chn[:, 2])
histogram!(β1_chn[:, 3])
histogram!(β1_chn[:, 4])

# mode (MLE and MAP) estimates (Optimization with Optim.jl) as an option (continuous only)
# Note that loading Optim explicitly is required for mode estimation to function,
# as Turing does not load the opimization suite unless Optim is loaded as well.

# Generate a MLE estimate (default optimizer is LBFGS optimizer).
mle_estimate = optimize(model, MLE())
# Use Newton
# mle_estimate = optimize(model, MLE(), Newton())

# Generate a MAP estimate (default optimizer is LBFGS optimizer).
map_estimate = optimize(model, MAP())

# Analyze mode estimate
# Import StatsBase to use it's statistical methods.
# using StatsBase

# Print out the coefficient table.
# coeftable(mle_estimate)

# Analytical Solution
inv(X' * X) * X' * y