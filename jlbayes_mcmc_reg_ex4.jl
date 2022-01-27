# estimate with MCMC (Turing) for prior (Laplas and half Cauchy
# https://juliastats.org/Distributions.jl/stable/univariate/#Distributions.Laplace
# half-Cauchy not implemented yet
# https://github.com/JuliaStats/Distributions.jl/issues/124
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

# set prior (Laplace for β, half-Cauchy for σ²)