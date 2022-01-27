# https://turing.ml/dev/docs/using-turing/quick-start
# https://turing.ml/dev/tutorials/
# https://storopoli.io/Bayesian-Julia/pages/4_Turing/

# Immport libraries.
using Turing, StatsPlots, Random

# Set the true probability of heads in a coin.
p_true = 0.5

# Iterate from having seen 0 observations to 100 observations.
Ns = 0:100

# Draw data from a Bernoulli distribution, i.e. draw heads or tails.
Random.seed!(12)
data = rand(Bernoulli(p_true), last(Ns))

# Declare our Turing model.
@model function coinflip(y)
    # Our prior belief about the probability of heads in a coin.
    p ~ Beta(1, 1)

    # The number pf observations.
    N = length(y)
    for n in 1:N
        # Heads or tails of a coin are drawn from a Bernoulli distribution.
        y[n] ~ Bernoulli(p)
    end
end

# Setting of the Hamiltnian Monte Carlo (HMC) sampler.
iterations = 1000
ϵ = 0.05
τ = 10

# Start sampling.
chain = sample(coinflip(data), HMC(ϵ, τ), iterations)

# Plot a summary of the sampling process for the parameter p, i.e. the probability of heads in a coin.
histogram(chain[:p])


# the order of setting priors is important
rnd = rand(Normal(10, 4), 100)
histogram(rnd)

@model ordered_prior(x) = begin
    # Assumptions
    σ ~ InverseGamma(2,3)
    μ ~ Normal(0,sqrt(σ))
    # Observations
    x ~ Normal(μ, sqrt(σ))
end
chain = sample(ordered_prior(rnd), NUTS(0.65), 3_000)
summarize(chain)
plot(chain)

@model missordered_prior(x) = begin
    # Assumptions
    μ ~ Normal(0,sqrt(σ)) # --> UndefVarError: σ not defined
    σ ~ InverseGamma(2,3)
    # Observations
    x ~ Normal(μ, sqrt(σ))
end
chain = sample(missordered_prior(rnd), NUTS(0.65), 3_000)