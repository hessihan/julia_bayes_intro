using Random, Plots, DataFrames
# Bayesian Inference for Bernoulli parameter (inference success probability)
# Define HPD (highest posterior density) interval
"""
HPD interval for beta dist
    
    Inputs
    ------
    ci0:  initial value of HPD interval
    α:    parameter 1 of Beta dist
    β:    parameter 2 of Beta dist
    prob: probabilty of HPD interval
    
    Outputs
    -------
    HPD interval

"""
function beta_hpdi(ci0, α, β, prob)
    return nothing
end

# Define calculation of posterior for Bernoulli parameter
"""
Posterior Distribution (Beta dist) for Bernoulli parameter

    Inputs
    ------
    data: data (0 or 1)
    a0: parameter 1 of prior (Beta dist)
    b0: parameter 2 of prior (Beta dist)
    prob: interval probability (0 < prob < 1)

    Outputs
    -------
    results: posterior summary stats table (DataFrames)
    a: parameter 1 of posterior (Beta dist)
    b: parameter 2 of posterior (Beta dist)

"""
function bernoulli_stats()
    return nothing
end

# Generate moc data from Bernoulli
p = 0.25
n = 50
Random.seed!(99)
data = rand(Bernoulli(p), n)

size(data)[1]


# Calculate posterior statistics
