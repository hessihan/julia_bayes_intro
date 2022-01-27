using Random, Plots, StatsPlots, Distributions, DataFrames, Printf
# Define calculation of posterior
"""
Posterior Distribution (Gamma dist) for Poisson parameter λ

    Inputs
    ------
    data: data (0 or 1)
    α0: shape parameter of prior (Gamma dist)
    β0: inversed scale parameter of prior (Gamma dist)
    prob: credible interval probability (0 < prob < 1)

    Outputs
    -------
    results: posterior summary stats table (DataFrames)
    α_star: shape parameter of posterior (Gamma dist)
    β: inversed scale parameter (1 / θ) of posterior (Gamma dist)

"""
function poisson_stats(data, α0, β0, prob)
    # posterior
    n = size(data)[1]
    sum_data = sum(data)
    α_star = sum_data + α0
    β_star = n + β0
    θ_star = 1.0 / β_star
    d_posterior = Gamma(α_star, θ_star)
    # mean
    mean_λ = mean(d_posterior)
    # median
    median_λ = median(d_posterior)
    # mode
    mode_λ = mode(d_posterior)
    # Analytic Mode for Beta Dist
    # (a - 1.0) / (a + b - 2.0)
    # std
    sd_λ = std(d_posterior)
    # credible interval
    # ci upper boud
    ci_λ_upper = quantile(d_posterior, prob - (1-prob)/2)
    # ci lower boud
    ci_λ_lower = quantile(d_posterior, (1-prob)/2)
    ci_λ = [ci_λ_lower, ci_λ_upper]

    posterior_stats = reshape([mean_λ, median_λ, mode_λ, sd_λ, ci_λ], 1, 5)
    stats_string = ["Mean", "Median", "Mode", "SD", "CI"]
    results = DataFrame(posterior_stats, stats_string)
    return results, α_star, β_star
end

# Generate moc datas
λ = 3.0
n = 50
Random.seed!(99)
data = rand(Poisson(lam), n)

# Set prior
α0 = 1.0
β0 = 1.0

# Calculate posterior statistics
prob = 0.95
results, α_star, β_star = poisson_stats(data, α0, β0, prob)
print(results)

# Visualize Posterior
θ0, θ_star = 1.0/β0, 1.0/β_star
plot(Gamma(α0, θ0), xlims=(0, 6), ylims=(0, 1.75), xlabel="λ", ylabel="Probability Density", label=@sprintf("prior: Gamma(%.1f, %.1f)",α0, θ0), linestyle=:dash)
plot!(Gamma(α_star, θ_star), label=@sprintf("prior: Gamma(%.1f, %.2f)",α_star, θ_star), linestyle=:solid)