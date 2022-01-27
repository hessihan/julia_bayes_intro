using Random, Plots, StatsPlots, Distributions, DataFrames, Printf, SpecialFunctions
# Define student t distribution with loc and scale parameter
# https://juliastats.org/Distributions.jl/v0.21/univariate/#Distributions.LocationScale
# https://discourse.julialang.org/t/scale-and-location-to-tdist-in-distributions/74360
# p(x|\nu, \mu, \sigma^2) = \frac{\Gamma(\frac{\nu+1}{2})}{\Gamma(\frac{\nu}{2})\sqrt{\pi\nu\sigma^2}}\left[1 + \frac{(x-\mu)^2}{\nu\sigma^2}\right]^{-\frac{\nu+1}{2}}
"""
t-distribution probability density function from definition.
"""
function tdist_from_def(x, ν, μ, σ)
    return (gamma((ν+1)/2) / (gamma(ν/2) * sqrt(pi*ν*σ^2)) * (1 + (x-μ)^2/(ν*σ^2))^(-(ν+1)/2))
end

"""
t-distribution probability density function usinf Distributions.LocationScale
"""
function tdist_from_LocationScale(ν, μ, σ)
    return LocationScale(μ, σ, TDist(ν))
end

# tdist_from_def(5, 7, 10, 0.7)
# pdf(tdist_from_LocationScale(7, 10, 0.7), 5)

# Define calculation of posterior
"""
Conjugate Priors for Normal Dist parameters (μ|σ^2 ~ Normal, σ^2 ~ InverseGamma)

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
    β: inversed scale parameter (1/θ) of posterior (Gamma dist)

"""
function gaussian_stats(data, μ0, n0, ν0, λ0, prob)
    # posterior
    n = size(data)[1]
    mean_data = mean(data)
    ssd_data = (n-1) * var(data) # var is unbiased variance scaled by n-1
    n_star = n + n0
    μ_star = (n * mean_data + n0 * μ0) / n_star
    ν_star = n + ν0
    λ_star = ssd_data + (n * n0 / n_star) * (μ0 - mean_data)^2 + λ0
    τ_star = sqrt(λ_star / (ν_star * n_star))

    posterior_σ_sq = InverseGamma(ν_star/2, λ_star/2)
    marginal_posterior_μ = LocationScale(μ_star, τ_star, TDist(ν_star))

    sd_μ = std(marginal_posterior_μ)
    ci_μ_upper = quantile(marginal_posterior_μ, prob - (1-prob)/2)
    ci_μ_lower = quantile(marginal_posterior_μ, (1-prob)/2)
    ci_μ = [ci_μ_lower, ci_μ_upper]
    mean_σ_sq = mean(posterior_σ_sq)
    mode_σ_sq = mode(posterior_σ_sq)
    median_σ_sq = median(posterior_σ_sq)
    sd_σ_sq = std(posterior_σ_sq)
    ci_σ_sq_upper = quantile(posterior_σ_sq, prob - (1-prob)/2)
    ci_σ_sq_lower = quantile(posterior_σ_sq, (1-prob)/2)
    ci_σ_sq = [ci_σ_sq_lower, ci_σ_sq_upper]

    stats_μ = reshape([μ_star, μ_star, μ_star, sd_μ, ci_μ], 1, 5)
    stats_σ_sq = reshape([mean_σ_sq, median_σ_sq, mode_σ_sq, sd_σ_sq, ci_σ_sq], 1, 5)

    stats_string = ["Mean", "Median", "Mode", "SD", "CI"]
    results = DataFrame([stats_μ; stats_σ_sq], stats_string)
    return results, marginal_posterior_μ, posterior_σ_sq
end

# Generate moc data
μ = 1.0
σ = 2.0
n = 50
Random.seed!(99)
data = rand(Normal(μ, σ), n)
# histogram(data, bins=20)

# Set prior
μ0 = 0.0
n0 = 0.2
ν0 = 5.0
λ0 = 7.0
τ0 = sqrt(λ0 / (ν0 * n0))

prior_σ_sq = InverseGamma(ν0/2, λ0/2)
marginal_prior_μ = LocationScale(μ0, τ0, TDist(ν0))

# Calculate posterior statistics
prob = 0.95
results, marginal_posterior_μ, posterior_σ_sq = gaussian_stats(data, μ0, n0, ν0, λ0, prob)
print(results)

# Visualize Posterior
l = @layout [grid(1, 2)]

p1 = plot(
    marginal_prior_μ, xlims=(-6, 6), ylims=(0, 1.55), xlabel="μ", ylabel="Probability Density", 
    label=@sprintf("prior: T(ν=%.1f, μ=%.1f, σ^2=%.1f)", params(params(marginal_prior_μ)[3])[1], params(marginal_prior_μ)[1], params(marginal_prior_μ)[2]), linestyle=:dash)
plot!(marginal_posterior_μ, label=@sprintf("posterior: T(ν=%.1f, μ=%.1f, σ^2=%.1f)", params(params(marginal_posterior_μ)[3])[1], params(marginal_posterior_μ)[1], params(marginal_posterior_μ)[2]), linestyle=:solid)

p2 = plot(
    prior_σ_sq, xlims=(0, 10), ylims=(0, 0.65), xlabel="σ^2", ylabel="Probability Density", 
    label=@sprintf("prior: InverseGamma(α=%.1f, θ=%.1f)", params(prior_σ_sq)[1], params(prior_σ_sq)[2]), linestyle=:dash)
plot!(posterior_σ_sq, label=@sprintf("posterior: InverseGamma(α=%.1f, θ=%.1f)", params(posterior_σ_sq)[1], params(posterior_σ_sq)[2]), linestyle=:solid)

plot(p1, p2, laypout=l)
# μの推定ヘタクソじゃね?