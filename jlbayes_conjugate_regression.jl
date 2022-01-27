# https://mrandri19.github.io/2020/09/28/bayesian-linear-regression-with-conjugate-priors.html
# https://qiita.com/haru1843/items/3956dab2fd0d448cd02b
# https://discourse.julialang.org/t/solving-ax-b-for-large-matrix-dimesnions-efficiently-in-julia/51504
# https://stackoverflow.com/questions/57270276/identity-matrix-in-julia

# bayesian inferense (using conjugate prior analyticaly) of coefficients and variance in error term for linear regression model

using Random, Plots, StatsPlots, Distributions, LinearAlgebra, DataFrames, Printf, LaTeXStrings

"""
Conjugate Priors for linear regression model (β|σ^2 ~ MvNormal, σ^2 ~ InverseGamma)

    Inputs
    ------
    y:    data, independent variable (n vector)
    X:    data, dependent variables (n * k matrix)
    β0:   mean of conditional prior (multivariate normal β0|σ^2 ~ MvNormal) for coefficients
    A0:   precision matrix (inverse of Σ) of conditional prior for coefficients
    ν0:   shape parameter of prior for variance of error term
    λ0:   scale parameter of prior for variance of error term
    prob: credible interval probability (0 < prob < 1)

    Outputs
    -------
    results: posterior summary stats table (DataFrames)
    marginal_posterior_β: vector of marginal posterior for each coefficients (single β|D ~ TDist)
    posterior_σ2: posterior for variance of error term (σ^2|D ~ InverseGamma)
"""
function regression_stats(y, X, β0, A0, ν0, λ0, prob)
    XX = X' * X
    Xy = X' * y
    # @time XX \ Xy # fast and efficient
    # @time inv(XX) * Xy
    β_ols = XX \ Xy # OLS estimator
    A_star = XX + A0
    β_star = A_star \ (Xy + A0 * β0)
    C_star = inv(inv(XX) + inv(A0))
    ν_star = size(y)[1] + ν0
    ssr = sum((y - X * β_ols).^2) # SSR sum of squared residual 残差二乗和
    # @time (β0 - β_ols)' * C_star * (β0 - β_ols)
    # @time dot(β0 - β_ols, C_star, β0 - β_ols) # faster
    λ_star = ssr + dot(β0 - β_ols, C_star, β0 - β_ols) + λ0
    H_star = (λ_star / ν_star) * inv(A_star)
    h_star = diag(sqrt(H_star))

    marginal_posterior_β = LocationScale.(β_star, h_star, TDist(ν_star)) # broadcasting through coefficients
    sd_β = std.(marginal_posterior_β)
    ci_β_upper = quantile.(marginal_posterior_β, prob - (1-prob)/2)
    ci_β_lower = quantile.(marginal_posterior_β, (1-prob)/2)
    # ci_β = [ci_β_lower, ci_β_upper]
    stats_β = [β_star β_star β_star sd_β ci_β_lower ci_β_upper]

    posterior_σ2 = InverseGamma(ν_star/2, λ_star/2)
    mean_σ2 = mean(posterior_σ2)
    median_σ2 = median(posterior_σ2)
    mode_σ2 = mode(posterior_σ2)
    sd_σ2 = std(posterior_σ2)
    ci_σ2_upper = quantile(posterior_σ2, prob - (1-prob)/2)
    ci_σ2_lower = quantile(posterior_σ2, (1-prob)/2)
    # ci_σ2 = [ci_σ2_lower, ci_σ2_upper]
    stats_σ2 = [mean_σ2 median_σ2 mode_σ2 sd_σ2 ci_σ2_lower ci_σ2_upper]
    stats = [stats_β; stats_σ2]
    stats_string = ["Mean", "Median", "Mode", "SD", @sprintf("CI(%d)_Lower", prob*100), @sprintf("CI(%d)_Upper", prob*100)]
    results = DataFrame(stats, stats_string)
    return results, marginal_posterior_β, posterior_σ2, β_ols
end

# Generate moc data
n = 50
dim = 1
Random.seed!(99)
u = rand(Normal(0, 0.7), n) # true value σ^2 = 0.49
x = rand(Uniform(-sqrt(3.0), sqrt(3.0)), n, dim) # x generated from Uniform[-√3, √3]
X = [ones(n) x] # dependent variable matrix with ones
β = Vector(1:1:dim+1) # true coefficients
y = X*β + u # true data generating process


# set prior
k = size(X, 2)
β0 = zeros(k)
τ0 = 0.2 # no ridge penalty ... τ0 = Inf ?

# τ0 = 0.001
# τ0 = 0.01
# τ0 = 0.1
# τ0 = 1
# τ0 = 10
# τ0 = 100
# τ0 = 1000

A0 = τ0 * I # Identity matrix
ν0 = 5.0
λ0 = 7.0
H0 = (λ0 / ν0) * inv(A0) # scale matrix for marginal prior of multivariate coefficient (mv Tdist)
h0 = diag(sqrt(Matrix(H0, k, k))) # scale params for marginal prior of single coefficient (T dist) H0の対角成分
marginal_prior_β = LocationScale.(β0, h0, TDist(ν0)) # broadcasting through coefficients
prior_σ2 = InverseGamma(ν0/2, λ0/2)

# calculate posterior
prob = 0.95
results, marginal_posterior_β, posterior_σ2, β_ols = regression_stats(y, X, β0, A0, ν0, λ0, prob)
print("$results\n")

# Visualize Posterior
plots = []

for i = 1:k
    label_prior = @sprintf("marginal prior: T(ν=%.1f, μ=%.1f, σ^2=%.1f)", params(params(marginal_prior_β[i])[3])[1], params(marginal_prior_β[i])[1], params(marginal_prior_β[i])[2])
    p = plot(marginal_prior_β[i], linestyle=:dash, label=label_prior, xlims=(0.0, 3.0), ylims=(0.0, 4.0), xlabel=@sprintf("β%d", i), ylabel="Probability Density")
    label_posterior = @sprintf("marginal posterior: T(ν=%.1f, μ=%.1f, σ^2=%.1f)", params(params(marginal_posterior_β[i])[3])[1], params(marginal_posterior_β[i])[1], params(marginal_posterior_β[i])[2])
    plot!(marginal_posterior_β[i], linestyle=:solid, label=label_posterior)
    push!(plots, p)
end

label_prior = @sprintf("prior: InverseGamma(α=%.1f, θ=%.1f)", params(prior_σ2)[1], params(prior_σ2)[2])
p = plot(prior_σ2, linestyle=:dash, label=label_prior, xlims=(0.0, 3.0), ylims=(0.0, 4.0), xlabel=L"\sigma^2", ylabel="Probability Density")
label_posterior = @sprintf("prior: InverseGamma(α=%.1f, θ=%.1f)", params(posterior_σ2)[1], params(posterior_σ2)[2])
plot!(posterior_σ2, linestyle=:solid, label=label_posterior)
push!(plots, p)

plot(plots..., layout=(1, k+1))

# plot ols
scatter(x, y, label="data")
# scatter(x, y, xlims=(-0.1, 0.1), ylims=(0.9, 1.5))
plot!(x, X * β, label=(@sprintf("True: y = %.d + %.dx", β[1], β[2])))
plot!(x, X * β_ols, label=(@sprintf("OLS: y = %.5f + %.5fx", β_ols[1], β_ols[2])))
β_star = results[:, "Mean"][1:k]
plot!(x, X * β_star, label=(@sprintf("Bayes: y = %.5f + %.5fx (τ0=%.2f, Ridge)", β_star[1], β_star[2], τ0)))

# significance test on coefficients H0: βj = 0, H1: βj ≠ 0
# evaluate with SDDR
"""
calculate SDDR (compare density of posterior and prior on particular value)
# https://docs.julialang.org/en/v1/manual/methods/
# https://stackoverflow.com/questions/61088294/declaring-the-name-of-argument-when-invoking-a-function

    Inputs
    ------
    β_h0: test value, (default is 0)
    posterior
    prior

    Output
    ------
    sddr
"""
function sddr(posterior, prior, ;null_hypothesis::Float64=0.0)
    return pdf(posterior, null_hypothesis) / pdf(prior, null_hypothesis)
end

print(log10.(sddr.(marginal_posterior_β, marginal_prior_β, null_hypothesis=0.0)))