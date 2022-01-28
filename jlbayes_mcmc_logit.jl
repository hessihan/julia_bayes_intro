# logit
# https://discourse.julialang.org/t/bayesian-logistic-regression-with-turing-jl/60105/9

using Turing
using Distributions
using Random
using LinearAlgebra
using StatsPlots
using MCMCChains
using DataFrames
using Printf
# using Plots # reesport
# using Statistics # reesport

using StatsFuns: logistic, logit

# logistic function # 1 / (1 + exp(-x)), y = logistic(x'β) = 1 / (1 + exp(-x'β)) = exp(x'β) / (1 + exp(x'β))
plot(logistic.(Vector(-10:0.1:10)))
# logit function # log(x / (1 - x)), logit(y) = inversed logistic(y) = log(y / (1 - y)) = x'β
plot(logit.(Vector(0:0.01:1)))

# geberate moc data from logit model
n = 1_000
num_var = 2
Random.seed!(99)
X = rand(Uniform(-sqrt(3.0), 2.0*sqrt(3.0)), n, num_var) # case-specific variables Z_i Matrix
X = [ones(n) X]
β = Vector([0.0, 0.5, -0.5]) # true coefficients
# V = X * β # representative utility, # utility (latent variable) y_star = U = Xβ + ϵ = V + ϵ
# ロジットのロジスティック関数内に誤差項を含めるのは誤り。ロジスティック分布すること自体が確率変数.
p = logistic.(X * β) # probability (する確率)
y = rand.(Bernoulli.(p)) # observable choice 0 or 1

# set prior, β ~ MvNormal(β0, A0), no σ²
n, k = size(X)
β0 = zeros(k)
A0 = 0.01 * I

@model function logit_model(X, y)
    # Set the priors only for β.
    β ~ MvNormal(β0, inv(A0))

    # Write the likelihood, can't vectorize?
    n = size(X, 1)
    for i = 1:n
        y[i] ~ Bernoulli(logistic(X[i, :]' * β))
    end
end

# Draw random number from posterior.
model = logit_model(X, y)

n_draws = 5_000
# n_chains = 4
n_chains = 1
n_tune = 1_000
chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains)
# chn = sample(model, NUTS(), MCMCThreads(), n_draws, n_chains)[n_tune:n_draws, :, :]
β_chn = chn[n_tune+1:n_draws, :, :] # delete initial and exclude odd (hamiltonian bluh bluh) chain 
β_chn = group(β_chn, :β)

summarize(β_chn)
quantile(β_chn)
plot(β_chn)

# Marginal effect 限界効果
# https://cran.r-project.org/web/packages/margins/vignettes/Introduction.html
# https://turinglang.github.io/MCMCChains.jl/dev/chains/
# https://docs.juliahub.com/MCMCChains/QRkwo/3.0.12/
"""
Marginal Effect of logit regression for continuous variable

∂Pr(Y=1|X)/∂X_j = ∂Pr(logistic(Xβ))/∂X_j 
                = (exp(Xβ) / (1 + exp(Xβ))²) β_j

The marginal effect on X_j depends on β_j, X_j and other β and X. Therefore, 

    β_chn:          random drawed β
    β_interest:     variables of interest, :Symbol
                    one of a symbol in `chain` argument.
    x:              independent variable vector to calculate marginal 
                    effect (individual, at mean, etc...). 
    Output
    ------
    dydxs:   marginal effects regarding to each random drawed β, ((n_draws - n_tune) * n_chains) matrix
"""
function marginal_effect_logit(β_chn, β_interest, x)
    β_rand = get_params(β_chn).β # get random value
    β_rand_interest = β_chn[β_interest].data
    dydxs = []
    for which_chain = chains(β_chn)
        β_rand_single_chain = reduce(hcat, [β_rand[j].data[:, which_chain] for j=1:length(β_rand)])
        β_rand_interest_single_chain = β_rand_interest[:, which_chain]
        dydx = exp.(β_rand_single_chain * x) ./ (1 .+ exp.(β_rand_single_chain * x)).^2 .* β_rand_interest_single_chain
        push!(dydxs, dydx)
    end
    dydxs = reduce(hcat, dydxs)
    return dydxs
end

rand_margins = marginal_effect_logit(β_chn, Symbol("β[2]"), X[100, :])
density(rand_margins)

"""
Average Marginal Effect of all sample.
"""
function marginal_effect_logit_mae(β_chn, β_interest, X)
    # https://discourse.julialang.org/t/converting-a-matrix-into-an-array-of-arrays/17038/7
    # https://discourse.julialang.org/t/how-to-broadcast-over-only-certain-function-arguments/19274
    return mean(marginal_effect_logit.((β_chn,), (β_interest,), [r for r in eachrow(X)]))
end

rand_mae = marginal_effect_logit_mae(β_chn, Symbol("β[2]"), X)
density(rand_mae)


marginal_effect_logit.((β_chn,), (Symbol("β[2]"),), [r for r in eachrow(X)])

step = 0.5
interest_dim = findall(x -> x == Symbol("β[2]"), names(β_chn))
other_dim = findall(x -> x != Symbol("β[2]"), names(β_chn))
x_interest = X[:, interest_dim]
# https://stackoverflow.com/questions/37661221/julia-select-all-but-one-element-in-array-matrix
x_other = X[:, 1:end .!= interest_dim]
x_interest_theoritical = Vector(minimum(x_interest):step:maximum(x_interest))
x_interest_theoritical
# Average other x
X_theoritical_ames = []
for x in x_interest_theoritical
    print(x)
    print("\n")
    X_theoritical_ame = copy(X)
    X_theoritical_ame[:, interest_dim] .= x
    push!(X_theoritical_ames, X_theoritical_ame)
end
marginal_effect_logit_mae.((β_chn,), (Symbol("β[2]"),), X_theoritical_ames)


# Plot Marginal Effect regarding to different value of X_j in interest
"""
Margins plot

Plot Marginal Effect regarding to different value of X_j in interest
# https://docs.julialang.org/en/v1/manual/functions/#Optional-Arguments
# https://docs.julialang.org/en/v1/manual/functions/#Keyword-Arguments
"""
function plot_marginal_effect(β_chn::Chains, β_interest::Symbol, x_interest_theoritical::Vector; at_mean::Bool=false, ci_p::Float64=0.95)
    interest_dim = findall(x -> x == β_interest, names(β_chn))
    other_dim = findall(x -> x != β_interest, names(β_chn))
    x_interest = X[:, interest_dim]
    # https://stackoverflow.com/questions/37661221/julia-select-all-but-one-element-in-array-matrix
    x_other = X[:, 1:end .!= interest_dim]
    # x_interest_theoritical = Vector(minimum(x_interest):step:maximum(x_interest))

    if at_mean
        # at mean other x
        X_theoritical_atmean = Matrix{Float64}(undef, size(x_interest_theoritical, 1), length(names(β_chn)))
        X_theoritical_atmean[:, interest_dim] = x_interest_theoritical
        for other in other_dim
            X_theoritical_atmean[:, other] .= mean(X, dims=1)[other]
        end
        rand_margins = marginal_effect_logit.((β_chn,), (β_interest,), [r for r in eachrow(X_theoritical_atmean)])
        ylabel = @sprintf("Marginal Effect at Means, CI(%d)", ci_p*100)
    else
        # Average other x
        X_theoritical_ames = []
        for x in x_interest_theoritical
            X_theoritical_ame = copy(X)
            X_theoritical_ame[:, interest_dim] .= x
            push!(X_theoritical_ames, X_theoritical_ame)
        end
        rand_margins = marginal_effect_logit_mae.((β_chn,), (β_interest,), X_theoritical_ames)
        ylabel = @sprintf("Average Marginal Effect, CI(%d)", ci_p*100)
    end

    # margins plot
    # https://discourse.julialang.org/t/asymmetric-error-bars-and-box-plots-in-julia/73647/2
    rand_margins_median = median.(rand_margins)
    ci_lower = quantile.(rand_margins, (1-ci_p)/2)
    ci_upper = quantile.(rand_margins, ci_p-(1-ci_p)/2)
    plot(x_interest_theoritical, rand_margins_median, color="black", xlabel=@sprintf("%s's x", β_interest), ylabel=ylabel,  legend=false)
    display(scatter!(x_interest_theoritical, rand_margins_median, yerror=(ci_lower, ci_upper), color="cornflowerblue"))
    return rand_margins
end

x_hypo = Vector(-2.0:0.5:3.5)

plot_marginal_effect(β_chn, Symbol("β[2]"), x_hypo, at_mean=false)
plot_marginal_effect(β_chn, Symbol("β[2]"), x_hypo, at_mean=true)
plot_marginal_effect(β_chn, Symbol("β[2]"), x_hypo, at_mean=true, ci_p=0.5)

plot_marginal_effect(β_chn, Symbol("β[3]"), x_hypo, at_mean=false)
plot_marginal_effect(β_chn, Symbol("β[3]"), x_hypo, at_mean=true)