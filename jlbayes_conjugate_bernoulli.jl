using Random, Plots, StatsPlots, Distributions, DataFrames, Optim
# Bayesian Inference for Bernoulli parameter (inference success probability)
# Define HPD (highest posterior density) interval
# HPD are difficult to calculate and not be used nowadays. 
# Maybe this? https://github.com/tpapp/HighestDensityRegions.jl
"""
HPD interval for beta dist
Optim.jl
http://julianlsolvers.github.io/Optim.jl/v0.9.3/user/tipsandtricks/#dealing-with-constant-parameters
    
    Inputs
    ------
    ci0:  initial value for calculating HPD interval numerically
    α:    parameter 1 of Beta dist
    β:    parameter 2 of Beta dist
    prob: probabilty of HPD interval
    
    Outputs
    -------
    HPD interval

"""
function beta_hpdi(ci0, α, β, prob)
    """
    Continuous equation for HPD interval
    
        Inputs
        ------
        v: HPD interval (2-element Vector)
        a: parameter 1 of Beta dist
        b: parameter 2 of Beta dist
        p: probabilty of HPD interval
        
        Outputs
        -------
        HPD interval condition equations (2-element Vector)

    """
    function hpdi_conditions(v, a, b, p)
        # HPD interval probability = p
        eq1 = cdf(Beta(a, b), v[2]) - cdf(Beta(a, b), v[1]) - p
        # p(lower|D) = p(upper|D)
        eq2 = pdf(Beta(a, b), v[2]) - pdf(Beta(a, b), v[1])
        return [eq1, eq2]
    end
    return nothing
end

function hpdi_conditions(v, a, b, p)
    # HPD interval probability = p
    eq1 = cdf(Beta(a, b), v[2]) - cdf(Beta(a, b), v[1]) - p
    # p(lower|D) = p(upper|D)
    eq2 = pdf(Beta(a, b), v[2]) - pdf(Beta(a, b), v[1])
    return [eq1, eq2]
end

function hpdi_conditions(v)
    # HPD interval probability = p
    eq1 = cdf(Beta(14, 38), v[2]) - cdf(Beta(14, 38), v[1]) - 0.95
    # p(lower|D) = p(upper|D)
    eq2 = pdf(Beta(14, 38), v[2]) - pdf(Beta(14, 38), v[1])
    return [eq1, eq2]
end


# Optimization
hpdi_conditions([0.17, 0.41])
Optim.optimize(hpdi_conditions, [0.17, 0.41])
hpdi_conditions([0, 0])
Optim.optimize(hpdi_conditions, [0, 0])
NLsolve.nlsolve(hpdi_conditions)

# Define calculation of posterior for Bernoulli parameter
"""
Posterior Distribution (Beta dist) for Bernoulli parameter π

    Inputs
    ------
    data: data (0 or 1)
    a0: parameter 1 of prior (Beta dist)
    b0: parameter 2 of prior (Beta dist)
    prob: credible interval probability (0 < prob < 1)

    Outputs
    -------
    results: posterior summary stats table (DataFrames)
    a: parameter 1 of posterior (Beta dist)
    b: parameter 2 of posterior (Beta dist)

"""
function bernoulli_stats(data, a0, b0, prob)
    # posterior
    n = size(data)[1]
    sum_data = sum(data)
    a = sum_data + a0
    b = n - sum_data + b0
    d_posterior = Beta(a, b)
    # mean
    mean_π = mean(d_posterior)
    # median
    median_π = median(d_posterior)
    # mode
    mode_π = mode(d_posterior)
    # Analytic Mode for Beta Dist
    # (a - 1.0) / (a + b - 2.0)
    # std
    sd_π = std(d_posterior)
    # credible interval
    # ci upper boud
    ci_π_upper = quantile(d_posterior, prob - (1-prob)/2)
    # ci lower boud
    ci_π_lower = quantile(d_posterior, (1-prob)/2)
    ci_π = [ci_π_lower, ci_π_upper]

    posterior_stats = reshape([mean_π, median_π, mode_π, sd_π, ci_π], 1, 5)
    stats_string = ["Mean", "Median", "Mode", "SD", "CI"]
    results = DataFrame(posterior_stats, stats_string)
    return results, a, b
end

# Generate moc data from Bernoulli
p = 0.25
n = 50
Random.seed!(99)
data = rand(Bernoulli(p), n)

# Set prior
a0 = 1.0
b0 = 1.0

# Calculate posterior statistics
prob = 0.95
results, a, b = bernoulli_stats(data, a0, b0, prob)
print(results)

# Visualize Posterior
plot(Beta(a0, b0), xlims=(0, 1), ylims=(0, 7), xlabel="Success Probability q", ylabel="Probability Density", label="prior: Beta(1, 1)", linestyle=:dash)
plot!(Beta(a, b), label="posterior: Beta($a, $b)", linestyle=:solid)

# Plot posterior by different prior
Random.seed!(99)
data = rand(Bernoulli(p), 250) # more data
value_size = [10, 50, 250]
value_a0 = [1.0, 6.0]
value_b0 = [1.0, 4.0]
styles = [:dot, :dashdot, :dash, :solid]

l = @layout [grid(1, 2)]
plots = []
for index = 1:2
    style_index = 1
    a0_i = value_a0[index]
    b0_i = value_b0[index]
    pl = plot(Beta(a0_i, b0_i), linestyle=styles[style_index], label="prior: Beta($a0_i, $b0_i)", color="cornflower blue",
             xlims=(0, 1), ylims=(0, 15.5), xlabel="Success Probability q", ylabel="Probability Density")
    for n_j in value_size
        style_index += 1
        sum_data = sum(data[1:n_j])
        a_j = sum_data + a0_i
        b_j = n_j - sum_data + b0_i
        plot!(Beta(a_j, b_j), linestyle=styles[style_index], label="posterior: Beta($a_j, $b_j)", color="cornflower blue")
    end
    push!(plots, pl)
end
plot(plots..., layout=l)

# Animation
# https://stackoverflow.com/questions/55794068/animating-subplots-using-plots-jl-efficiently
Random.seed!(99)
data = rand(Bernoulli(p), 250) # more data
value_size = Vector(1:1:100)
value_a0 = [1.0, 6.0]
value_b0 = [1.0, 4.0]

gr(fmt = :png)
anim = @animate for n_j in value_size
    sum_data = sum(data[1:n_j])
    a_j_1 = sum_data + value_a0[1]
    b_j_1 = n_j - sum_data + value_b0[1]
    # all_p[1][1][:z] = Beta(a_j, b_j)
    p1 = plot(Beta(a_j_1, b_j_1), color="cornflower blue", label="posterior: Beta($a_j_1, $b_j_1)", xlims=(0, 1), ylims=(0, 15.5), xlabel="Success Probability q", ylabel="Probability Density", title="n=$n_j")
    a0_1 = value_a0[1]
    b0_1 = value_b0[1]
    plot!(Beta(a0_1, b0_1), linestyle=:dot, label="prior: Beta($a0_1, $b0_1)")

    a_j_2 = sum_data + value_a0[2]
    b_j_2 = n_j - sum_data + value_b0[2]
    # all_p[2][1][:z] = Beta(a_j, b_j)
    p2 = plot(Beta(a_j_2, b_j_2), color="cornflower blue", label="posterior: Beta($a_j_2, $b_j_2)", xlims=(0, 1), ylims=(0, 15.5), xlabel="Success Probability q", ylabel="Probability Density", title="n=$n_j")
    a0_2 = value_a0[2]
    b0_2 = value_b0[2]
    plot!(Beta(a0_2, b0_2), linestyle=:dot, label="prior: Beta($a0_2, $b0_2)")

    layout = @layout [grid(1, 2)]
    plot(p1, p2; layout)
end
gif(anim, "posterior.gif", fps=10)