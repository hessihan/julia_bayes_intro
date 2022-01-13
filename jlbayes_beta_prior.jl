using Plots, Distributions

plot(
    Uniform(0, 1),
    label="Uniform(0, 1)",
    xlabel="Success Probability q",
    ylabel="Probability Density",
    xlims=(0, 1),
    ylims=(0, 2.8)
)
plot!(
    Beta(4, 6),
    label="Beta(α=4, β=6)",
    linestyle=:dash
)