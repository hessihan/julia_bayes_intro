# https://storopoli.io/Bayesian-Julia/pages/3_prob_dist/
# https://nbviewer.org/github/takuizum/julia/blob/master/jmd/summaryPkg.html
using Plots, StatsPlots, LaTeXStrings
using Distributions, Random

# Random draw
Random.seed!(123)
d = DiscreteUniform(1, 6)
rand(d, 1)

# DiscreteUniform
d = DiscreteUniform(1, 6)
params(d)
params(d)[1]
params(d)[2]
span(d)
probval(d)
probval(d) * span(d)
minimum(d)
maximum(d)

plot(
    d,
    label="6-sided Dice",
    markershape=:circle,
    xlabel=L"\theta",
    ylabel="Probability Mass",
    ylims=(0, 0.3)
)

# Bernoulli
d = Bernoulli(0.1)
params(d)
succprob(d)
failprob(d)

plot(
    Bernoulli(0.5),
    markershape=:circle,
    label=L"p=0.5",
    alpha=0.5,
    xlabel=L"\theta",
    ylabel="Probability Mass",
    ylim=(0, 1)
)
plot!(
    Bernoulli(0.2),
    markershape=:circle,
    label=L"p=0.2",
    alpha=0.5
)

# Binomial
d = Binomial(5, 0.5)

plot(Binomial(5, 0.5),
        markershape=:circle,
        label=L"p=0.5",
        alpha=0.5,
        xlabel=L"\theta",
        ylabel="Probability Mass"
    )
plot!(Binomial(5, 0.2),
        markershape=:circle,
        label=L"p=0.2",
        alpha=0.5)

# Poisson
d = Distributions.Poisson(3)
params(d)
mean(d)

plot(Poisson(1),
        markershape=:circle,
        label=L"\lambda=1",
        alpha=0.5,
        xlabel=L"\theta",
        ylabel="Mass"
    )
plot!(Poisson(4),
    markershape=:circle,
    label=L"\lambda=4",
    alpha=0.5)

# NegativeBinomial
plot(NegativeBinomial(1, 0.5),
        markershape=:circle,
        label=L"k=1",
        alpha=0.5,
        xlabel=L"\theta",
        ylabel="Mass"
    )
plot!(NegativeBinomial(2, 0.5),
        markershape=:circle,
        label=L"k=2",
        alpha=0.5)

# Normal
d = Normal(0, 2)
params(d)
mean(d)
std(d)
var(d)

plot(
    Normal(0, 1),
    label=L"\sigma=1",
    lw=5,
    xlabel=L"\theta",
    ylabel="Probability Density",
    xlims=(-4, 4)
)
plot!(Normal(0, 0.5), label=L"\sigma=0.5", lw=5)
plot!(Normal(0, 2), label=L"\sigma=2", lw=5)

# LogNormal
d = LogNormal(0, 2)
params(d)
meanlogx(d)
varlogx(d)
stdlogx(d)

plot(LogNormal(0, 1),
        label=L"\sigma=1",
        lw=5,
        xlabel=L"\theta",
        ylabel="Density",
        xlims=(0, 3)
    )
plot!(LogNormal(0, 0.25), label=L"\sigma=0.25", lw=5)
plot!(LogNormal(0, 0.5), label=L"\sigma=0.5", lw=5)

# Exponential
plot(Exponential(1),
        label=L"\lambda=1",
        lw=5,
        xlabel=L"\theta",
        ylabel="Density",
        xlims=(0, 4.5)
    )
plot!(Exponential(0.5), label=L"\lambda=0.5", lw=5)
plot!(Exponential(1.5), label=L"\lambda=2", lw=5)

# Student-t
plot(TDist(2),
        label=L"\nu=2",
        lw=5,
        xlabel=L"\theta",
        ylabel="Density",
        xlims=(-4, 4)
    )
plot!(TDist(8), label=L"\nu=8", lw=5)
plot!(TDist(30), label=L"\nu=30", lw=5)

# Beta
plot(Beta(1, 1),
        label=L"a=b=1",
        lw=5,
        xlabel=L"\theta",
        ylabel="Density",
        xlims=(0, 1)
    )
plot!(Beta(3, 2), label=L"a=3, b=2", lw=5)
plot!(Beta(2, 3), label=L"a=2, b=3", lw=5)