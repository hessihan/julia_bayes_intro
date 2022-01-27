using Plots, StatsPlots, Distributions
# https://discourse.julialang.org/t/several-contour-plots-in-one-3d-plot/50986/2

# define multi-norm density function manually
function mvnormpdf(μ, Σ)
    dim = size(μ)[1]
    return 
end

size([1, 2])

Z = [pdf(MultivariateNormal([0, 0], [1 0; 0 1]),[i,j]) for i in Vector(-10:0.1:10), j in Vector(-10:0.1:10)]
plot(Vector(-10:0.1:10), Vector(-10:0.1:10), Z, st=:surface, c=:blues, alpha=0.5)
Z = [pdf(MultivariateNormal([0, 0], [5 0; 0 5]),[i,j]) for i in Vector(-10:0.1:10), j in Vector(-10:0.1:10)]
plot!(Vector(-10:0.1:10), Vector(-10:0.1:10), Z, st=:surface, c=:blues, alpha=0.5)
Z = [pdf(MultivariateNormal([0, 0], [2 3; 1 2]),[i,j]) for i in Vector(-10:0.1:10), j in Vector(-10:0.1:10)]
plot!(Vector(-10:0.1:10), Vector(-10:0.1:10), Z, st=:surface, c=:blues, alpha=0.5)