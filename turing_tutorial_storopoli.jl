# https://storopoli.io/Bayesian-Julia/pages/4_Turing/

using Turing
using Statistics: mean, std
using Random:seed!
seed!(123)

@model linreg(X, y; predictors=size(X, 2)) = begin
    #priors
    α ~ Normal(mean(y), 2.5 * std(y))
    β ~ filldist(TDist(3), predictors)
    σ ~ Exponential(1)

    #likelihood
    y ~ MvNormal(α .+ X * β, σ)
end;

using DataFrames, CSV, HTTP

url = "https://raw.githubusercontent.com/storopoli/Bayesian-Julia/master/datasets/kidiq.csv"
kidiq = CSV.read(HTTP.get(url).body, DataFrame)
describe(kidiq)

X = Matrix(select(kidiq, Not(:kid_score)))
y = kidiq[:, :kid_score]
model = linreg(X, y);

chain = sample(model, NUTS(), MCMCThreads(), 2_000, 4)
# typeof(chain)
summarystats(chain)
quantile(chain)