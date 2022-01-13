using Plots, Distributions

value_α = [0.5, 1.0, 2.0, 4.0]
value_β = [0.5, 1.0, 2.0, 4.0]

l = @layout [grid(4, 4)]
plots = []
for a in value_α, b in value_β
    if a == 0.5
        title = "β=$b"
    else 
        title = ""
    end
    if b == 0.5
        ylabel = "α=$a"
        yticks = [0.0, 2.5]
    else
        ylabel = ""
        yticks = nothing
    end
    if a == 4.0
        xticks = [0.0, 0.5, 1.0]
    else
        xticks = nothing
    end
    p = plot(Beta(a, b), label=false, xlims=(0.0, 1.0), ylims=(0.0, 4.5), title=title, ylabel=ylabel, xticks=xticks, yticks=yticks, grid = false)
    push!(plots, p)
end
plot(plots..., layout=l)