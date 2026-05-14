using CSV
using CairoMakie
using FileIO
using StatsBase
using DataFrames

function load_data(path)
    df = CSV.read(path, DataFrame, delim=',', header=true)
    df[!, 1] = convert.(Int,df[!, 1])
    df[!, 2] = convert.(Int,df[!, 2])
    df[!, 3] = convert.(Int,df[!, 3])
    df[!, 4] = convert.(Int,df[!, 4])
    df
end

function create_plot(all_data, C, D, P, N, title, xlabel, ylabel, filename)
    with_theme(theme_latexfonts()) do
    f = Figure(size=(1000, 700))
    lsize=35
    tsize=30
    msize=25

    if isnothing(C)
        xax_scale = log10
        xticks = [2, 5, 10, 20, 50, 100, 200]
    elseif isnothing(D)
        xticks = [2, 4, 6, 8, 10]
        xax_scale = identity
    elseif isnothing(P)
        xax_scale = log10
        xticks = [100, 200, 500, 1000, 2000]
    else
        xax_scale = log10
        xticks = [20, 50, 100, 200, 500, 1000]
    end

    ax = Axis(f[1, 1], title=title, xlabel=xlabel, ylabel=ylabel, titlesize=lsize,
        xlabelsize=lsize, ylabelsize=lsize, xticklabelsize=tsize, yticklabelsize=tsize, xticks=xticks,
        xscale=xax_scale #, yscale=log10
        )

    for (key, data) in all_data
        if isnothing(C)
            rows = data[(data[!, 1] .== N) .& (data[!, 3] .== D) .& (data[!, 4] .== P), :]
            x = rows[!, 2]
        elseif isnothing(D)
            rows = data[(data[!, 2] .== C) .& (data[!, 1] .== N) .& (data[!, 4] .== P), :]
            x = rows[!, 3]
        elseif isnothing(P)
            rows = data[(data[!, 2] .== C) .& (data[!, 3] .== D) .& (data[!, 1] .== N), :]
            x = rows[!, 4]
        else
            rows = data[(data[!, 1] .<2000) .& (data[!, 2] .== C) .& (data[!, 3] .== D) .& (data[!, 4] .== P), :]
            x = rows[!, 1]
        end

        rsp = sortperm(x)
        x = x[rsp]
        rows = rows[rsp, :]

        println(x)

        iters_all = [collect(row[[n for n in names(rows) if occursin("iters", n)]]) for row in eachrow(rows)]
        iters = [it[it .!= -1] for it in iters_all]
    
        ts_all = [collect(row[[n for n in names(rows) if occursin("run", n)]]) for row in eachrow(rows)]
        ts = [ts[ts .!= -1] for ts in ts_all]

        md = median.(ts)
        ts = [t[t .< 2*md[i]] for (i, t) in enumerate(ts)]

        ms = mean.(ts)
        cs = std.(ts)

        band!(ax, x, ms .- 3.0 .* cs, ms .+ 3 .* cs, alpha=0.5)
        scatter!(ax, x, ms, label=key, markersize=msize)
        lines!(ax, x, ms, linewidth=4)
    end

    axislegend(ax, position=:lt, labelsize=lsize, markersize=msize)
    save(filename*".svg", f)
    save(filename*".pdf", f)
end
end

function create_plots()
    base_result_path = "benchmark/bench_results"
    pathjl = joinpath(base_result_path, "benchmarks_thresh_1e-3_gpu.csv")
    pathjl_local = joinpath(base_result_path, "benchmarks_thresh_1e-3_local_gpu.csv")
    pathjlc = joinpath(base_result_path, "benchmarks_thresh_1e-3.csv")
    pathjlc_local = joinpath(base_result_path, "benchmarks_thresh_1e-3_local.csv")

    C = 200
    N = 500
    D = 2
    P = 1000

    datajl = load_data(pathjl)
    datajl_local = load_data(pathjl_local)
    datajlc = load_data(pathjlc)
    datajlc_local = load_data(pathjlc_local)

    all_data = [("J+CPU+N", datajlc), ("J+CPU+O", datajlc_local), ("J+GPU+N", datajl), ("J+GPU+O", datajl_local)]

    base_plot_path = "benchmark/bench_plots"
    create_plot(all_data, nothing, D, P, N, "N=$N, D=$D, P=$P", "number of components", "runtime in s", joinpath(base_plot_path, "num_components"))
    create_plot(all_data, C, nothing, P, N, "N=$N, C=$C, P=$P", "dimension", "runtime in s", joinpath(base_plot_path, "dimension"))
    create_plot(all_data, C, D, nothing, N, "N=$N, C=$C, D=$D", "number of projections", "runtime in s", joinpath(base_plot_path, "num_projections"))
    create_plot(all_data, C, D, P, nothing, "C=$C, D=$D, P=$P", "number of samples", "runtime in s", joinpath(base_plot_path, "num_samples"))
end

function create_plot_py_jl()
    base_result_path = "benchmark/bench_results"

    pathjl_local = joinpath(base_result_path, "benchmarks_100_iters_local_gpu.csv")
    pathjlc_local = joinpath(base_result_path, "benchmarks_100_iters_local.csv")

    pathpyg = joinpath(base_result_path, "python_results_100_iterations_gpu.csv")
    #pathpyc = joinpath(base_result_path, "python_results_100_iterations.csv")

    C = 1
    D = 2
    P = 1000

    datajl_local = load_data(pathjl_local)
    datajlc_local = load_data(pathjlc_local)
    data_py_gpu = load_data(pathpyg)

    all_data = [("J+CPU+O", datajlc_local), ("J+GPU+O", datajl_local), ("P+GPU+O", data_py_gpu)]
    
    base_plot_path = "benchmark/bench_plots"
    save_path = joinpath(base_plot_path, "jl_py_num_samples")

    with_theme(theme_latexfonts()) do
    f = Figure(size=(1000, 700))
    lsize=35
    tsize=30
    msize=25

    xax_scale = log10
    xticks = [20, 100, 1000, 5000]

    ax = Axis(f[1, 1], title="C=$C, D=$D, P=$P", xlabel="number of samples", ylabel="runtime in s", titlesize=lsize,
        xlabelsize=lsize, ylabelsize=lsize, xticklabelsize=tsize, yticklabelsize=tsize, xticks=xticks,
        xscale=xax_scale #, yscale=log10
        )

    for (key, data) in all_data
        rows = data[:, :]
        x = rows[!, 1]

        rsp = sortperm(x)
        x = x[rsp]
        rows = rows[rsp, :]

        println(x)

        ts_all = [collect(row[[n for n in names(rows) if occursin("run", n) || occursin("time", n)]]) for row in eachrow(rows)]
        ts = [ts[ts .!= -1] for ts in ts_all]

        md = median.(ts)
        ts = [t[t .< 2*md[i]] for (i, t) in enumerate(ts)]

        ms = mean.(ts)
        cs = std.(ts)

        band!(ax, x, ms .- 3.0 .* cs, ms .+ 3 .* cs, alpha=0.5)
        scatter!(ax, x, ms, label=key, markersize=msize)
        lines!(ax, x, ms, linewidth=4)
    end

    axislegend(ax, position=:lt, labelsize=lsize, markersize=msize)
    save(save_path*".svg", f)
    save(save_path*".pdf", f)
end
end

create_plots()
create_plot_py_jl()