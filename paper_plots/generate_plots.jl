using PCDSampling
using Distributions
using CairoMakie
using FileIO
using Random
using DelimitedFiles
using GaussianMixtures
using CSV
using DataFrames

function qualitative_comparison()
    Random.seed!(42)
    N=50
    vs = [sqrt(2), 1.0]
    vs = [2, 1.0]
    target = MvNormal(zeros(2), vs)
    mv_target = MixtureModel([target])
    init = randn(2, N)
    dirs = uniform_directions_2d(1000)
    X_local, _ = draw_samples(mv_target, N, dirs; init_samples=copy(init), use_local=true, stop_cond=max_iters_and_small_delta(2000, 1e-10)) 
    X_lut, _ = draw_samples(mv_target, N, dirs; init_samples=copy(init), use_local=false, stop_cond=max_iters_and_small_delta(2000, 1e-10)) 
    X_lcd = readdlm("./paper_plots/data/lcd_samples.csv")

    for (key, X) in Dict("original" => X_local, "proposed" => X_lut, "LCD" => X_lcd)
        display(cov(X, dims=2))

        # println(key, " LCD: ", LCD.dist_asym(X, vs.^2))

        lsize=50
        ticklsize=45
        h = 500
        with_theme(theme_latexfonts()) do
        # if (key == "LCD")
            f = Figure(size=(1000, h))
        # else
        #     f = Figure(size=(800, h))
        # end
        ax = Axis(f[1, 1], aspect=DataAspect(), xlabel=L"x_1 \rightarrow", ylabel=L"x_2 \rightarrow", 
                    xlabelsize=lsize, ylabelsize=lsize, xticklabelsize=ticklsize, yticklabelsize=ticklsize)
        xlims!(ax, -5, 5)
        ylims!(ax, -2.5, 2.5)
        cp = contourf!(ax, -5:0.05:5, -2.5:0.05:2.5, (x, y) -> Distributions.pdf(target, [x, y]), 
            colormap=:viridis)
        scatter!(ax, X, markersize=30, color=:red)
        if (key == "LCD")
            Colorbar(f[1, 2], cp, label=L"\text{pdf}", labelsize=lsize, ticklabelsize=ticklsize, width=20, halign=:left)
        end
        colsize!(f.layout, 1, 800)
        rowsize!(f.layout, 1, 400)

        resize_to_layout!(f)
        
        save("./paper_plots/plots/$key.pdf", f)
        save("./paper_plots/plots/$key.svg", f)
        end
    end
end

function first_page(;N=200)
    Random.seed!(42)
    C = 100
    ws = rand(C)
    ws ./= sum(ws)
    
    # Data from: https://github.com/algorithmicathlete/algorithmicathlete/blob/main/mid-range-dead/years/2000.csv
    data = CSV.read("paper_plots/data/2000.csv", DataFrame)
    X = Matrix{Float64}(dropmissing(data[:, [:LOC_X, :LOC_Y]]))

    X ./= [100.0 100.0]
    println(extrema(X[:, 1]))
    println(extrema(X[:, 2]))
    
    X = X[X[:, 2] .< 3.0, :]
    X = X[X[:, 1].^2 .+ X[:, 2].^2 .> 0.2^2, :]
    display(size(X))
    
    target = MixtureModel(GMM(C, X))

    dirs = uniform_directions_2d(1000)
    X, _ = draw_samples(target, N, dirs, stop_cond=max_iters_and_small_delta(10000, 1e-8), N_lut=200) 
    
    with_theme(theme_latexfonts()) do
    f = Figure(width=800, height=400)
    ax = Axis(f[1, 1], aspect=DataAspect(), xlabel=L"x_1 \rightarrow", ylabel=L"x_2 \rightarrow", 
                xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
    
                xs = -2.5:0.01:2.5
    ys = -0.4:0.01:2.8
    zs = [Distributions.pdf(target, [x, y]) for x in xs, y in ys]
    cp = contourf!(ax,xs, ys, sqrt.(zs), colormap=:viridis)

    scatter!(ax, X, markersize=10, color=:red)
    Colorbar(f[1, 2], cp, label = L"\text{pdf}", labelsize=20, ticklabelsize=20, ticks=[0.2, 0.4, 0.6])
    rowsize!(f.layout, 1, 260)
    resize_to_layout!(f)
    
    save("./paper_plots/plots/first_page.pdf", f)
    save("./paper_plots/plots/first_page.svg", f)

    f = Figure(width=800, height=400)
    ax = Axis(f[1, 1], aspect=DataAspect(), xlabel=L"x_1 \rightarrow", ylabel=L"x_2 \rightarrow", 
                xlabelsize=25, ylabelsize=25, xticklabelsize=20, yticklabelsize=20)
    xs = -2.5:0.01:2.5
    ys = -0.4:0.01:2.8
    zs = [Distributions.pdf(target, [x, y]) for x in xs, y in ys]
    cp = contourf!(ax,xs, ys, sqrt.(zs), colormap=:viridis)
    scatter!(ax, rand(target, N), markersize=10, color=:red)
    Colorbar(f[1, 2], cp, label = L"\text{pdf}", labelsize=20, ticklabelsize=20, ticks=[0.2, 0.4, 0.6])
    rowsize!(f.layout, 1, 260)
    resize_to_layout!(f)
    
    save("./paper_plots/plots/first_page_random.pdf", f)
    save("./paper_plots/plots/first_page_random.svg", f)
    end
end

function slices_and_projections()
    N = 10000
    xmoon = cos.(range(0, pi, N))
    ymoon = sin.(range(0, pi, N))
    samples = [xmoon;; ymoon] .+ 0.2*randn(N, 2)

    a = -pi/7
    R = [cos(a) -sin(a); sin(a) cos(a)]

    samples = samples * R
    density = MixtureModel(GMM(10, samples))
    proj = project(density, [1.0, 0.0])

    with_theme(theme_latexfonts()) do
    f = Figure()
    ax = Axis3(f[1, 1], azimuth=-pi/4, elevation=0.55, aspect=:data, protrusions=0.0)
    uy = 2.0
    xlims!(ax, -2.0, 2.0)
    ylims!(ax, -1.0, uy)
    zlims!(ax, 0.0, 1.2)
    xs = -2.0:0.05:2.0
    ys = -1.0:0.05:uy
    
    lower = Makie.Point3f.(xs, fill(uy, length(xs)), fill(0.0, length(xs)))
    upper = Makie.Point3f.(xs, fill(uy, length(xs)), Distributions.pdf.(Ref(proj), xs))
    band!(ax, lower, upper, color=(0.8, 0.8, 1.0))
    lines!(ax, xs, fill(uy, length(xs)), Distributions.pdf.(Ref(proj), xs), linewidth=3, color=:blue)
    
    zs = [Distributions.pdf(density, [x, y]) for x in xs, y in ys]
    surface!(ax,xs, ys, zs, colormap=:Purples_3)
    
    xv1 = 0.4
    xv2 = -0.2
    xv3 = -0.8
    lines!(ax, fill(xv1, length(ys)), ys, [Distributions.pdf(density, [xv1, y]) for y in ys], linewidth=3, color=(19, 79, 26)./255)
    lines!(ax, fill(xv2, length(ys)), ys, [Distributions.pdf(density, [xv2, y]) for y in ys], linewidth=3, color=(43, 143, 55)./255)
    lines!(ax, fill(xv3, length(ys)), ys, [Distributions.pdf(density, [xv3, y]) for y in ys], linewidth=3, color=(61, 196, 77)./255)

    arrow_base = Makie.Point3f([0.6, -1.2, 0.0])
    arrows2d!([arrow_base], [Makie.Point3f([-1.0, 0.0, 0.0])])

    text!(f.scene, L"u", position=(145, 65), fontsize=30, color=:black, space=:pixel)

    hidedecorations!(ax)
    hidespines!(ax)
    resize_to_layout!(f)

    save("paper_plots/plots/slices_and_projections.pdf", f)
    save("paper_plots/plots/slices_and_projections.png", f)
    end
end

function generate_plots()
    first_page()
    qualitative_comparison()
    slices_and_projections()
end

generate_plots()
