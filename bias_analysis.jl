#!/usr/bin/env julia
# bias_analysis.jl
#
# Quantifies systematic bias in (D)CNN transfer-function predictions vs. ground truth,
# for both 1D (delay only) and 2D (velocity-delay) models.
# Addresses referee comment #10.
# Generated with assistance from Claude Code.
#
# Usage:
#   julia bias_analysis.jl [1D_results_dir] [2D_results_dir]

using JLD2, Statistics, Plots, Distributions, Printf, LaTeXStrings
gr()

const DEFAULT_1D_DIR = "/run/media/kirk/e9e7a38c-82a8-49cb-8ec3-e0c33e573437/FINESST2024/alpineResults/1DNormal/"
const DEFAULT_2D_DIR = "/run/media/kirk/e9e7a38c-82a8-49cb-8ec3-e0c33e573437/FINESST2024/alpineResults/2DNormal/"
const PAPER_DIR      = "/home/kirk/Documents/research/Dexter/DCNNPaper/paperTeX/"

# Shared plot style applied to every panel
const PKW = (
    tick_direction = :out,
    minorticks     = 5,
    minorgrid      = true,
    widen          = false,
    framestyle     = :box,
    left_margin    = 6Plots.mm,
    right_margin   = 6Plots.mm,
    bottom_margin  = 6Plots.mm,
    top_margin     = 1Plots.mm,
    fontfamily     = "Computer Modern",
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

function ttest_onesample(x)
    x = x[.!isnan.(x)]
    n = length(x)
    n < 2 && return (NaN, NaN, NaN, NaN, n)
    μ = mean(x);  σ = std(x)
    t = μ / (σ / sqrt(n))
    p = 2 * cdf(TDist(n - 1), -abs(t))
    return (μ, σ, t, p, n)
end

function print_stats_table(metrics)
    println("Δ = predicted − truth.  t-test H₀: mean bias = 0.  ** = p < 0.05\n")
    @printf("  %-30s  %9s  %9s  %8s  %8s\n", "Metric", "mean", "std", "t-stat", "p-value")
    println("  " * "-"^70)
    for (label, Δ, unit) in metrics
        μ, σ, t_stat, p, n = ttest_onesample(Δ)
        sig = (p < 0.05) ? " **" : "   "
        @printf("  %-30s  %+8.3f %-5s  %8.3f %-5s  %+6.2f  %.4f%s\n",
                label, μ, unit, σ, unit, t_stat, p, sig)
    end
    println()
end

function scatter_panel(x_vals, y_vals, xlab, ylab, title_str)
    mask = .!isnan.(x_vals) .& .!isnan.(y_vals)
    x = x_vals[mask];  y = y_vals[mask]
    lo = min(minimum(x), minimum(y))
    hi = max(maximum(x), maximum(y))
    p = scatter(x, y;
                xlabel=xlab, ylabel=ylab, title=title_str,
                markersize=2, markerstrokewidth=0, alpha=0.35,
                color=:dodgerblue, legend=false,
                PKW...)
    plot!(p, [lo, hi], [lo, hi]; lw=1.5, ls=:dash, color=:gray)
    return p
end

function bias_panel(Δ, title_str, xlab; nbins=40, xlim_pctile=nothing, legend_pos=:topleft)
    μ, σ, t_stat, p, n = ttest_onesample(Δ)
    Δ_clean = filter(!isnan, Δ)
    p16 = quantile(Δ_clean, 0.16)
    p84 = quantile(Δ_clean, 0.84)
    # Compute bin edges so that zero always falls at a bin center
    xlo = isnothing(xlim_pctile) ? minimum(Δ_clean) : quantile(Δ_clean, xlim_pctile)
    xhi = isnothing(xlim_pctile) ? maximum(Δ_clean) : quantile(Δ_clean, 1 - xlim_pctile)
    w   = (xhi - xlo) / nbins
    k_start = floor(Int, xlo / w - 0.5)
    k_end   = ceil(Int,  xhi / w - 0.5)
    edges   = [(k + 0.5) * w for k in k_start:k_end]
    ph = histogram(Δ_clean;
                   bins=edges, normalize=:probability,
                   xlabel=xlab, ylabel="Fraction",
                   title=title_str,
                   color=:dodgerblue, alpha=0.7, label=false,
                   PKW...)
    if !isnothing(xlim_pctile)
        xlims!(ph, xlo, xhi)
    end
    vspan!(ph, [p16, p84]; color=:crimson, alpha=0.2, label=false)
    vline!(ph, [0.0]; lw=2, ls=:dash,  color=:black,  label="zero")
    mean_label = "mean = $(latexstring(@sprintf("%+.2f^{+%.2f}_{-%.2f}", μ, p84-μ, μ-p16)))"
    vline!(ph, [μ];   lw=2, ls=:solid, color=:crimson, label=mean_label)
    plot!(ph; legend=legend_pos, legendframealpha=0,
          background_color_legend=:transparent, foreground_color_legend=:transparent)
    return ph
end

# ---------------------------------------------------------------------------
# 1D analysis
# ---------------------------------------------------------------------------

function moments_1D(Ψ, t)
    total = sum(Ψ)
    total <= 0.0 && return (NaN, NaN)
    τ_bar  = sum(t .* Ψ) / total
    σ_τ_sq = sum((t .- τ_bar).^2 .* Ψ) / total
    return (τ_bar, sqrt(max(σ_τ_sq, 0.0)))
end

function run_1D(saveDir)
    println("\n========== 1D Bias Analysis ==========")
    println("Loading: $(joinpath(saveDir, "results_dict.jld2"))")
    d = load(joinpath(saveDir, "results_dict.jld2"), "dict")

    pred  = d["pred"];  truth = d["test"][2];  t = Float64.(d["t"])
    @assert ndims(pred) == 3 "Expected 3D array for 1D results."
    nT, nLC, nSamples = size(pred)
    println("Dimensions: nT=$nT, nLC=$nLC, nSamples=$nSamples")

    pred = max.(pred, 0f0);  truth = max.(truth, 0f0)

    τ_bar_true = fill(NaN, nSamples);  τ_bar_pred = fill(NaN, nSamples)
    σ_τ_true   = fill(NaN, nSamples);  σ_τ_pred   = fill(NaN, nSamples)

    println("Computing moments...")
    for i in 1:nSamples
        τ_bar_true[i], σ_τ_true[i] = moments_1D(Float64.(truth[:, 1, i]), t)
        τ_bar_pred[i], σ_τ_pred[i] = moments_1D(Float64.(pred[:,  1, i]), t)
    end

    Δτ_bar = τ_bar_pred .- τ_bar_true
    Δσ_τ   = σ_τ_pred   .- σ_τ_true

    println(); print_stats_table([
        ("Centroid lag  Δτ̄",  Δτ_bar, "days"),
        ("Lag width  Δσ_τ",   Δσ_τ,   "days"),
    ])

    # Scatter
    ps1 = scatter_panel(τ_bar_true, τ_bar_pred,
                        L"True $\bar{\tau}$ [days]",
                        L"Predicted $\bar{\tau}$ [days]",
                        L"(D)CNN centroid lag $\bar{\tau}$")
    ps2 = scatter_panel(σ_τ_true, σ_τ_pred,
                        L"True $\sigma_\tau$ [days]",
                        L"Predicted $\sigma_\tau$ [days]",
                        L"(D)CNN lag extent $\sigma_\tau$")
    fig_scatter = plot(ps1, ps2; layout=(1,2), size=(800,410), dpi=200,
                       top_margin=2Plots.mm, plot_titlevspan=0.08,
                       plot_title="(D)CNN predictions vs. ground truth — 1D test set")
    out1 = joinpath(PAPER_DIR, "bias_analysis_1D_scatter.pdf")
    savefig(fig_scatter, out1);  println("Saved: $out1")

    # Histograms
    ph1 = bias_panel(Δτ_bar,
                     L"Centroid lag bias $\Delta\bar{\tau}$",
                     L"$\Delta\bar{\tau}$ [days]"; xlim_pctile=0.005)
    ph2 = bias_panel(Δσ_τ,
                     L"Lag extent bias $\Delta\sigma_\tau$",
                     L"$\Delta\sigma_\tau$ [days]"; xlim_pctile=0.005)
    fig_hist = plot(ph1, ph2; layout=(1,2), size=(800,410), dpi=200,
                    top_margin=2Plots.mm, plot_titlevspan=0.08,
                    plot_title="(D)CNN systematic bias — 1D test set")
    out2 = joinpath(PAPER_DIR, "bias_analysis_1D_histograms.pdf")
    savefig(fig_hist, out2);  println("Saved: $out2")

    out3 = joinpath(saveDir, "bias_analysis_1D_results.jld2")
    save(out3, "τ_bar_true", τ_bar_true, "τ_bar_pred", τ_bar_pred,
               "σ_τ_true",   σ_τ_true,   "σ_τ_pred",   σ_τ_pred,
               "Δτ_bar", Δτ_bar, "Δσ_τ", Δσ_τ, "t", t)
    println("Saved: $out3")
end

# ---------------------------------------------------------------------------
# 2D analysis
# ---------------------------------------------------------------------------

function load_velocity_axis(saveDir, nLC)
    for fname in ("trainTestCFull.jld2", "trainTestC.jld2")
        fpath = joinpath(saveDir, fname)
        isfile(fpath) || continue
        result = jldopen(fpath, "r") do f
            for key in ("v", "ν", "vCenters", "velocity")
                if key in keys(f)
                    v = read(f, key)
                    if length(v) == nLC
                        println("Loaded velocity axis ('$key') from $fname: " *
                                "$(round(minimum(v), digits=1)) to $(round(maximum(v), digits=1)) km/s")
                        return Float64.(v)
                    end
                end
            end
            return nothing
        end
        !isnothing(result) && return result
    end
    v = collect(range(-10000.0, stop=10000.0, length=nLC))
    println("Warning: velocity axis not found — using default ±10000 km/s over $nLC bins.")
    return v
end

function moments_2D(Ψ, t, v)
    total = sum(Ψ)
    total <= 0.0 && return (NaN, NaN, NaN, NaN)
    τ_profile = vec(sum(Ψ, dims=2))
    τ_bar     = sum(t .* τ_profile) / total
    σ_τ_sq    = sum((t .- τ_bar).^2 .* τ_profile) / total
    v_profile = vec(sum(Ψ, dims=1))
    v_bar     = sum(v .* v_profile) / total
    σ_v_sq    = sum((v .- v_bar).^2 .* v_profile) / total
    return (τ_bar, v_bar, sqrt(max(σ_τ_sq, 0.0)), sqrt(max(σ_v_sq, 0.0)))
end

function run_2D(saveDir)
    println("\n========== 2D Bias Analysis ==========")
    println("Loading: $(joinpath(saveDir, "results_dict.jld2"))")
    d = load(joinpath(saveDir, "results_dict.jld2"), "dict")

    pred  = d["pred"];  truth = d["test"][2];  t = Float64.(d["t"])
    @assert ndims(pred) == 4 "Expected 4D array for 2D results."
    nT, nLC, nC, nSamples = size(pred)
    println("Dimensions: nT=$nT, nLC=$nLC, nC=$nC, nSamples=$nSamples")

    v = load_velocity_axis(saveDir, nLC)
    pred = max.(pred, 0f0);  truth = max.(truth, 0f0)

    τ_bar_true = fill(NaN, nSamples);  τ_bar_pred = fill(NaN, nSamples)
    v_bar_true = fill(NaN, nSamples);  v_bar_pred = fill(NaN, nSamples)
    σ_τ_true   = fill(NaN, nSamples);  σ_τ_pred   = fill(NaN, nSamples)
    σ_v_true   = fill(NaN, nSamples);  σ_v_pred   = fill(NaN, nSamples)

    println("Computing moments...")
    for i in 1:nSamples
        τ_bar_true[i], v_bar_true[i], σ_τ_true[i], σ_v_true[i] = moments_2D(Float64.(truth[:,:,1,i]), t, v)
        τ_bar_pred[i], v_bar_pred[i], σ_τ_pred[i], σ_v_pred[i] = moments_2D(Float64.(pred[:,:,1,i]),  t, v)
    end

    Δτ_bar = τ_bar_pred .- τ_bar_true
    Δv_bar = v_bar_pred .- v_bar_true
    Δσ_τ   = σ_τ_pred   .- σ_τ_true
    Δσ_v   = σ_v_pred   .- σ_v_true

    println(); print_stats_table([
        ("Centroid lag  Δτ̄",        Δτ_bar, "days"),
        ("Centroid velocity  Δv̄",   Δv_bar, "km/s"),
        ("Lag width  Δσ_τ",         Δσ_τ,   "days"),
        ("Velocity width  Δσ_v",    Δσ_v,   "km/s"),
    ])

    # Scatter
    ps1 = scatter_panel(τ_bar_true, τ_bar_pred,
                        L"True $\bar{\tau}$ [days]",
                        L"Predicted $\bar{\tau}$ [days]",
                        L"(D)CNN centroid lag $\bar{\tau}$")
    ps2 = scatter_panel(v_bar_true, v_bar_pred,
                        L"True $\bar{v}$ [km/s]",
                        L"Predicted $\bar{v}$ [km/s]",
                        L"(D)CNN centroid velocity $\bar{v}$")
    ps3 = scatter_panel(σ_τ_true, σ_τ_pred,
                        L"True $\sigma_\tau$ [days]",
                        L"Predicted $\sigma_\tau$ [days]",
                        L"(D)CNN lag extent $\sigma_\tau$")
    ps4 = scatter_panel(σ_v_true, σ_v_pred,
                        L"True $\sigma_v$ [km/s]",
                        L"Predicted $\sigma_v$ [km/s]",
                        L"(D)CNN velocity extent $\sigma_v$")
    fig_scatter = plot(ps1, ps2, ps3, ps4;
                       layout=(2,2), size=(800,760), dpi=200,
                       plot_title="(D)CNN predictions vs. ground truth — 2D test set")
    out1 = joinpath(PAPER_DIR, "bias_analysis_2D_scatter.pdf")
    savefig(fig_scatter, out1);  println("Saved: $out1")

    # Histograms
    ph1 = bias_panel(Δτ_bar,
                     L"Centroid lag bias $\Delta\bar{\tau}$",
                     L"$\Delta\bar{\tau}$ [days]";    xlim_pctile=0.005)
    ph2 = bias_panel(Δv_bar,
                     L"Centroid velocity bias $\Delta\bar{v}$",
                     L"$\Delta\bar{v}$ [km/s]";       xlim_pctile=0.005, legend_pos=:topright)
    ph3 = bias_panel(Δσ_τ,
                     L"Lag extent bias $\Delta\sigma_\tau$",
                     L"$\Delta\sigma_\tau$ [days]";   xlim_pctile=0.005)
    ph4 = bias_panel(Δσ_v,
                     L"Velocity extent bias $\Delta\sigma_v$",
                     L"$\Delta\sigma_v$ [km/s]";      xlim_pctile=0.005, legend_pos=:topright)
    fig_hist = plot(ph1, ph2, ph3, ph4;
                    layout=(2,2), size=(800,760), dpi=200,
                    plot_title="(D)CNN systematic bias — 2D test set")
    out2 = joinpath(PAPER_DIR, "bias_analysis_2D_histograms.pdf")
    savefig(fig_hist, out2);  println("Saved: $out2")

    out3 = joinpath(saveDir, "bias_analysis_2D_results.jld2")
    save(out3,
         "τ_bar_true", τ_bar_true, "τ_bar_pred", τ_bar_pred,
         "v_bar_true", v_bar_true, "v_bar_pred", v_bar_pred,
         "σ_τ_true",   σ_τ_true,   "σ_τ_pred",   σ_τ_pred,
         "σ_v_true",   σ_v_true,   "σ_v_pred",   σ_v_pred,
         "Δτ_bar", Δτ_bar, "Δv_bar", Δv_bar,
         "Δσ_τ",   Δσ_τ,   "Δσ_v",   Δσ_v,
         "t", t, "v", v)
    println("Saved: $out3")
end

if abspath(PROGRAM_FILE) == @__FILE__
    dir1D = length(ARGS) >= 1 ? ARGS[1] : DEFAULT_1D_DIR
    dir2D = length(ARGS) >= 2 ? ARGS[2] : DEFAULT_2D_DIR
    dir1D != "" && isdir(dir1D) && run_1D(dir1D)
    dir2D != "" && isdir(dir2D) && run_2D(dir2D)
    run_memecho()
end
