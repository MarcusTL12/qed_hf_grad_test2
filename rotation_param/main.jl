using OhMyREPL
using Plots
using Rotations
using LinearAlgebra
plotly()

include("../common.jl")


function rotate_geo(r, θ, ϕ)
    Rθ = AngleAxis(deg2rad(θ), 1.0, 0.0, 0.0)
    Rϕ = AngleAxis(deg2rad(ϕ), 0.0, 0.0, 1.0)

    R = Rθ * Rϕ

    R * r
end

function test_h2o()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = [
        0.338169 -0.0474363 0.00728898
        1.0885 0.527633 0.00346908
        -0.41258 0.5271 0.00916178
    ]'

    freq = 0.5
    pol = [0, 1, 0]
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 44; eT="eT_clean")

    ef = make_tot_energy_function(rf)

    @time ef(r)
end

function make_h2o_surface_1d()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = [
        0.338169 -0.0474363 0.00728898
        1.0885 0.527633 0.00346908
        -0.41258 0.5271 0.00916178
    ]'

    freq = 0.5
    pol = [0, 1, 0]
    coup = 0.05

    rm("tmp", recursive=true, force=true)
    mkpath("tmp")
    efs = [
        begin
            rf = make_runner_func(
                "tmp/grad$i", freq, pol, coup, atoms, basis, 1;
                eT="eT_clean", restart=false
            )
            make_tot_energy_function(rf)
        end
        for i in 1:Threads.nthreads()
    ]

    θs = -180:180

    Es = [0.0 for _ in θs]

    Threads.@threads for i in eachindex(θs)
        th_id = Threads.threadid()
        θ = θs[i]
        Es[i] = efs[th_id](rotate_geo(r, θ, 0))
    end

    Es .-= minimum(Es)
    Es ./= 27.211396132
    Es .*= 1e6

    plot(θs, Es)

    Threads.@threads for i in eachindex(θs)
        th_id = Threads.threadid()
        θ = θs[i]
        Es[i] = efs[th_id](rotate_geo(r, 0, θ))
    end

    Es .-= minimum(Es)
    Es ./= 27.211396132
    Es .*= 1e6

    plot!(θs, Es)
end

function make_h2o_surface_2d()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = [
        0.338169 -0.0474363 0.00728898
        1.0885 0.527633 0.00346908
        -0.41258 0.5271 0.00916178
    ]'

    freq = 0.5
    pol = [0, 1, 0]
    coup = 0.05

    rm("tmp", recursive=true)
    mkpath("tmp")
    efs = [
        begin
            rf = make_runner_func(
                "tmp/grad$i", freq, pol, coup, atoms, basis, 1;
                eT="eT_clean", restart=false
            )
            make_tot_energy_function(rf)
        end
        for i in 1:Threads.nthreads()
    ]

    θs = -100:5:100
    ϕs = -100:5:100

    Es = [0.0 for _ in θs, _ in ϕs]

    inds = [(i, j) for i in eachindex(θs), j in eachindex(ϕs)]

    Threads.@threads for (i, j) in inds
        th_id = Threads.threadid()
        θ = θs[i]
        ϕ = ϕs[j]
        Es[i, j] = efs[th_id](rotate_geo(r, θ, ϕ))
    end

    Es .-= minimum(Es)
    Es ./= 27.211396132
    Es .*= 1e6

    surface(ϕs, θs, Es; size=(1600, 900))
end
