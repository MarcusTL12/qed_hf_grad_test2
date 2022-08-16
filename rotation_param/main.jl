using OhMyREPL
using Plots
using Rotations
using LinearAlgebra

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
        0.338169  -0.0474363  0.00728898
        1.0885     0.527633   0.00346908
       -0.41258    0.5271     0.00916178
    ]'

    freq = 0.5
    pol = [0, 1, 0]
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4; eT="eT_clean")

    ef = make_tot_energy_function(rf)

    @time ef(r)
end
