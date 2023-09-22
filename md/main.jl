using OhMyREPL
using LinearAlgebra
using Plots
using DSP
using Statistics
using KernelDensity
using QuadGK
using Plots.PlotMeasures

include("../common.jl")

############## MD ############

const au2s = 2.418884326585747e-17
const au2ps = au2s * 1e12

const mp = 1836

const atom_mass = Dict([
    "H" => 1.008 * mp,
    "He" => 4.0026 * mp,
    "Li" => 6.94 * mp,
    "Be" => 9.0122 * mp,
    "B" => 10.81 * mp,
    "C" => 12.011 * mp,
    "N" => 14.007 * mp,
    "O" => 15.999 * mp,
    "F" => 18.998 * mp,
    "Ne" => 20.18 * mp
])

function get_accl(masses, g)
    a = zeros(size(g))
    for (i, (c1, c2)) in enumerate(zip(eachcol(a), eachcol(g)))
        c1[:] = -c2 / masses[i]
    end
    a
end

function write_atoms(io::IO, atoms, r, v, a)
    r /= Å2B
    v /= Å2B
    for (atm, rc, vc, ac) in zip(atoms, eachcol(r), eachcol(v), eachcol(a))
        println(io, atm, "    ", rc[1], " ", rc[2], " ", rc[3],
            " ", vc[1], " ", vc[2], " ", vc[3],
            " ", ac[1], " ", ac[2], " ", ac[3])
    end
end

function calc_kin_e(masses, v)
    sum(0.5 * m * (vc ⋅ vc) for (m, vc) in zip(masses, eachcol(v)))
end

function calculate_momentum(v, masses)
    mom = zeros(Float64, 3)

    for (i, vc) in enumerate(eachcol(v))
        mom .+= vc * masses[i]
    end

    mom
end

function do_md(io::IO, n_steps, Δt, atoms, e_grad_func, r, v=zeros(size(r));
    add_first=true, t0=0.0)
    masses = [atom_mass[a] for a in atoms]

    V, g = e_grad_func(r)
    a = get_accl(masses, g)

    dip_mom = get_dipole(e_grad_func.runner_func.name)

    tot_mom = calculate_momentum(v, masses)
    drift_v = tot_mom / sum(masses)

    for vc in eachcol(v)
        vc .-= drift_v
    end

    K = calc_kin_e(masses, v)

    t = t0

    if add_first
        println(io, length(atoms))
        println(io, "i = ", 0, "; t = ", t, "; V = ", V, "; K = ", K,
            "; qed-freq = ", e_grad_func.runner_func.inp_func.freq,
            "; qed-pol = ", e_grad_func.runner_func.inp_func.pol,
            "; qed-coup = ", e_grad_func.runner_func.inp_func.coup,
            "; basis = ", e_grad_func.runner_func.inp_func.basis,
            "; Δt = ", Δt,
            "; µ = ", dip_mom)
        write_atoms(io, atoms, r, v, a)
    end

    for i in 1:n_steps
        println("Starting iteration $i")

        v_half = v + 0.5 * a * Δt
        r += v_half * Δt
        V, g = e_grad_func(r)
        a = get_accl(masses, g)
        v = v_half + 0.5 * a * Δt

        dip_mom = get_dipole(e_grad_func.runner_func.name)

        t += Δt

        tot_mom = calculate_momentum(v, masses)
        drift_v = tot_mom / sum(masses)

        for vc in eachcol(v)
            vc .-= drift_v
        end

        K = calc_kin_e(masses, v)

        println(io, length(atoms))
        println(io, "i = ", i, "; t = ", t, "; V = ", V, "; K = ", K,
            "; qed-freq = ", e_grad_func.runner_func.inp_func.freq,
            "; qed-pol = ", e_grad_func.runner_func.inp_func.pol,
            "; qed-coup = ", e_grad_func.runner_func.inp_func.coup,
            "; basis = ", e_grad_func.runner_func.inp_func.basis,
            "; Δt = ", Δt,
            "; µ = ", dip_mom)
        write_atoms(io, atoms, r, v, a)

        if isfile("stop")
            println("Stopping MD!")
            break
        end
    end
end

function get_last_conf(filename)
    t = 0.0
    freq = 0.0
    pol = ""
    coup = 0.0
    basis = ""
    Δt = 0.0

    atoms = String[]
    r = Float64[]
    v = Float64[]

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            atoms = String[]
            r = Float64[]
            v = Float64[]

            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")

            t = parse(Float64, split(ls[2], " = ")[2])
            freq = parse(Float64, split(ls[5], " = ")[2])
            pol = split(ls[6], " = ")[2]
            coup = parse(Float64, split(ls[7], " = ")[2])
            basis = split(ls[8], " = ")[2]
            Δt = parse(Float64, split(ls[9], " = ")[2])

            for _ in 1:n_atm
                l = popfirst!(lines)
                ls = split(l)
                push!(atoms, ls[1])
                append!(r, parse(Float64, rc) for rc in ls[2:4])
                append!(v, parse(Float64, vc) for vc in ls[5:7])
            end
        end
    end

    atoms,
    reshape(r, 3, length(r) ÷ 3) * Å2B,
    reshape(v, 3, length(v) ÷ 3) * Å2B,
    t,
    freq,
    eval(Meta.parse(pol)),
    coup,
    basis,
    Δt
end

function resume_md(filename, n_steps;
    Δt=nothing, freq=nothing, pol=nothing, coup=nothing, basis=nothing,
    omp=nothing, v_scale=1.0, name="grad")
    atoms, r, v, t, freq_l, pol_l, coup_l, basis_l, Δt_l = get_last_conf(filename)

    v *= v_scale

    if isnothing(freq)
        freq = freq_l
    end
    if isnothing(pol)
        pol = pol_l
    end
    if isnothing(coup)
        coup = coup_l
    end
    if isnothing(basis)
        basis = basis_l
    end
    if isnothing(Δt)
        Δt = Δt_l
    end

    pol /= norm(pol)
    rf = make_runner_func(name, freq, pol, coup, atoms, basis, omp)

    e_grad_func = make_e_and_grad_func(rf)

    open(filename, "a") do io
        do_md(io, n_steps, Δt, atoms, e_grad_func, r, v; add_first=false, t0=t)
    end
end

function keep_temp(filename, target_temp, sim_steps, avg_steps; name="grad")
    resume_md(filename, sim_steps, name=name)

    avg = get_avg_last_T(filename, avg_steps)

    while !isfile("stop")
        println("Changing temp from $avg to $target_temp")
        resume_md(filename, sim_steps; v_scale=clamp(√(target_temp / avg), 0.8, 1.2), name=name)
        avg = get_avg_last_T(filename, avg_steps)
    end
end

function resume_md_qed_ccsd(filename, n_steps;
    Δt=nothing, freq=nothing, pol=nothing, coup=nothing, basis=nothing,
    omp=nothing, v_scale=1.0, name="grad")
    atoms, r, v, t, freq_l, pol_l, coup_l, basis_l, Δt_l = get_last_conf(filename)

    v *= v_scale

    if isnothing(freq)
        freq = freq_l
    end
    if isnothing(pol)
        pol = pol_l
    end
    if isnothing(coup)
        coup = coup_l
    end
    if isnothing(basis)
        basis = basis_l
    end
    if isnothing(Δt)
        Δt = Δt_l
    end

    pol /= norm(pol)
    rf = make_runner_func_qed_ccsd(name, freq, pol, coup, atoms, basis, omp)

    e_grad_func = make_e_and_grad_func(rf)

    open(filename, "a") do io
        do_md(io, n_steps, Δt, atoms, e_grad_func, r, v; add_first=false, t0=t)
    end
end

function keep_temp_qed_ccsd(filename, target_temp, sim_steps, avg_steps; name="grad")
    resume_md_qed_ccsd(filename, sim_steps, name=name)

    avg = get_avg_last_T(filename, avg_steps)

    while !isfile("stop")
        println("Changing temp from $avg to $target_temp")
        resume_md_qed_ccsd(filename, sim_steps; v_scale=clamp(√(target_temp / avg), 0.8, 1.2), name=name)
        avg = get_avg_last_T(filename, avg_steps)
    end
end

############ ANALYSIS ########

function get_t(filename)
    ts = Float64[]
    n_atm = 0
    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")
            push!(ts, parse(Float64, split(ls[2], " = ")[2]))

            for _ in 1:n_atm
                popfirst!(lines)
            end
        end
    end
    ts
end

function get_tVK(filename)
    ts = Float64[]
    Vs = Float64[]
    Ks = Float64[]
    n_atm = 0
    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")
            t, V, K = [parse(Float64, split(ls[i], " = ")[2]) for i in 2:4]
            push!(ts, t)
            push!(Vs, V)
            push!(Ks, K)

            for _ in 1:n_atm
                popfirst!(lines)
            end
        end
    end
    ts, Vs, Ks, n_atm
end

function get_rv(filename)
    rs = Float64[]
    vs = Float64[]
    atoms = String[]

    n_atm = 0
    n_frames = 0

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_frames += 1
            n_atm = parse(Int, popfirst!(lines))
            popfirst!(lines)

            atoms = String[]

            for _ in 1:n_atm
                l = popfirst!(lines)
                ls = split(l)
                push!(atoms, ls[1])
                append!(rs, parse(Float64, n) for n in ls[2:4])
                append!(vs, parse(Float64, n) for n in ls[5:7])
            end
        end
    end

    reshape(rs, 3, n_atm, n_frames),
    reshape(vs, 3, n_atm, n_frames),
    atoms
end

function get_µ(filename)
    µs = Float64[]

    n_atm = 0
    n_frames = 0

    open(filename) do io
        lines = Iterators.Stateful(eachline(io))

        while !isempty(lines)
            n_atm = parse(Int, popfirst!(lines))
            l = popfirst!(lines)
            ls = split(l, "; ")
            µ = get(ls, 10, missing)

            if !ismissing(µ)
                append!(µs, eval(Meta.parse(µ)))
            end

            for _ in 1:n_atm
                popfirst!(lines)
            end
        end
    end

    reshape(µs, 3, length(µs) ÷ 3)
end

function plot_tVK(filename; is=:, time=false)
    ts, Vs, Ks = get_tVK(filename)

    E0 = Vs[1] + Ks[1]
    @show E0
    Vs .-= E0

    if time
        plot(ts[is], Vs[is]; label="Potential", leg=:bottomleft)
        plot!(ts[is], Ks[is]; label="Kinetic")
        plot!(ts[is], (Vs+Ks)[is]; label="Total")
    else
        plot(Vs[is]; label="Potential", leg=:bottomleft)
        plot!(Ks[is]; label="Kinetic")
        plot!((Vs+Ks)[is]; label="Total")
    end
end

function plot_VK_overlay(filename; is=:)
    ts, Vs, Ks = get_tVK(filename)

    E0 = Vs[1] + Ks[1]
    @show E0
    Vs .-= E0

    plot(eachindex(ts)[is], -Vs[is]; label="Potential", leg=:bottomleft)
    plot!(eachindex(ts)[is], Ks[is]; label="Kinetic")
    # plot!(ts, Vs + Ks; label="Total")
end

function calculate_T_instant(Ks, N_atm)
    2Ks / (kB * 3(N_atm - 1))
end

function plot_T(filename)
    ts, _, Ks, n_atm = get_tVK(filename)

    Ts = calculate_T_instant(Ks, n_atm)

    plot(ts, Ts; ylabel="T [K]")
end

function get_avg_last_T(filename, n)
    _, _, Ks, n_atm = get_tVK(filename)
    Ts = calculate_T_instant(Ks, n_atm)
    mean(@view Ts[end-(n-1):end])
end

function compare_T(filenames)
    plot(; ylabel="T [K]", leg=:bottomright)

    for (i, filename) in enumerate(filenames)
        ts, _, Ks, n_atm = get_tVK(filename)
        Ts = calculate_T_instant(Ks, n_atm)
        plot!(ts, Ts; label="$i")
    end
    plot!()
end

function calc_T_window_avg(Ts, n)
    w = ones(Float64, n) / n

    conv(Ts, w)[1:length(Ts)]
end

function plot_T_window_avg(filename, n)
    ts, _, Ks, n_atm = get_tVK(filename)

    Ts = calculate_T_instant(Ks, n_atm)

    T_avg = calc_T_window_avg(Ts, n)

    @show last(T_avg)
    @show extrema(T_avg[end-n+1:end])

    plot(ts, T_avg; leg=false)
end

function compare_T_window_avg(filenames, n)
    plot(; ylabel="T [K]", leg=:bottomright)

    for (i, filename) in enumerate(filenames)
        ts, _, Ks, n_atm = get_tVK(filename)
        Ts = calculate_T_instant(Ks, n_atm)
        T_avg = calc_T_window_avg(Ts, n)
        plot!(ts, T_avg; label="$i")
    end
    plot!()
end

function compare_E(filenames)
    plot(; ylabel="E", leg=:bottomright)

    for (i, filename) in enumerate(filenames)
        ts, Vs, Ks, n_atm = get_tVK(filename)

        E0 = Vs[1] + Ks[1]
        Vs .-= E0

        plot!(Vs + Ks; label="$i")
    end

    plot!()
end

function calculate_total_momentum(filename)
    _, vs, atoms = get_rv(filename)
    masses = [atom_mass[a] for a in atoms]

    moms = Float64[]

    for f in 1:size(vs, 3)
        append!(moms, calculate_momentum((@view vs[:, :, f]), masses))
    end

    reshape(moms, 3, size(vs, 3))
end

function calculate_center_of_mass_conf(r, atoms)
    masses = [atom_mass[a] for a in atoms]

    com = zeros(Float64, 3)

    for (i, rc) in enumerate(eachcol(r))
        com .+= rc * masses[i]
    end

    com / sum(masses)
end

function calculate_center_of_mass(filename)
    rs, _, atoms = get_rv(filename)

    coms = Float64[]

    for f in 1:size(rs, 3)
        append!(coms, calculate_center_of_mass_conf((@view rs[:, :, f]), atoms))
    end

    reshape(coms, 3, size(rs, 3))
end

function calculate_trans_E(filename)
    _, _, atoms = get_rv(filename)
    moms = calculate_total_momentum(filename)

    mass = sum(atom_mass[a] for a in atoms)

    Es = Float64[]

    for mom in eachcol(moms)
        push!(Es, mom'mom / (2 * mass))
    end

    Es
end

function calculate_ang_mom(r, v, com, atoms)
    masses = [atom_mass[a] for a in atoms]

    am = zeros(Float64, 3)

    for (i, (rc, vc)) in enumerate(zip(eachcol(r), eachcol(v)))
        am .+= ((rc - com) × vc) * masses[i]
    end

    am
end

function calculate_tot_ang_mom(filename)
    rs, vs, atoms = get_rv(filename)
    coms = calculate_center_of_mass(filename)

    ams = Float64[]

    for f in 1:size(rs, 3)
        @views append!(ams,
            calculate_ang_mom(rs[:, :, f], vs[:, :, f], coms[:, f], atoms))
    end

    reshape(ams, 3, size(rs, 3))
end

function calc_mom_intertia(r, L, atoms)
    masses = [atom_mass[a] for a in atoms]

    I = 0.0

    for (i, rc) in enumerate(eachcol(r))
        I += masses[i] * (rc'rc - (L'rc)^2 / L'L)
    end

    I
end

function calculate_rot_energy(filename)
    rs, _, atoms = get_rv(filename)
    ams = calculate_tot_ang_mom(filename)

    Es = Float64[]

    for (f, L) in enumerate(eachcol(ams))
        I = @views calc_mom_intertia(rs[:, :, f], L, atoms)
        @views push!(Es, L'L / 2I)
    end

    Es
end

function calculate_radial_dist(r, atoms, from_atm, to_atm)
    dists = Float64[]

    for i in 1:length(atoms)
        atm1 = atoms[i]
        if atm1 == from_atm
            r1 = @view r[:, i]

            range2 = if from_atm == to_atm
                (i+1):length(atoms)
            else
                1:length(atoms)
            end

            for j in range2
                atm2 = atoms[j]
                if atm2 == to_atm
                    r2 = @view r[:, j]
                    push!(dists, norm(r1 - r2))
                end
            end
        end
    end

    dists
end

function get_last_n_radial_dist(filename, from_atm, to_atm, n, spacing=1)
    r, _, atoms = get_rv(filename)

    dists = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(dists, calculate_radial_dist((@view r[:, :, i]),
            atoms, from_atm, to_atm))
    end

    dists
end

function plot_dist!(data; label="")
    xs = range(extrema(data)...; length=1000)

    U = kde(data)

    plot!(xs, x -> pdf(U, x); label=label)
end

function plot_rad_dist!(data; label="", nrm=false, end_override=nothing)
    x_begin, x_end = extrema(data)

    if !isnothing(end_override)
        x_end = end_override
    end

    xs = range(x_begin, x_end; length=1000)

    U = kde(data)

    f = x -> pdf(U, x) / x

    g = if nrm
        integral = quadgk(f, x_begin, x_end)[1]
        x -> f(x) / integral
    else
        f
    end

    plot!(xs, g; label=label)
end

function calculate_std_dev_mass(r, atoms)
    com = calculate_center_of_mass_conf(r, atoms)

    dev = zeros(Float64, 3)

    for (rc, atm) in zip(eachcol(r), atoms)
        dev += atom_mass[atm] * (rc - com) .^ 2
    end

    dev /= sum(atom_mass[atom] for atom in atoms)

    sqrt.(dev)
end

function get_last_n_std_dev_mass(filename, n, spacing=1)
    r, _, atoms = get_rv(filename)

    devs = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(devs, calculate_std_dev_mass((@view r[:, :, i]), atoms))
    end

    reshape(devs, 3, length(devs) ÷ 3)
end

function get_std_dev_mass(filename)
    r, _, atoms = get_rv(filename)

    devs = Float64[]

    for i in axes(r, 3)
        append!(devs, calculate_std_dev_mass((@view r[:, :, i]), atoms))
    end

    reshape(devs, 3, length(devs) ÷ 3)
end

function calculate_dev_from_pol_h2o(r, pol)
    devs = Float64[]

    for i in 1:3:size(r, 2)
        @views oh1 = r[:, i+1] - r[:, i]
        @views oh2 = r[:, i+2] - r[:, i]

        plane_vec = oh1 × oh2

        plane_vec /= norm(plane_vec)

        push!(devs, abs(plane_vec ⋅ pol))
    end

    devs
end

function get_last_n_dev_from_pol(filename, pol, n, spacing=1)
    r, _, atoms = get_rv(filename)

    devs = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(devs, calculate_dev_from_pol_h2o((@view r[:, :, i]), pol))
    end

    devs
end

function calculate_dip_dev_from_pol_h2o(r, pol)
    devs = Float64[]

    for i in 1:3:size(r, 2)
        @views oh1 = r[:, i+1] - r[:, i]
        @views oh2 = r[:, i+2] - r[:, i]

        dip_vec = oh1 + oh2

        dip_vec /= norm(dip_vec)

        push!(devs, abs(dip_vec ⋅ pol))
    end

    devs
end

function get_last_n_dip_dev_from_pol(filename, pol, n, spacing=1)
    r, _, atoms = get_rv(filename)

    devs = Float64[]

    rng = 1:size(r, 3)
    rng = rng[end-(spacing*n-1):spacing:end]

    for i in rng
        append!(devs, calculate_dip_dev_from_pol_h2o((@view r[:, :, i]), pol))
    end

    devs
end

function plot_µ_hist(filename)
    µs = get_µ(filename)

    plot(@view µs[1, :]; label="x")
    plot!(@view µs[2, :]; label="y")
    plot!(@view µs[3, :]; label="z")
end

############ NICE PRESENTABLE PLOTS ##################

function make_rad_dist_func(data, x_begin, x_end)
    U = kde(data)

    f = x -> pdf(U, x) / x

    integral = quadgk(f, x_begin, x_end)[1]

    x -> f(x) / integral
end

function plot_rad_dist_h2o()
    plot(; layout=(3, 1), size=(500, 500))
    spacing = 1
    n = 4000 ÷ spacing
    x_begin = 0.5
    x_end = 6
    xs = range(x_begin, x_end; length=1000)

    plot_nr = 1

    function add_plot!(a1, a2)
        data_c = get_last_n_radial_dist("md/many_h2o/30h2o_0.1.xyz", a1, a2, n, spacing)
        data_f = get_last_n_radial_dist("md/many_h2o/30h2o_free.xyz", a1, a2, n, spacing)

        f_c = make_rad_dist_func(data_c, x_begin, x_end)
        f_f = make_rad_dist_func(data_f, x_begin, x_end)

        plot!(xs, f_c; subplot=plot_nr, label="0.1", ylabel="$a1-$a2")
        plot!(xs, f_f; subplot=plot_nr, label="0.0")

        plot_nr += 1

        plot!()
    end

    add_plot!("H", "H")
    add_plot!("O", "H")
    add_plot!("O", "O")
end

function plot_plan_dev()
    plot(; size=(500, 500), leg=:topleft)
    spacing = 1
    n = 4000 ÷ spacing
    plot_dist!(get_last_n_dev_from_pol("md/many_h2o/30h2o_0.1.xyz", [0, 1, 0], n, spacing); label="0.1")
    plot_dist!(get_last_n_dev_from_pol("md/many_h2o/30h2o_free.xyz", [0, 1, 0], n, spacing); label="0.0")
end

function plot_mass_dev()
    plot(; layout=(3, 1), size=(500, 500), leg=:topleft)

    spacing = 1
    n = 4000 ÷ spacing

    data_c = get_last_n_std_dev_mass("md/many_h2o/30h2o_0.1.xyz", n, spacing)
    data_f = get_last_n_std_dev_mass("md/many_h2o/30h2o_free.xyz", n, spacing)

    xs = range(1000, 4000; length=1000)

    function add_plot!(ax)
        data_c_ax = @view data_c[ax, :]
        data_f_ax = @view data_f[ax, :]

        U_c = kde(data_c_ax)
        U_f = kde(data_f_ax)

        f_c = x -> pdf(U_c, x)
        f_f = x -> pdf(U_f, x)

        plot!(xs, f_c; label="0.1", subplot=ax, ylabel=string("xyz"[ax]))
        plot!(xs, f_f; label="0.0", subplot=ax)
    end

    add_plot!(1)
    add_plot!(2)
    add_plot!(3)
end

function plot_mass_dev_history()
    plot(; layout=(3, 1), link=:x, size=(1000, 600), leg=:topright,
        left_margin=50px, bottom_margin=30px)
    plot!(; subplot=3, xlabel="t [ps]")

    data_c = get_std_dev_mass("md/many_h2o/30h2o_0.1.xyz")
    t_c = get_t("md/many_h2o/30h2o_0.1.xyz") * au2ps
    data_f = get_std_dev_mass("md/many_h2o/30h2o_free.xyz")
    t_f = get_t("md/many_h2o/30h2o_free.xyz") * au2ps

    rgb = (:red, :green, :blue)

    labels = [
        ("0.1", "0.0"),
        (nothing, nothing),
        (nothing, nothing)
    ]

    yl = [0.0, 4.0]

    function add_plot!(ax)
        data_c_ax = @view data_c[ax, :]
        data_f_ax = @view data_f[ax, :]

        plot!(t_c, data_c_ax; label=labels[ax][1], subplot=ax, ylabel=string("xyz"[ax]),
            yguidefont=font(rgb[ax], :bold, rotation=-90.0, pointsize=20), ylims=yl)
        plot!(t_f, data_f_ax; label=labels[ax][2], subplot=ax)
    end

    add_plot!(1)
    add_plot!(2)
    add_plot!(3)
end

############ TESTS ###########

function test_h2o()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = Float64[
        0.0605815 0.999184 -0.059765
        0.0605815 -0.059765 0.999184
        2.8514e-8 -3.26305e-8 4.11651e-9
    ] * Å2B

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/h2o_anims/$(coup)_$basis.xyz", "w") do io
        do_md(io, 1000, 25.0, atoms, e_grad_func, r)
    end
end

function test_h2o_wobble()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = Float64[
        -2.08349 -1.1156 -2.36234
        1.03247 0.99867 0.357
        0.0137 -0.01814 -0.62266
    ] * Å2B

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.0

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/h2o_anims/test.xyz", "w") do io
        do_md(io, 100, 10.0, atoms, e_grad_func, r)
    end
end

function test_2h2o()
    atoms = split_atoms("OHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        0.00074 -0.00014 0.00003
        0.03041 0.10161 0.97727
        -0.27764 -0.92380 -0.08655
        -0.00484 0.00131 2.75024
        0.78062 -0.00539 3.32392
        -0.68819 0.38765 3.32456
    ]' * Å2B

    freq = 0.5
    pol = [0.1, 1, 0.1]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/2h2o_anims/$(coup)_$basis.xyz", "w") do io
        do_md(io, 10, 10.0, atoms, e_grad_func, r)
    end
end

function test_3h2o()
    atoms = split_atoms("OHHOHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        -1.99995 1.33519 0.00206
        -1.03334 1.26848 0.01524
        -2.29042 0.64850 0.62097
        1.37734 0.83744 0.08635
        2.33328 0.78267 -0.06307
        1.17761 0.05034 0.61503
        1.37730 4.83740 0.08630
        2.33330 4.78270 -0.06310
        1.17760 4.05030 0.61500
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/3h2o_anims/$(coup)_$basis.xyz", "w") do io
        do_md(io, 10, 10.0, atoms, e_grad_func, r)
    end
end

function test_7h2o()
    atoms = split_atoms("OHHOHHOHHOHHOHHOHHOHH")
    basis = "cc-pvdz"
    r = Float64[
        -2.76420 -0.72774 -1.45990
        -2.21757 -0.06831 -0.95981
        -2.73768 -0.31471 -2.34353
        -1.22859 1.09937 -0.13416
        -0.44342 0.54547 0.12127
        -0.82078 1.98081 -0.02962
        -5.18189 0.42640 0.26551
        -4.61686 -0.26208 -0.13692
        -6.00859 -0.04387 0.45657
        -2.11075 -3.14665 -0.38463
        -2.32780 -2.29126 -0.83165
        -2.99498 -3.54743 -0.33553
        0.49472 -2.93885 0.29290
        0.73698 -3.88059 0.29522
        -0.47329 -3.01635 0.09410
        -3.63850 2.67099 0.09481
        -2.90992 2.02140 0.11762
        -4.40376 2.06407 0.20872
        1.00864 -0.29512 0.33482
        1.98068 -0.32415 0.31304
        0.81598 -1.26755 0.37072
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.01

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 4)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/7h2o.xyz", "w") do io
        do_md(io, 10, 20.0, atoms, e_grad_func, r)
    end
end

function test_10h2o()
    atoms = split_atoms("OHH"^10)
    basis = "cc-pvdz"
    r = Float64[
        -1.36725 -0.92663 -1.90879
        -1.74234 -0.11926 -1.45603
        -1.26991 -0.53860 -2.80044
        -3.22927 -2.38220 -0.41777
        -2.63780 -2.04056 -1.12709
        -2.57307 -2.56072 0.28293
        -2.64427 1.14824 -0.85354
        -2.42916 1.95307 -0.31878
        -3.50371 0.91709 -0.43548
        0.33397 0.18985 1.13976
        0.57476 -0.73762 0.90050
        -0.56818 0.01346 1.50704
        1.16183 1.19691 -1.34410
        1.78383 0.50993 -1.63489
        0.72519 0.74469 -0.58806
        0.58685 -2.04755 -0.37608
        0.82665 -2.88213 -0.81286
        -0.11466 -1.70341 -0.97980
        -2.14222 3.15189 0.94066
        -1.15601 3.20465 0.86100
        -2.27686 3.72975 1.71058
        -4.62220 -0.26519 0.47588
        -4.32243 -1.12357 0.08076
        -5.58760 -0.35347 0.38297
        -2.18761 -0.53766 1.97494
        -2.93282 -0.22087 1.41557
        -2.62261 -0.57164 2.84473
        0.53011 2.94997 0.77347
        0.88772 2.78719 -0.12816
        0.64677 2.04447 1.14681
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    pol = pol / norm(pol)
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 48)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/10h2o_0.05.xyz", "w") do io
        do_md(io, 3, 40.0, atoms, e_grad_func, r)
    end
end

function test_20h2o()
    atoms = split_atoms("OHH"^20)
    basis = "cc-pvdz"
    r = Float64[
        -4.10708 -0.59028 1.09302
        -3.35828 -0.42162 1.70466
        -4.65083 -1.22593 1.62529
        -1.55147 0.80097 1.32250
        -0.60809 1.09225 1.28322
        -1.54094 0.23115 0.51465
        -3.31957 2.84156 1.31202
        -2.69302 2.07763 1.40890
        -3.09921 3.34560 2.11670
        -5.91520 -2.39777 1.96031
        -6.29883 -3.27661 2.18525
        -6.74820 -1.97369 1.63425
        -3.51597 -1.74516 -1.30834
        -2.65476 -1.28598 -1.47388
        -3.75644 -1.26200 -0.48157
        -1.11930 -0.50871 -1.11091
        -0.35691 -1.13493 -1.03793
        -0.68763 0.24291 -1.57946
        -5.43437 -3.62526 -0.64297
        -5.04722 -2.92460 -1.21468
        -5.33602 -3.18950 0.23093
        -5.20762 1.53250 -0.20021
        -4.82975 0.71100 0.20321
        -4.82010 2.18110 0.43407
        0.93402 1.57687 0.40754
        1.65636 1.05117 -0.01379
        0.53769 1.95834 -0.41251
        2.18577 -0.20070 2.28319
        2.71239 -0.24533 1.45470
        1.70447 0.63681 2.11649
        -7.54663 0.38725 -1.00831
        -6.75914 0.90291 -0.70306
        -7.57429 0.66059 -1.94213
        -0.07584 -4.63225 -1.53491
        0.00616 -5.54496 -1.86053
        -1.06291 -4.57391 -1.45856
        -7.64513 -4.33674 0.94991
        -7.04397 -4.45802 0.18357
        -8.08645 -3.49725 0.69850
        -2.87349 2.74617 -1.45164
        -2.98052 3.06667 -0.52658
        -3.72554 2.26660 -1.53877
        2.73200 -0.29075 -0.50400
        3.45098 -0.55688 -1.10593
        2.11485 -1.05868 -0.62766
        -0.34820 2.04718 -1.99494
        -0.19156 2.53141 -2.82679
        -1.28703 2.32819 -1.82224
        -2.76085 -4.42778 -1.41518
        -2.94899 -3.46279 -1.48862
        -3.65585 -4.72904 -1.15070
        -8.22768 -1.57532 0.68895
        -7.94392 -0.93223 -0.01331
        -9.09068 -1.18096 0.91417
        0.95768 -2.60679 2.01643
        0.87012 -3.03437 2.88500
        1.38334 -1.75232 2.27407
        0.93890 -2.29545 -0.67166
        0.87353 -2.56510 0.28390
        0.67586 -3.16797 -1.07064
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    # pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 12)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/20h2o_0.1.xyz", "w") do io
        do_md(io, 1, 50.0, atoms, e_grad_func, r)
    end
end

function test_ethylene()
    atoms = split_atoms("CCHHHH")
    basis = "aug-cc-pvdz"
    r = Float64[
        0.0000 0.0000 0.0000
        1.3400 0.0000 0.0000
        1.8850 0.9440 0.0000
        1.8850 -0.9440 0.0000
        -0.5450 0.9440 0.0000
        -0.5450 -0.9440 0.0000
    ]' * Å2B

    freq = 0.5
    pol = [1, 0.1, 0.1]
    pol /= norm(pol)
    coup = 0.05

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 12)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/matteo/ethylene.xyz", "w") do io
        do_md(io, 10, 10.0, atoms, e_grad_func, r)
    end
end

function test_50h2o()
    atoms = split_atoms("OHH"^50)
    basis = "cc-pvdz"
    r = Float64[
        -3.97466 1.83162 0.40399
        -3.37232 1.90121 1.18706
        -4.42389 0.98318 0.65446
        0.62218 1.18643 -0.27378
        1.23063 1.44448 -1.01094
        0.39518 2.08655 0.06701
        -2.60233 -1.74213 0.78937
        -1.78938 -2.29143 0.88561
        -2.30519 -1.19603 0.01863
        -5.15898 -0.67801 0.65109
        -4.43289 -1.34438 0.62849
        -5.57811 -0.86609 -0.22696
        -3.10217 4.23547 -0.77087
        -3.45227 3.38714 -0.40921
        -2.93609 4.68209 0.09763
        -0.00827 3.75536 0.47217
        0.50171 3.87004 1.31226
        -0.87739 4.08258 0.81444
        -1.48946 -0.24390 -1.19342
        -0.83122 0.32381 -0.71764
        -0.91765 -0.50877 -1.94852
        2.09877 -0.88753 0.75386
        2.67426 -1.11119 -0.01834
        1.59401 -0.12013 0.38458
        0.03571 -2.67449 1.02259
        0.82781 -2.08026 1.02044
        0.20061 -3.09739 0.14415
        2.40802 1.35916 -2.26106
        3.38105 1.26021 -2.13723
        2.44018 2.14978 -2.85712
        -3.01594 -0.56736 3.26943
        -2.24695 -1.04080 3.66371
        -3.02872 -1.02398 2.39273
        -0.53678 1.68372 -3.95220
        -0.71460 2.43666 -3.34085
        -1.01722 2.00361 -4.74971
        -0.80268 3.89173 -2.20093
        -0.39098 3.86473 -1.30150
        -1.73376 4.06997 -1.90664
        0.99796 -2.86267 -1.60035
        0.79391 -1.98397 -1.99110
        1.97250 -2.75452 -1.48552
        -2.21015 1.93438 2.59177
        -1.27770 1.68420 2.79000
        -2.65590 1.18305 3.05627
        -0.53811 -1.77520 3.60015
        -0.17645 -2.47733 4.17325
        -0.39354 -2.18488 2.71178
        0.71630 5.04371 -4.10489
        0.13780 4.59917 -3.43170
        0.49518 5.97523 -3.91419
        -2.32053 4.55673 1.76570
        -2.45311 3.71370 2.26674
        -1.99730 5.12087 2.51462
        0.59809 -0.52012 -3.09873
        1.36731 0.03846 -2.81193
        0.12739 0.21495 -3.58516
        0.50203 3.84169 -6.59368
        0.55071 4.33269 -5.73767
        -0.41937 3.49022 -6.51187
        -5.83729 -0.83504 -1.96615
        -5.12358 -0.22795 -2.27364
        -6.50649 -0.70170 -2.65908
        -3.75905 0.93841 -2.24136
        -2.92579 0.47223 -1.98749
        -3.91704 1.41159 -1.39080
        3.06766 3.54722 -3.80335
        3.17490 3.31829 -4.75853
        2.34679 4.21656 -3.88779
        3.05122 0.07662 3.10140
        3.99854 -0.20539 3.04531
        2.73291 -0.38057 2.28213
        0.35535 0.89704 3.44813
        1.30415 0.62302 3.45153
        -0.04168 0.00483 3.59768
        -1.98671 2.79920 -6.09807
        -2.73680 2.93636 -5.45434
        -2.50429 2.38979 -6.81773
        3.05546 2.63067 -6.39628
        3.27848 3.16535 -7.20031
        2.07421 2.68672 -6.49357
        5.34671 3.23974 1.46300
        5.33893 4.21304 1.40228
        4.59589 3.10278 2.09354
        5.62010 -0.30431 2.20802
        6.31748 -0.93988 1.92113
        5.50774 0.16164 1.34389
        -3.95463 2.70504 -4.36839
        -3.92485 2.00218 -3.67464
        -4.54867 3.33813 -3.88867
        -5.53597 0.68892 2.97262
        -5.68712 0.09753 2.20261
        -4.76823 0.23643 3.38093
        5.06432 1.95044 -2.69645
        5.08609 1.50451 -3.58004
        4.64410 2.79762 -2.98118
        5.87659 1.11770 -0.20443
        5.74850 1.98377 0.25735
        5.67440 1.40853 -1.12955
        2.46483 4.31348 -8.42837
        2.21485 4.84982 -9.19877
        1.62191 4.29920 -7.91479
        4.50599 0.69994 -5.08862
        4.03164 -0.08295 -5.46234
        4.09041 1.39793 -5.65473
        0.94959 3.61324 3.05765
        1.92391 3.50029 3.19645
        0.66791 2.69646 3.29873
        3.47348 2.78843 3.44300
        3.35539 1.80603 3.45499
        3.97022 2.88822 4.29792
        5.17966 2.77569 5.52825
        5.90464 2.44033 4.94187
        5.63215 2.77679 6.38888
        7.32775 -1.18903 0.24250
        6.70445 -1.73381 -0.29379
        7.07835 -0.29102 -0.08106
        0.97552 -2.84713 -4.53486
        0.63633 -1.99044 -4.17993
        0.89541 -3.39015 -3.72012
        -5.25900 4.17529 -2.52879
        -4.51402 4.49324 -1.97057
        -5.88609 3.92307 -1.80847
        -6.33339 3.29377 -0.15919
        -5.58778 2.76124 0.19325
        -6.97711 3.20999 0.56659
        -0.86258 5.50529 3.79979
        -0.57018 5.86248 4.65510
        -0.12829 4.87814 3.59233
        8.73680 -3.30378 -0.98790
        9.67060 -3.51618 -0.82376
        8.59752 -2.53575 -0.38934
        6.74009 1.83385 3.57690
        6.60812 2.34503 2.74980
        6.46129 0.94336 3.26296
        5.20920 -1.64012 -3.66169
        5.41342 -0.73929 -4.00019
        4.68523 -1.95520 -4.43883
        3.55736 -1.92878 -1.37671
        4.31938 -2.48175 -1.06669
        3.98410 -1.62667 -2.21508
        6.05845 -2.95984 -1.44429
        6.93803 -3.41229 -1.46277
        6.00554 -2.65093 -2.38220
        3.36686 -1.78260 -5.66357
        2.56848 -2.08760 -5.17095
        3.14693 -2.23721 -6.51480
        1.72411 -3.35800 -7.14504
        1.24151 -3.34159 -6.28610
        1.16252 -3.94605 -7.67649
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    # pol = pol / norm(pol)
    coup = 0.0

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 80)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/50h2o_free.xyz", "w") do io
        do_md(io, 1, 40.0, atoms, e_grad_func, r)
    end
end

function test_30h2o()
    atoms = split_atoms("OHH"^30)
    basis = "cc-pvdz"
    r = Float64[
        -5.93743 -0.45233 1.48921
        -6.33336 -0.78457 0.64673
        -5.86642 0.50401 1.27179
        0.18981 -2.68496 -0.54076
        0.91260 -2.82305 -1.19111
        0.14421 -1.69898 -0.59281
        2.10831 -1.57333 -2.43000
        3.03709 -1.35744 -2.22835
        1.64693 -0.84724 -1.93708
        1.61265 3.30699 1.21524
        2.20104 3.84719 1.80200
        1.39949 2.58820 1.85424
        -2.09708 3.92455 0.81232
        -1.36901 4.39917 0.34831
        -2.82124 4.04535 0.15494
        -4.78235 1.89763 0.49696
        -4.19934 1.21135 0.09897
        -4.64347 2.62676 -0.15111
        0.46282 0.02733 -1.04517
        -0.22312 0.40597 -1.65012
        0.25116 0.59013 -0.25849
        -2.85437 0.02112 -0.32849
        -2.48762 0.36424 -1.18662
        -2.13366 0.39810 0.24432
        -2.38647 -3.66198 -0.86503
        -1.43310 -3.41834 -0.80345
        -2.52278 -4.00402 0.04889
        0.22713 4.55610 -0.69464
        0.71051 3.92928 -0.08857
        0.85499 5.30166 -0.64343
        -2.10318 1.47560 5.00578
        -1.96415 0.61390 4.54864
        -2.69473 1.90638 4.33587
        -3.38280 2.53497 2.86574
        -2.89897 3.18772 2.30972
        -4.07374 2.26332 2.21981
        -0.24624 -2.89308 2.21954
        -0.98497 -3.53818 2.14524
        0.07551 -2.90360 1.28987
        -1.28985 0.34034 -5.07654
        -0.59703 -0.37133 -5.05062
        -2.08760 -0.24967 -5.12385
        -1.58925 3.81289 -2.66533
        -1.14813 3.76306 -3.55321
        -0.83173 4.12718 -2.11049
        0.49701 -1.66282 -4.68154
        1.12643 -1.61851 -3.92310
        0.49487 -2.62052 -4.85238
        -1.63923 1.05824 -2.44603
        -1.59428 0.82083 -3.40431
        -1.69041 2.04027 -2.53198
        -1.24376 -0.70665 3.46690
        -0.71727 -1.49603 3.17687
        -2.09139 -0.95058 3.02136
        -0.91642 1.42189 0.84620
        -0.44210 1.42241 1.71297
        -1.29393 2.33391 0.89468
        -3.91068 3.86996 -1.31076
        -3.12908 3.82035 -1.92030
        -4.51173 4.41223 -1.85467
        0.56486 1.44073 3.16534
        0.61571 1.88105 4.04356
        0.09990 0.61148 3.43745
        -0.56388 2.98048 -5.02626
        -0.07374 3.10221 -5.85785
        -0.85023 2.03870 -5.13240
        -3.37871 -1.33658 -4.63868
        -3.59947 -1.46187 -3.68410
        -3.95215 -2.01499 -5.03474
        -3.53778 -1.47344 1.99952
        -3.37950 -1.03013 1.13259
        -4.48153 -1.17997 2.11183
        -6.57844 -1.32973 -1.02181
        -5.73877 -1.59307 -1.46496
        -7.23772 -1.67235 -1.64777
        -2.74366 -4.04384 1.86023
        -3.35882 -4.53649 2.43367
        -3.09102 -3.12224 1.98414
        0.18475 2.39129 5.88571
        0.25474 1.77621 6.63998
        -0.76125 2.20660 5.63202
        2.96907 4.66899 3.10565
        3.79723 4.92260 3.54747
        2.34310 4.73978 3.87085
        1.38071 4.73472 5.26948
        0.69310 5.29535 5.66964
        1.03171 3.84138 5.51416
        -4.04443 -1.92427 -2.00650
        -3.53588 -2.73132 -1.72367
        -3.65224 -1.29813 -1.35189
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    # pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 43)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/many_h2o/30h2o_0.1.xyz", "w") do io
        do_md(io, 1, 40.0, atoms, e_grad_func, r)
    end
end

function test_pna_30h2o()
    atoms = split_atoms("C"^6 * "H"^4 * "NHHNOO" * "OHH"^30)
    basis = "cc-pvdz"
    r = Float64[
        -4.76493 2.19929 -0.13922
        -3.64625 1.36944 -0.07466
        -2.46417 1.80738 0.52639
        -2.44146 3.05770 1.14711
        -3.55820 3.89501 1.10004
        -4.72027 3.45908 0.45829
        -5.66108 1.82955 -0.62452
        -3.51049 4.86675 1.58168
        -1.55536 3.38558 1.68549
        -3.71026 0.36970 -0.49995
        -1.39478 0.92863 0.67490
        -0.48396 1.34876 0.85802
        -1.37339 0.12491 0.04347
        -5.91316 4.30914 0.45279
        -6.81643 4.04440 -0.35565
        -5.95725 5.24304 1.27321
        -3.63753 4.98966 -2.46066
        -2.78522 4.52479 -2.25598
        -3.29779 5.88216 -2.64707
        0.76240 4.64917 -0.34073
        0.82445 3.90220 0.31593
        1.55526 5.14874 -0.06654
        1.27407 2.66487 1.31782
        1.67801 1.88571 0.86664
        1.27959 2.33565 2.24973
        -2.35633 -0.56771 2.90388
        -2.06319 -0.07144 2.10071
        -1.90485 -1.42399 2.72001
        -0.50195 -2.55736 2.08134
        0.29052 -1.95841 2.13594
        -0.16029 -3.32231 2.58020
        -1.78944 0.99028 -2.73936
        -2.74985 0.83462 -2.92080
        -1.83496 1.90358 -2.37504
        -4.36325 0.66251 -3.67824
        -5.18187 0.29341 -3.25005
        -4.68800 1.58841 -3.81559
        -4.89082 3.28743 -4.09426
        -5.78249 3.67363 -4.18591
        -4.50891 3.91557 -3.42754
        1.51682 -0.85033 2.14978
        1.48473 -0.06597 2.74431
        1.87052 -0.43842 1.32735
        -1.20311 3.93888 -2.17178
        -0.53263 4.17612 -1.48840
        -0.60922 3.69783 -2.92140
        -4.86560 4.93721 4.04586
        -4.16255 5.63180 3.98195
        -5.49071 5.27818 3.37341
        -4.95759 0.38337 3.15468
        -4.05680 -0.00399 3.03793
        -4.71608 1.13259 3.74803
        1.09891 1.42474 3.73670
        1.46908 1.49633 4.63741
        0.13255 1.48419 3.97322
        -1.46851 1.47619 4.49103
        -1.85393 0.65884 4.09253
        -2.31031 1.96597 4.64986
        -4.07409 2.48507 4.79027
        -4.49369 2.58923 5.66542
        -4.31852 3.36929 4.40728
        0.24529 2.96667 -4.37690
        -0.57599 3.17243 -4.89633
        0.39192 2.06044 -4.74143
        -2.87035 0.20024 -5.89186
        -3.42502 0.20773 -5.07051
        -3.42306 -0.35943 -6.46732
        -2.27809 2.93180 -5.43612
        -2.45357 2.03590 -5.80279
        -3.17306 3.15206 -5.09641
        0.16049 -0.75820 -2.02373
        0.46718 -0.84956 -2.95251
        -0.56482 -0.10121 -2.18145
        -0.27933 0.27858 -4.86128
        -0.89747 0.56468 -4.14672
        -0.95475 0.05869 -5.54195
        -8.24072 1.56335 -0.93256
        -8.08460 1.27711 0.00298
        -8.16215 2.53199 -0.81294
        -7.32477 0.80109 1.51676
        -6.55635 0.90466 2.12408
        -7.35290 -0.18719 1.53405
        -6.66655 -0.02691 -2.46303
        -6.87324 -0.81023 -1.91943
        -7.32881 0.60320 -2.07909
        1.87638 0.50382 -0.27150
        2.11281 1.13578 -1.00087
        1.32818 -0.10984 -0.82168
        1.91355 2.49288 -2.12397
        1.68370 3.35376 -1.72455
        1.53400 2.61077 -3.02427
        -1.69372 -1.95707 -0.29688
        -1.21427 -2.28335 0.50271
        -0.94468 -1.84372 -0.92492
        -1.34997 6.74588 1.26739
        -0.87165 6.03840 0.79065
        -1.44398 7.42465 0.57603
        -2.85812 6.71416 3.58619
        -2.32775 6.71486 2.75296
        -2.30580 7.28287 4.14961
        -6.36944 -1.69297 1.83657
        -5.66646 -2.02605 1.22824
        -5.81033 -1.22101 2.49212
        -4.31935 -2.47549 0.20467
        -3.41324 -2.12272 0.02876
        -4.22835 -3.36101 -0.18716
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    # pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("pna10h2o", freq, pol, coup, atoms, basis, 12)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/pna/pna_30h2o_0.1.xyz", "w") do io
        do_md(io, 1, 40.0, atoms, e_grad_func, r)
    end
end

function test_pnpa_6etac()
    atoms = split_atoms("O"^4 * "N" * "C"^8 * "H"^7 * ("COHHCCCO" * "H"^6)^6)
    basis = "cc-pvdz"
    r = Float64[
        0.86082809882290 1.04575217626220 -1.60868410762516
        -5.08554550568333 0.32275076670539 0.28403645501939
        -4.53206344983781 -1.73623993095328 -0.08538322411966
        1.26464649489417 2.69589401342465 -0.10465757692456
        -4.28252762631185 -0.53987330929154 -0.05115342918972
        -0.39998886677044 0.71688984421731 -1.16372929670534
        -2.93349172276481 -0.10662635240513 -0.43461215928336
        -1.34687735162295 1.67429686796608 -0.76698785128584
        -0.71753329301637 -0.64930810347711 -1.19608706991968
        -2.62354005905289 1.25563603277410 -0.40029789051631
        -1.99423633487103 -1.06458354815992 -0.82921309125752
        1.62902214629902 2.03157685084951 -1.03986680118748
        2.96786752212912 2.10893407828314 -1.71034669041999
        -1.08614254068652 2.73042678292576 -0.73942166586152
        0.05021451070353 -1.36669385639999 -1.49879141827232
        -3.38520823119007 1.97089198271617 -0.09099536134681
        -2.27213645080775 -2.11824936407978 -0.84220525256778
        3.65520377207132 1.48697708439175 -1.11415586050990
        3.32757031382318 3.14625203070846 -1.68044447806608
        2.95638926219640 1.72220946436553 -2.73554957638635
        6.89961728501370 2.15920628296982 -0.62048005367088
        7.87330182574073 3.22132922673434 -0.55026882904607
        5.99190693817251 2.48385020068348 -0.08448910234445
        6.62657562013351 1.95223989254170 -1.66622322498055
        7.49881189988468 0.93540029251884 0.04326605010274
        7.77502903747229 4.36610084119759 -1.25281086060535
        6.66767320177670 4.49083288752580 -2.27299185229618
        8.57354201259834 5.24818368228889 -1.02476263761944
        6.76059633141287 5.46640172227074 -2.76464697787869
        6.70063807286220 3.68218241796256 -3.01872097333905
        5.68443491107515 4.42385692555589 -1.77924632105346
        6.74749920806454 0.13167336769760 0.07519460657799
        7.80655739599581 1.16524400913369 1.07430415525614
        8.37835492035018 0.57798549529075 -0.51577015334786
        2.69818455323182 -2.80835719487018 1.54934422714561
        2.03980392052353 -2.64306457766921 0.28291087598015
        3.16151972949724 -3.81071342595212 1.58934138988429
        3.48922293126109 -2.05270596623938 1.66519997352795
        1.65018479022663 -2.65813038871457 2.63445875066866
        2.71146591248991 -2.51244239019024 -0.87525302660940
        4.22156547458306 -2.53326038815564 -0.83018672515022
        2.07647202081654 -2.36549769026242 -1.89565735047172
        4.60347140166050 -2.51097827531565 -1.85750141700457
        4.57163401209789 -1.64341504167323 -0.28235638406822
        4.60413738545654 -3.42546966831141 -0.31078345200400
        2.11258111846731 -2.79150830609875 3.62537644088295
        0.85021425965482 -3.40559954477391 2.51877634578023
        1.19462210199358 -1.65673154790440 2.59332610866354
        10.13024294032854 3.98957233861952 1.95243066673141
        8.93491129057221 3.39203938436365 2.49234867701369
        9.90999301364601 4.40481322242803 0.95666308966281
        10.46146582487557 4.80857595935690 2.61462682090793
        11.18739242683210 2.90739154210409 1.86883540258231
        7.78143550767844 4.06868685159692 2.59087746722087
        7.79869789674058 5.55535643824593 2.31789120030545
        6.77835651090755 3.46619075849506 2.91219907639011
        6.79414030075112 5.95630063820994 2.49904845472759
        8.51920153214830 6.06498493839458 2.97761753907181
        8.10178485941280 5.75886164557450 1.27871518380010
        12.12511914019630 3.33094002918217 1.47587370579183
        10.86115522908174 2.10231822657676 1.19309467267681
        11.38950346061806 2.47139795131960 2.85972495365067
        3.50074524893035 -0.48103777644374 -5.55138011499430
        4.12808613648904 0.42706659685067 -4.62656528422888
        3.44670346496990 -0.00876976733308 -6.54786160098659
        4.11410736965753 -1.39405820071672 -5.64220429772866
        2.11977537546606 -0.81296421916344 -5.02501093834311
        5.37593265022553 0.90183854788499 -4.79379089785147
        6.14198841173611 0.47522315159163 -6.02886240458723
        5.83272220610060 1.64693704172661 -3.95869331085182
        7.12931175229825 0.95145616772468 -6.00794648433540
        6.26415066307118 -0.61891814119998 -6.06249093916708
        5.61101435661400 0.77268615272909 -6.94702661954821
        1.61621904025010 -1.50133906595604 -5.72252404502039
        1.50698882339142 0.09740060555697 -4.93686029873449
        2.17603571871586 -1.29454191856233 -4.03743648660075
        5.40961781634195 0.61755952344718 4.17420302356020
        4.94785677188737 0.09227505375110 2.91225097832947
        5.96705996046869 1.54908479218834 3.99330411605927
        4.54144136488352 0.84051249223467 4.81837396903471
        6.29458478517886 -0.43391505695516 4.81336092499201
        4.20534990249755 0.81708347950887 2.05771126604673
        3.70958880019331 2.17128398252604 2.49941052120525
        3.96457875028069 0.33641492103209 0.96750253305383
        3.02396696298363 2.56648721486289 1.73956499421946
        3.18181950955043 2.09969597100342 3.46349793188907
        4.56250544320318 2.85481408206009 2.64192875021884
        6.67083671570971 -0.06621977005748 5.78121721181099
        7.15943522014916 -0.65806033940873 4.17020875187141
        5.74047700330451 -1.36951169887336 4.98905577564215
        -8.07064145761638 -2.20348433442715 0.51268493294925
        -8.66707889861672 -2.08442621769550 -0.78492460734709
        -7.04945845440994 -1.78994830431227 0.50158930258093
        -8.66405819763748 -1.63999225432790 1.25498560565828
        -8.03946350854881 -3.67777167761602 0.86797256485888
        -8.86022254217190 -0.89173964874022 -1.39724513381760
        -8.49328192104784 0.36064709280131 -0.62578768701967
        -9.31696980479912 -0.87646306158215 -2.51076471980012
        -8.71150573627209 1.22852355257564 -1.25961191904275
        -9.07922918139455 0.43494081589803 0.30471085435526
        -7.42918638312883 0.36388167532341 -0.34251290509978
        -7.58857602994329 -3.81816625079183 1.86326005383025
        -7.44012614149513 -4.23782722451654 0.13384873104925
        -9.05592317047338 -4.10175311169502 0.87958413447924
    ]' * Å2B

    freq = 0.5
    pol = [0, 1, 0]
    # pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func("pnpa6etac", freq, pol, coup, atoms, basis, 48)

    e_grad_func = make_e_and_grad_func(rf)
    open("md/pnpa/pnpa_6etac_0.1.xyz", "w") do io
        do_md(io, 1, 40.0, atoms, e_grad_func, r)
    end
end
