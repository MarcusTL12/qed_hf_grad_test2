using OhMyREPL
using Plots
using Rotations
using LinearAlgebra
using Plots.PlotMeasures
# plotly()

include("../common.jl")

# plot_size = (400, 300)
plot_size = (1200, 1000)

const au2eV = 27.211396641308

function rotate_geo(r, θ, ϕ)
    Rθ = AngleAxis(deg2rad(θ), 1.0, 0.0, 0.0)
    Rϕ = AngleAxis(deg2rad(ϕ), 0.0, 0.0, 1.0)

    R = Rθ * Rϕ

    R * r
end

function find_ind(rng, val)
    lo = 1
    hi = length(rng)

    while hi - lo > 1
        mid = (hi + lo) ÷ 2
        if rng[mid] < val
            lo = mid
        else
            hi = mid
        end
    end

    lo, hi
end

function lin_interp(xs, ys, zs, x, y)
    x1, x2 = find_ind(xs, x)
    y1, y2 = find_ind(ys, y)

    z11 = zs[x1, y1]
    z12 = zs[x1, y2]
    z21 = zs[x2, y1]
    z22 = zs[x2, y2]

    x2f = (x - xs[x1]) / (xs[x2] - xs[x1])
    x1f = 1 - x2f

    y2f = (y - ys[y1]) / (ys[y2] - ys[y1])
    y1f = 1 - y2f

    z11 * x1f * y1f + z12 * x1f * y2f + z21 * x2f * y1f + z22 * x2f * y2f
end

function plot_nice_grid(xs, ys, zs, x_len, y_len)
    xs2 = range(extrema(xs)...; length=x_len)
    ys2 = range(extrema(ys)...; length=y_len)

    zs2 = [lin_interp(xs, ys, zs, x, y) for x in xs2, y in ys2]

    surface(xs2, ys2, zs2; xlabel="θ", ylabel="ϕ", size=(900, 700))
end

function make_surface(atoms, basis, r, coup, θs, ϕs, ntask, nomp)
    freq = 0.5
    pol = [0, 1, 0]

    rfs = [
        make_runner_func("rot$i", freq, pol, coup, atoms, basis, nomp; eT="eT_clean", restart=false)
        for i in 1:ntask
    ]

    efs = make_tot_energy_function.(rfs)

    all_angles = [(θ, ϕ) for θ in θs, ϕ in ϕs]

    es = zeros(length(θs), length(ϕs))

    amt_done = Threads.Atomic{Int}(0)

    progressbar = @async begin
        a = 0
        last_a = 0
        b = length(es)
        while a < b
            a = amt_done[]
            if a != last_a
                p = round(a / b * 100; digits=2)
                println("$a / $b = $p%")
                last_a = a
            end
            sleep(1)
        end
    end

    tasks = [
        begin
            @async for i in th:ntask:length(es)
                θ, ϕ = all_angles[i]
                es[i] = efs[th](rotate_geo(r, θ, ϕ))
                Threads.atomic_add!(amt_done, 1)
            end
        end for th in 1:ntask
    ]

    for t in tasks
        wait(t)
    end

    wait(progressbar)

    e0 = lin_interp(θs, ϕs, es, 0.0, 0.0)
    es .-= e0
    es .*= au2eV * 1000

    surface(ϕs, θs, es; xlabel="ϕ", ylabel="θ", zlabel="meV", size=plot_size)
end

function make_surface_1d(atoms, basis, r, coup, θs, ntask, nomp, xorz=:x)
    freq = 0.5
    pol = [0, 1, 0]

    rfs = [
        make_runner_func("rot$i", freq, pol, coup, atoms, basis, nomp; eT="eT_clean", restart=false)
        for i in 1:ntask
    ]

    efs = make_tot_energy_function.(rfs)

    es = zeros(length(θs))

    amt_done = Threads.Atomic{Int}(0)

    progressbar = @async begin
        a = 0
        last_a = 0
        b = length(es)
        while a < b
            a = amt_done[]
            if a != last_a
                p = round(a / b * 100; digits=2)
                println("$a / $b = $p%")
                last_a = a
            end
            sleep(1)
        end
    end

    tasks = [
        begin
            @async for i in th:ntask:length(es)
                if xorz == :x
                    θ = θs[i]
                    ϕ = 0
                elseif xorz == :z
                    θ = 0
                    ϕ = θs[i]
                end
                es[i] = efs[th](rotate_geo(r, θ, ϕ))
                Threads.atomic_add!(amt_done, 1)
            end
        end for th in 1:ntask
    ]

    for t in tasks
        wait(t)
    end

    wait(progressbar)

    e0 = minimum(es)
    es .-= e0
    es .*= au2eV * 1000

    plot(θs, es; xlabel="θ", ylabel="meV", leg=false, size=plot_size)
end

function make_surface_1d_both(atoms, basis, r, coup, θs, ϕs, ntask, nomp)
    freq = 0.5
    pol = [0, 1, 0]

    rfs = [
        make_runner_func("rot$i", freq, pol, coup, atoms, basis, nomp; eT="eT_qed_hf_grad_print", restart=false)
        for i in 1:ntask
    ]

    efs = make_tot_energy_function.(rfs)

    esθ = zeros(length(θs))
    esϕ = zeros(length(ϕs))

    amt_done = Threads.Atomic{Int}(0)

    progressbar = @async begin
        a = 0
        last_a = 0
        b = length(esθ) + length(esϕ)
        while a < b
            a = amt_done[]
            if a != last_a
                p = round(a / b * 100; digits=2)
                println("$a / $b = $p%")
                last_a = a
            end
            sleep(1)
        end
    end

    tasks = [
        begin
            @async for i in th:ntask:length(esθ)
                θ = θs[i]
                esθ[i] = efs[th](rotate_geo(r, θ, 0))
                Threads.atomic_add!(amt_done, 1)
            end
        end for th in 1:ntask
    ]

    for t in tasks
        wait(t)
    end

    tasks = [
        begin
            @async for i in th:ntask:length(esϕ)
                ϕ = ϕs[i]
                esϕ[i] = efs[th](rotate_geo(r, 0, ϕ))
                Threads.atomic_add!(amt_done, 1)
            end
        end for th in 1:ntask
    ]

    for t in tasks
        wait(t)
    end

    wait(progressbar)

    e0 = min(minimum(esθ), minimum(esϕ))
    esθ .-= e0
    esθ .*= au2eV * 1000
    esϕ .-= e0
    esϕ .*= au2eV * 1000

    plot(θs, esθ; xlabel="θ", ylabel="meV", label="x", size=plot_size, left_margin=30px)
    plot!(ϕs, esϕ; label="z")
end

function test_h2_1d()
    atoms = split_atoms("HH")
    basis = "cc-pvdz"
    r = [
        0 0 0
        1 0 0
    ]'

    coup = 0.05

    θs = range(-90, 89; length=100)

    make_surface_1d(atoms, basis, r, coup, θs, 40, 2, :z)
end

function test_h2()
    atoms = split_atoms("HH")
    basis = "cc-pvdz"
    r = [
        0 0 0
        1 0 0
    ]'

    coup = 0.05

    θs = range(-90, 89; length=30)
    ϕs = range(0, 179; length=30)

    make_surface(atoms, basis, r, coup, θs, ϕs, 40, 2)
end

function test_h2o()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = [
         0.0     0.0   0.0
        -0.749   0.0   0.578
         0.749  -0.0   0.578
    ]' * Å2B

    coup = 0.05

    θs = range(-90, 90; length=200)
    ϕs = range(-90 + 45 - 3, 90 + 45 - 3; length=200)

    make_surface(atoms, basis, r, coup, θs, ϕs, 80, 1)
end

function test_h2o_1d()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = [
        0.0     0.0   0.0
        0.578   0.0   0.749
        0.578  -0.0  -0.749
    ]' * Å2B

    coup = 0.05

    θs = range(-90 + 45 - 3, 90 + 45 - 3; length=1000)
    ϕs = range(-90 + 45 - 3, 90 + 45 - 3; length=1000)

    make_surface_1d_both(atoms, basis, r, coup, θs, ϕs, 88, 1)
end

function test_h2o_test()
    atoms = split_atoms("OHH")
    basis = "cc-pvdz"
    r = [
        0 0 0
        0.7 0 0.7
        0.7 0 -0.7
    ]' * Å2B

    coup = 0.05

    θs = range(-90, 89; length=100)
    ϕs = range(-90, 89; length=100)

    make_surface_1d_both(atoms, basis, r, coup, θs, ϕs, 80, 1)
end

function test_hof()
    atoms = split_atoms("OHF")
    basis = "cc-pvdz"
    r = [
        0.338169 -0.0474363 0.00728898
        1.0885 0.527633 0.00346908
        -0.41258 0.5271 0.00916178
    ]'

    coup = 0.05

    θs = range(-90, 89; length=30)
    ϕs = range(-90, 89; length=30)

    make_surface(atoms, basis, r, coup, θs, ϕs)
end

function test_nh3()
    atoms = split_atoms("NHHH")
    basis = "cc-pvdz"
    r = [
        0.251119 -0.0390666 0.251119
        1.14819 0.348937 0.00888569
        -0.408197 0.341194 -0.408197
        0.00888569 0.348937 1.14819
    ]'

    coup = 0.05

    θs = range(-90, 89; length=10)
    ϕs = range(-90, 89; length=10)

    make_surface(atoms, basis, r, coup, θs, ϕs)
end

function test_thalidomide()
    atoms = split_atoms("OOOOCCCCCCCCCCCCCNNHHHHHHHHHH")
    basis = "sto-3g"
    r = [
        1.2254092860971477 0.006861938213065494 -2.1714945074109813
        -0.7899371958433331 2.355375459088116 0.23748124484084765
        -5.084559799007496 1.1514146742332676 -0.08370259931824055
        -0.12249816767676583 -0.24933797007781613 2.143476708721496
        1.2949444402903305 -0.11491635808210324 -0.9941625532678013
        -1.1398381367827963 0.12668546715958545 -0.5482756969053151
        -2.1539542023378955 -0.958091086502314 -0.19141184986290552
        -3.532360921219951 -0.5493403494721103 -0.6992812042826344
        -3.951720714949701 0.8203170920766167 -0.22223800669476365
        -1.5659603167698986 1.501305903681236 -0.04217966639886689
        2.50494727525074 -0.26430282382103465 -0.1374393429220982
        3.8390329241100263 -0.32068795182765875 -0.48606267941647363
        4.758810312071359 -0.4583009432210533 0.5493171903148679
        4.344100400493373 -0.5342825362200728 1.877996849571215
        2.9950849583820554 -0.47542942281463174 2.2151258974847807
        2.0934571568924585 -0.33979851262048355 1.1792620519435508
        0.6065785750867101 -0.2447013107194743 1.2079412081584093
        0.20849148708605308 -0.16004158049244133 -0.12343484685280683
        -2.9242607986963054 1.7140161927058772 0.02868675690344421
        -1.0874690627630903 0.2183904743229904 -1.634080014762801
        -1.8480595912096727 -1.8995619899499905 -0.6469775659464011
        -2.174144913321342 -1.103568814460732 0.8879315610739102
        -3.5437475789693074 -0.5155281181983096 -1.7933166336582416
        -4.308033689367239 -1.2476435096982719 -0.3938841098471935
        -3.204900162440997 2.626126189086063 0.3381774368328167
        4.150601934369582 -0.2576414224298002 -1.5184620558856514
        5.815228456831038 -0.5051170588178925 0.32250426017924105
        5.08598750841268 -0.6386233349304403 2.658013020625192
        2.6630244196401613 -0.5300625043011359 3.241568401184959
    ]'

    coup = 0.05

    θs = range(-90, 89; length=10)
    ϕs = range(-180, 189; length=20)

    make_surface(atoms, basis, r, coup, θs, ϕs)
end

function test_2h2o()
    atoms = split_atoms("OHH"^2)
    basis = "cc-pvdz"
    r = [
        0.146772 0.431956 0.0767453
        0.0376185 0.139263 0.974309
        -0.434949 -0.121216 -0.423438
        -0.148746 -0.43922 2.917
        0.679645 -0.719107 3.28144
        -0.433083 0.274483 3.47122
    ]'

    coup = 0.05
    rf = make_runner_func("grad", freq, pol, coup, atoms, basis, 44; eT="eT_clean")

    θs = range(-90, 89; length=100)
    ϕs = range(-90, 89; length=100)

    make_surface(atoms, basis, r, coup, θs, ϕs)
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
    Es ./= au2eV
    Es .*= 1e6

    plot(θs, Es)

    Threads.@threads for i in eachindex(θs)
        th_id = Threads.threadid()
        θ = θs[i]
        Es[i] = efs[th_id](rotate_geo(r, 0, θ))
    end

    Es .-= minimum(Es)
    Es .*= 27.211396132
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
