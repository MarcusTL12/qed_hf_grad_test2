include("get_matrix.jl")
include("get_dipole.jl")
include("get_gradient.jl")

const Å2B = 1.8897261245650618
const kB = 3.166811563e-6

const eT_inout_dir = "eT_files"

function make_inp_func_old(freq, pol, coup, atoms, basis; restart=true)
    restart_str = restart ? "\n    restart" : ""
    function make_inp(r)
        r /= Å2B
        r = reshape(r, 3, length(r) ÷ 3)
        io = IOBuffer()

        print(
            io,
            """
system
    charge: 0
end system

do
    ground state
end do

memory
    available: 1920
end memory

method
    qed-hf
end method

solver scf$(restart_str)
    gradient threshold: 1d-10
end solver scf

qed
    modes:        1
    frequency:    {$freq}
    polarization: {$(pol[1]), $(pol[2]), $(pol[3])}
    coupling:     {$coup}
end qed

hf mean value
   dipole
   molecular gradient
end hf mean value

geometry
basis: $basis
"""
        )

        for (i, a) in enumerate(atoms)
            println(io, "    ", a, "    ", r[1, i], ' ', r[2, i], ' ', r[3, i])
        end

        println(io, "end geometry")

        String(take!(io))
    end
end

function make_inp_func_qed_hf(freq, pol, coup, atoms, basis; restart=true)
    restart_str = restart ? "\n    restart" : ""
    function make_inp(r)
        r /= Å2B
        r = reshape(r, 3, length(r) ÷ 3)
        io = IOBuffer()

        print(
            io,
            """
- system
    charge: 0

- do
    ground state

- memory
    available: 1920

- method
    qed-hf

- solver scf$(restart_str)
    gradient threshold: 1d-10

- boson
    modes:        1
    frequency:    {$freq}
    polarization: {$(pol[1]), $(pol[2]), $(pol[3])}
    coupling:     {$coup}

- hf mean value
   dipole
   molecular gradient

- geometry
basis: $basis
"""
        )

        for (i, a) in enumerate(atoms)
            println(io, "    ", a, "    ", r[1, i], ' ', r[2, i], ' ', r[3, i])
        end

        String(take!(io))
    end
end

function write_inp(inp, name)
    open("$(eT_inout_dir)/$(name).inp", "w") do io
        print(io, inp)
    end
end

function run_inp(name, omp, eT)
    if isnothing(omp)
        omp = parse(Int, read("omp.txt", String))
    end
    run(`$(homedir())/$(eT)/build/eT_launch.py $(eT_inout_dir)/$(name).inp --omp $(omp) --scratch ./scratch/$(name) -ks -s`)
    nothing
end

function delete_scratch(name)
    if isdir("./scratch/$(name)")
        rm("./scratch/$(name)"; recursive=true)
    end
end

function make_runner_func_old(name, freq, pol, coup, atoms, basis, omp;
    eT="eT_qed_hf_grad_print", restart=true)
    delete_scratch(name)
    inp_func = make_inp_func_old(freq, pol, coup, atoms, basis, restart=restart)
    function runner_func(r)
        inp = inp_func(r)
        write_inp(inp, name)
        run_inp(name, omp, eT)
    end
end

function make_runner_func(name, freq, pol, coup, atoms, basis, omp;
    eT="eT", restart=true)
    delete_scratch(name)
    inp_func = make_inp_func_qed_hf(freq, pol, coup, atoms, basis, restart=restart)
    function runner_func(r)
        inp = inp_func(r)
        write_inp(inp, name)
        run_inp(name, omp, eT)
    end
end

const tot_energy_reg = r"Total energy:\ +(-?\d+\.\d+)"

function get_tot_energy(name)
    m = match(tot_energy_reg, read("$(eT_inout_dir)/$(name).out", String))
    parse(Float64, m.captures[1])
end

function make_tot_energy_function(runner_func)
    function energy_function(r)
        runner_func(r)
        get_tot_energy(runner_func.name)
    end
end

function make_grad_func(runner_func)
    function grad_function(r)
        runner_func(r)
        get_gradient("$(eT_inout_dir)/$(runner_func.name)")
    end
end

function make_e_and_grad_func(runner_func)
    function e_and_grad(r)
        runner_func(r)

        get_tot_energy(runner_func.name),
        get_gradient("$(eT_inout_dir)/$(runner_func.name)")
    end
end

const atom_reg = r"[A-Z][a-z]?"

function split_atoms(atoms)
    [m.match for m in eachmatch(atom_reg, atoms)]
end

function write_xyz(filename, atoms, r, mode="w")
    open(filename, mode) do io
        println(io, length(atoms), '\n')
        for (i, a) in enumerate(atoms)
            println(io, "$a    $(r[1, i]) $(r[2, i]) $(r[3, i])")
        end
    end
end

function read_xyz(filename)
    open(filename) do io
        lines = Base.Stateful(eachline(io))
        n_atm = parse(Int, popfirst!(lines))
        popfirst!(lines)

        atoms = String[]
        r = Float64[]

        for l in lines
            s = split(l)
            push!(atoms, s[1])
            append!(r, (parse(Float64, x) for x in @view s[2:4]))
        end

        atoms, reshape(r, 3, n_atm)
    end
end
