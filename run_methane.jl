include("geomeTRIC/main.jl")

atoms = split_atoms("CHHHH")
basis = "cc-pVDZ"

r = Float64[
    -2.58192 -0.02379 -0.00000
    -1.47251 -0.02378 -0.00000
    -2.95171 -0.86697 0.61892
    -2.95172 -0.13819 -1.03968
    -2.95172 0.93380 0.42076
]

freq = 0.5
pol = [0.577350, 0.577350, 0.577350]
pol = pol / norm(pol)
coup = 0.0

rf = make_runner_func_qed_ccsd("methane", freq, pol, coup, atoms, basis, 64)

egf = make_e_and_grad_func(rf)

qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

conv_params = Dict([
    "convergence_energy" => 1e-8,  # Eh
    "convergence_grms" => 1e-9,    # Eh/Bohr
    "convergence_gmax" => 1e-9,    # Eh/Bohr
    "convergence_drms" => 1e-4,    # Angstrom
    "convergence_dmax" => 1e-4,    # Angstrom
])

m = engine.run_opt(qed_hf_engine, conv_params)

write_xyz_hist("2H2O_qed_ccsd.xyz", atoms, m.xyzs)
