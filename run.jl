include("geomeTRIC/main.jl")

atoms = split_atoms("OHHOHH")
basis = "cc-pVDZ"
# r = Float64[
#     0.224814 0.265419 -0.0118646
#     0.118539 0.0837589 0.914633
#     -0.605434 0.0282651 -0.397106
#     -0.0365481 -0.370659 2.89984
#     0.796774 -0.398863 3.34876
#     -0.65382 -0.0447871 3.53878
# ]

r = Float64[
    0.8028110964045356 0.8102646161362445 0.581753919281548
    0.27632285090288117 0.20529149777929528 1.2188326615778597
    0.031398706771020024 1.4422820933792153 0.33604309313325137
    -0.6173628167525232 -0.8365955450740084 2.334920179825137
    0.008129495496525713 -1.6201068453140106 2.5470499751017774
    -0.6546233021450376 -0.43410665186288894 3.276883433174618
]

freq = 0.5
pol = [0.577350, 0.577350, 0.577350]
pol = pol / norm(pol)
coup = 0.05

rf = make_runner_func_qed_ccsd("2H2O", freq, pol, coup, atoms, basis, 64)

egf = make_e_and_grad_func(rf)

qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

conv_params = Dict([
    "convergence_energy" => 1e-8,  # Eh
    "convergence_grms" => 1e-9,    # Eh/Bohr
    "convergence_gmax" => 1e-9,  # Eh/Bohr
    "convergence_drms" => 1e-6,    # Angstrom
    "convergence_dmax" => 1e-6,    # Angstrom
])

m = engine.run_opt(qed_hf_engine, conv_params)

write_xyz_hist("2H2O_qed_ccsd.xyz", atoms, m.xyzs)
