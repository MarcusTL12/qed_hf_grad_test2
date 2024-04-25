include("geomeTRIC/main.jl")

atoms = split_atoms("CCCCCCNHHNHHHHOO")
basis = "cc-pVDZ"
r = Float64[
     2.62228        2.16217        0.00000
     1.91618        3.37206        0.00000
     0.51818        3.37385        0.00000
    -0.20116        2.16218        0.00000
     0.51818        0.95051        0.00000
     1.91618        0.95229        0.00000
     4.03891        2.16217        0.00000
     2.44876        4.31496        0.00000
     0.00000        4.32435        0.00000
    -1.63056        2.16218        0.00000
     0.00000        0.00000        0.00000
     2.44875        0.00939        0.00000
     4.56523        1.27614        0.00000
     4.56523        3.04820        0.00000
    -2.23638        3.19152        0.00000
    -2.23638        1.13285        0.00000
]

freq = 0.5
pol = [0.0, 1.0, 0.0]
pol = pol / norm(pol)
coup = 0.05

rf = make_runner_func_qed_ccsd("pna_y", freq, pol, coup, atoms, basis, 64)

egf = make_e_and_grad_func(rf)

qed_hf_engine = engine.qed_hf_engine(egf, atoms, r)

m = engine.run_opt(qed_hf_engine)

write_xyz_hist("pna_ccpvdz_ypol.xyz", atoms, m.xyzs)
