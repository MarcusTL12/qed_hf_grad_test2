using PyCall
using OhMyREPL
using LinearAlgebra

pushfirst!(pyimport("sys")."path", "./scanner")

scanner = pyimport("pyscf_scanner")
pyscf = pyimport("pyscf")

geomopt = pyimport("pyscf.geomopt.geometric_solver")

include("../common.jl")

function test_2h2o_qed_ccsd()
    # mol = pyscf.M(atom="""
    #     O    0.224814   0.265419  -0.0118646
    #     H    0.118539   0.0837589  0.914633
    #     H   -0.605434   0.0282651 -0.397106
    #     O   -0.0365481 -0.370659   2.89984
    #     H    0.796774  -0.398863   3.34876
    #     H   -0.65382   -0.0447871  3.53878
    # """, basis="sto-3G")

    mol = pyscf.M(atom="""
        O   0.779636   0.783321   0.547209
        H   0.262225   0.194884   1.206843
        H   0.011905   1.425478   0.316734
        O  -0.612266  -0.818115   2.363548
        H   0.003579  -1.614294   2.555880
        H  -0.600685  -0.408071   3.302897
    """, basis="sto-3G")

    freq = 0.5
    pol = [0.577350, 0.577350, 0.577350]
    pol = pol / norm(pol)
    coup = 0.05

    rf = make_runner_func_qed_ccsd_mol("grad", freq, pol, coup, 48)

    egf = make_e_and_grad_func(rf)

    qed_ccsd_scanner = scanner.QED_CCSD_GradScanner(egf, mol)

    # conv_params = Dict([
    #     "convergence_energy" => 1e-8,  # Eh
    #     "convergence_grms" => 1e-9,    # Eh/Bohr
    #     "convergence_gmax" => 1e-9,    # Eh/Bohr
    #     "convergence_drms" => 1e-6,    # Angstrom
    #     "convergence_dmax" => 1e-6,    # Angstrom
    # ])

    # m = engine.run_opt(qed_hf_engine, conv_params)

    # write_xyz_hist("2H2O_qed_ccsd.xyz", atoms, m.xyzs)

    # m

    geomopt.optimize(qed_ccsd_scanner, convergence_gmax=1e-8)
end
