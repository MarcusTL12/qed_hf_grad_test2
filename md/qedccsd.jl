include("main.jl")

function test_h2o_qed_ccsd()
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

    rf = make_runner_func_qed_ccsd("qedccsd_grad",
        freq, pol, coup, atoms, basis, 48, eT="eT_dev")

    e_grad_func = make_e_and_grad_func(rf)

    open("md/qed_ccsd/h2o/$(coup)_$basis.xyz", "w") do io
        do_md(io, 100, 25.0, atoms, e_grad_func, r)
    end
end

function test_4h2_qed_ccsd()
    atoms = split_atoms("HHHHHHHH")
    basis = "aug-cc-pvdz"
    r = Float64[
        -0.22807 0.92666 0.82655
        -0.26170 0.97975 1.53168
        -3.10688 1.51385 -0.60675
        -2.43602 1.62998 -0.80091
        -0.82260 -1.01634 -1.10789
        -0.46925 -1.44152 -0.66556
        -2.59650 -0.51059 1.06391
        -2.87484 -1.07894 0.74657
    ]' * Å2B

    freq = 0.46
    pol = [0.0, 1.0, 0.0]
    pol = pol / norm(pol)
    coup = 0.1

    rf = make_runner_func_qed_ccsd("qedccsd_grad",
        freq, pol, coup, atoms, basis, 40)

    e_grad_func = make_e_and_grad_func(rf)

    open("md/qed_ccsd/h2/4H2_$(freq)_$(coup)_$basis.xyz", "w") do io
        do_md(io, 2, 25.0, atoms, e_grad_func, r)
    end
end

function test_8h2_qed_ccsd()
    atoms = split_atoms("HHHHHHHH"^2)
    basis = "aug-cc-pvdz"
    r = Float64[
        -4.50797 -1.59156 3.77235
        -3.86678 -1.35967 3.58246
        -1.46813 -1.81789 2.37891
        -1.35764 -2.03674 1.71491
        -1.47191 0.79577 1.50809
        -2.11794 1.00862 1.70402
        -4.72873 1.12436 2.64290
        -4.40291 1.32499 3.23853
        -1.86723 -2.53180 5.07516
        -2.06903 -3.12741 4.75005
        -1.45197 0.11524 4.29998
        -2.06256 0.42296 4.48329
        -4.42740 -1.04249 0.91515
        -3.73247 -0.98601 0.79285
        -3.71602 -3.61449 1.86213
        -3.41449 -3.89450 2.43817
    ]' * Å2B

    freq = 0.46
    pol = [0.0, 1.0, 0.0]
    pol = pol / norm(pol)
    coup = 0.0

    rf = make_runner_func_qed_ccsd("qedccsd_grad",
        freq, pol, coup, atoms, basis, 24; eT="eT_dev")

    e_grad_func = make_e_and_grad_func(rf)

    open("md/qed_ccsd/h2/8H2_$(freq)_$(coup)_$basis.xyz", "w") do io
        do_md(io, 2, 25.0, atoms, e_grad_func, r)
    end
end
