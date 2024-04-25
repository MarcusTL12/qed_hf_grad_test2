#!/bin/bash
#SBATCH --job-name=QED-CCSD_geoopt
#SBATCH --time=1-00:00:00        # d-hh:mm:ss
#SBATCH --mem=240G
#SBATCH --nodes=1 --ntasks=64
# #SBATCH --exclude=node[01-15]
#SBATCH --exclude=fock[01-05]

# julia run_geoopt_pna.jl
# julia run.jl
julia run_methane.jl
