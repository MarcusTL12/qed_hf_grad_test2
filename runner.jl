# Script meant for editing and running as job on cluster

# include("numgrad/main.jl")
include("md/main.jl")

# curfile = "md/many_h2o/30h2o_free.xyz"
# curfile = "md/many_h2o/30h2o_0.1.xyz"
# curfile = "md/many_h2o/50h2o_free.xyz"
# curfile = "md/many_h2o/20h2o_free.xyz"

# a = @async keep_temp("md/many_h2o/10h2o_free.xyz", 275, 1000, 600; name="10h2o_free")
# b = @async keep_temp("md/many_h2o/10h2o_0.05.xyz", 275, 1000, 600; name="10h2o_0.05")
# c = @async keep_temp("md/many_h2o/10h2o_0.1.xyz", 275, 1000, 600; name="10h2o_0.1")

# wait(a)
# wait(b)
# wait(c)

keep_temp_qed_ccsd("md/qed_ccsd/h2/4H2_0.46_0.1_aug-cc-pvdz.xyz", 70, 100, 50; name="4H2_0.1")
