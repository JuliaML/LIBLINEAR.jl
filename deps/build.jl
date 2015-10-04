cd(joinpath(Pkg.dir(), "LIBLINEAR", "deps", "liblinear-210"))
run(`make lib`)
run(`cp liblinear.so.2 ../liblinear.so.2`)
