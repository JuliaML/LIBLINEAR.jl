if OS_NAME == :Windows
  nothing
else
  cd(joinpath(Pkg.dir(), "LIBLINEAR", "deps", "liblinear-210"))
  run(`make lib`)
  run(`cp liblinear.so.3 ../liblinear.so.3`)
end