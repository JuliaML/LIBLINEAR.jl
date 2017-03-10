if !is_windows()
  cd(joinpath(Pkg.dir(), "LIBLINEAR", "deps", "liblinear-210"))
  run(`make lib`)
  run(`cp liblinear.so.3 ../liblinear.so.3`)
end
