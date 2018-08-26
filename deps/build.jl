if !Sys.iswindows()
  cd(joinpath(dirname(@__FILE__), "liblinear-210"))
  run(`make lib`)
  run(`cp liblinear.so.3 ../liblinear.so.3`)
end
