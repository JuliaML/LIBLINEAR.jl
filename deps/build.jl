if !Sys.iswindows()
  # Build standard library
  cd(joinpath(@__DIR__, "liblinear-221"))
  run(`make clean`)
  run(`make lib`)
  run(`cp liblinear.so.3 ../liblinear.so.3`)

  # Build weights mod
  cd(joinpath(@__DIR__, "liblinear-weights-221"))
  run(`make clean`)
  run(`make lib`)
  rm("../liblinear-weights.so.3", force=true, recursive=false)
  run(`cp liblinear.so.3 ../liblinear-weights.so.3`)
end
