using Documenter, TriangularReshapes

push!(LOAD_PATH, "../src/")
makedocs(
 sitename="TriangularReshapes",
 pages = [
          "Home" => "index.md",
         ]
)
deploydocs(
  repo = "github.com/Algopaul/TriangularReshapes.git",
  versions = nothing
)
