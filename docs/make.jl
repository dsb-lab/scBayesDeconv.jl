push!(LOAD_PATH,"../docs/src/")
push!(LOAD_PATH,"../src")

using Documenter, scBayesDeconv

makedocs(sitename="scBayesDeconv.jl",
pages = [
    "Home" => "index.md",
    "Examples" => ["examples/Artificial Convolutions.md",
                    "examples/Comparison with FFT deconvolution.md",
                    ],
    "API.md"
],
format = Documenter.HTML(prettyurls = false)
)

deploydocs(
    repo = "github.com/gatocor/scBayesDeconv.jl",
)