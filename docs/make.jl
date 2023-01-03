push!(LOAD_PATH,"../docs/src/")
push!(LOAD_PATH,"../src")

using Documenter, scBayesDeconv

makedocs(sitename="scBayesDeconv.jl",
pages = [
    "Usage" => "index.md",
    "Examples" => [],
    "API.md"
],
format = Documenter.HTML(prettyurls = false)
)

deploydocs(
    repo = "github.com/gatocor/scBayesDeconv.jl",
)