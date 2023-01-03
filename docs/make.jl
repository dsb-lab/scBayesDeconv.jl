push!(LOAD_PATH,"../docs/src/")
push!(LOAD_PATH,"../src")

using Documenter, scBayesDeconv

makedocs(sitename="scBayesDeconv.jl",
pages = [
    "Home" => "index.md",
    "API.md"
],
format = Documenter.HTML(prettyurls = false)
)

deploydocs(
    repo = "github.com/gatocor/scBayesDeconv.jl",
)