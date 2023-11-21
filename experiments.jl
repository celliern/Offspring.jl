# %%
using DrWatson
@quickactivate

# %%
using DataFrames
include("./src/simple_model.jl")
using .SimpleModels

# %%
function run_exp(params::Dict; save=false)
    @unpack n_iter = params
    specie_mutation = Mutation(params["specie_mutation_probability"], params["specie_mutation_variance"])
    specie = Specie(
        params["specie_fecundity"],
        params["specie_gamma"],
        params["specie_plasticity"],
        params["specie_selection"],
        specie_mutation,
        params["specie_migration"],
    )
    site_climate = SiteClimate(
        params["site_climate_carrying"],
        params["site_climate_inivalue"],
        params["site_climate_variance"],
    )
    model = SimpleModel(specie, site_climate)
    results = run!(model, params["n_iter"])
    df = DataFrame(results)
    df[!, "replicate"] = fill(params["replicate"], nrow(df))
    df[!, "specie_plasticity"] = fill(params["specie_plasticity"], nrow(df))
    df[!, "site_climate_variance"] = fill(params["site_climate_variance"], nrow(df))
    if save
        csv_filename = datadir("simple_model", savename(params, "csv"; accesses=["site_climate_variance", "specie_plasticity", "replicate"]))
        wsave(csv_filename, df)
    end

    return params, df
end

# %%

allparams = Dict(
    "n_iter" => 10_000,
    "site_climate_variance" => 1, # [0.5, 0.8, 1.1, 1.2, 1.5, 2, 2.5],
    "site_climate_carrying" => 80_000,
    "site_climate_inivalue" => 50,
    "specie_fecundity" => 20,
    "specie_gamma" => 0.8,
    "specie_plasticity" => [1000, 50, 1],
    "specie_selection" => 1,
    "specie_mutation_probability" => 0.01,
    "specie_mutation_variance" => 0.1,
    "specie_migration" => 0.0,
    "replicate" => 1,
)

dicts = dict_list(allparams);

results = Vector{Any}(undef, length(dicts))
Threads.@threads for i in 1:length(dicts)
    results[i] = run_exp(dicts[i], save=true)
end

# %%
using AlgebraOfGraphics, CairoMakie, DataFramesMeta

# %%
dfs = getindex.(results, 2);
df = vcat(dfs...);
sort!(df, [:t, :specie_plasticity]);

# %%
specs = (
    data(df)
    * mapping(
        :t, :var,
        color=:specie_plasticity => nonnumeric,
        marker=:specie_plasticity => nonnumeric
    )
    * visual(Lines)
    * visual(Scatter, markersize=2)
)
draw(specs)