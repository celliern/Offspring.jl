module SimpleModels
export run!, SimpleModel, Specie, SiteClimate, Mutation

using Random, Distributions, Statistics, StatsBase
include("prealloc_arrays.jl")
using .PreallocVectors

struct Mutation
    probability::Float64
    variance::Float64
end

struct Specie
    fecundity::Int64
    gamma::Float64
    plasticity::Int64
    selection::Int64
    mutation::Mutation
    migration::Float64
end

struct SiteClimate
    carrying::Int64
    inivalue::Float64
    variance::Float64
end

struct PopulationStats
    t::Int64
    theta::Float64
    var::Float64
    mean::Float64
end

function get_stats(t, θ, z::AbstractVector{Float64})
    PopulationStats(t, θ, var(z), mean(z))
end

mature_cut(x::Float64, θ::Float64, specie::Specie) = (1 - specie.gamma) * exp(-(x - θ)^2 / (2 * specie.plasticity))
is_mature(x::Float64, θ::Float64, specie::Specie) = mature_cut(x, θ, specie) >= rand()
is_mature(z::AbstractVector{Float64}, θ::Float64, specie::Specie) = mature_cut.(z, θ, Ref(specie)) .>= rand(length(z))

select_cut(x::Float64, θ::Float64, specie::Specie) = exp(-(x - θ)^2 / (2 * specie.selection))
is_selected(x::Float64, θ::Float64, specie::Specie) = select_cut(x, θ, specie) >= rand()
is_selected(z::AbstractVector{Float64}, θ::Float64, specie::Specie) = select_cut.(z, θ, Ref(specie)) .>= rand(length(z))


function germination!(offspring::AbstractVector{Float64}, theta, specie::Specie)
    will_grow_idxs = findall(x -> is_mature(x, theta, specie), offspring)
    adult = offspring[will_grow_idxs]
    deleteat!(offspring, will_grow_idxs) # delete the grown offspring
    return offspring, adult
end

function selection(adult::AbstractVector{Float64}, theta, specie::Specie)
    return adult[is_selected(adult, theta, specie)]
end

function tile!(z, x, v)
    empty!(z)
    for (x, n) in zip(x, v)
        for i = 1:n
            push!(z, x)
        end
    end
    return z
end

function tile(x, v)
    z = Vector{Float64}()
    sizehint!(z, sum(v))
    for (x, n) in zip(x, v)
        for i = 1:n
            push!(z, x)
        end
    end
    return z
end

function create_newgen(adult::AbstractVector{Float64}, specie::Specie)
    n = length(adult)
    newborn_counts = rand(Poisson(specie.fecundity), n)
    newgen = tile!(newgen, adult, newborn_counts)
    return newgen
end

function create_newgen!(newgen::AbstractVector{Float64}, adult::AbstractVector{Float64}, specie::Specie)
    n = length(adult)
    newborn_counts = rand(Poisson(specie.fecundity), n)
    newgen = tile!(newgen, adult, newborn_counts)
    return newgen
end

function create_reg_newgen!(newgen::AbstractVector{Float64}, adult::Vector{Float64}, specie::Specie, carrying::Int64, n_offspring::Int64)
    empty!(newgen)
    newborn_counts = rand(Poisson(specie.fecundity), length(adult))
    site_capacity = carrying - n_offspring
    n_new_gen = min(site_capacity, sum(newborn_counts))
    resize!(newgen, n_new_gen)
    newgen[1:n_new_gen] .= sample(adult, FrequencyWeights(newborn_counts), n_new_gen)
    return newgen
end

function mutate!(newgen::AbstractVector{Float64}, specie::Specie)
    for i in length(newgen)
        if rand() < specie.mutation.probability
            newgen[i] = rand(Normal(newgen[i], specie.mutation.variance))
        end
    end
    return newgen
end

function regulate!(newgen::AbstractVector{Float64}, offspring::AbstractVector{Float64}, site_climate::SiteClimate)
    site_current_capacity = site_climate.carrying - length(offspring)
    if site_current_capacity < length(newgen)
        newgen_idxs = sample(1:length(newgen), site_current_capacity, replace=false, ordered=true)
        keepat!(newgen, newgen_idxs)
    end
    return newgen
end

struct SimpleModel
    specie::Specie
    site_climate::SiteClimate
    theta_dist::Distribution{Univariate,Continuous}
    offspring::Vector{Float64}
    newgen::PreallocVector{Float64}
end

function SimpleModel(specie::Specie, site_climate::SiteClimate)
    offspring = Vector{Float64}()
    sizehint!(offspring, site_climate.carrying)
    newgen = PreallocVector{Float64}()
    sizehint!(newgen, site_climate.carrying * 10)
    return SimpleModel(
        specie, site_climate, Normal(site_climate.inivalue, sqrt(site_climate.variance)), offspring, newgen
    )
end

convert(stats::Vector{PopulationStats},) = [PopulationStats(stats[i], model.site_climate) for i in 1:length(stats)]

function run!(model::SimpleModel, n_iter::Int; n_init_pop::Int=2000, init_pop_dist::Distribution{Univariate,Continuous}=Normal(50, 0.2))
    specie = model.specie
    site_climate = model.site_climate
    offspring = model.offspring
    theta_dist = model.theta_dist
    newgen = model.newgen

    empty!(offspring)
    append!(offspring, rand(init_pop_dist, n_init_pop))

    stats_records = Vector{PopulationStats}()
    sizehint!(stats_records, n_iter + 1)

    push!(stats_records, get_stats(0, site_climate.inivalue, offspring))

    for t in 1:n_iter
        theta = rand(theta_dist)
        offspring, adult = germination!(offspring, theta, specie)
        reproducing_adults = selection(adult, theta, specie)
        if !isempty(reproducing_adults)
            # create_reg_newgen!(newgen, reproducing_adults, specie, site_climate.carrying, length(offspring))
            create_newgen!(newgen, reproducing_adults, specie)
            mutate!(newgen, specie)
            regulate!(newgen, offspring, site_climate)
            append!(offspring, newgen)
        end
        push!(stats_records, get_stats(t, theta, offspring))
    end
    return stats_records
end
end