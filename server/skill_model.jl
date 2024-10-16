using RxInfer
using ExponentialFamilyProjection
using BayesBase
using Distributions
using Statistics
using JSON3

# Define the custom distribution first
struct TransformedBetaDistribution{H,T} <: ContinuousUnivariateDistribution
    skill::H
    learningrate::T
    difficulty::T
end

"""
    BayesBase.logpdf(dist::TransformedBetaDistribution, x)

Calculate the log probability density function for the TransformedBetaDistribution.

This function computes the logpdf using a Beta distribution with parameters derived from
the skill, learning rate, and difficulty of the TransformedBetaDistribution.

Parameters:
- `dist::TransformedBetaDistribution`: The distribution object containing skill, learning rate, and difficulty.
- `x`: The point at which to evaluate the logpdf.

Returns:
- The log probability density at point `x`.

Reasoning for α and β:
- α (success parameter) = skill + learning_rate - difficulty + 1
  - Increases with higher skill and learning rate
  - Decreases with higher difficulty
  - "+1" ensures the parameter is always positive
- β (failure parameter) = difficulty - (skill + learning_rate) + 1
  - Increases with higher difficulty
  - Decreases with higher skill and learning rate
  - "+1" ensures the parameter is always positive

This formulation creates a distribution where:
- Higher skill and learning rate shift the distribution towards higher performance values
- Higher difficulty shifts the distribution towards lower performance values

Note:
The max function with 0.1 ensures that the parameters are always at least 0.1,
preventing invalid Beta distributions and improving numerical stability.
"""

BayesBase.logpdf(dist::TransformedBetaDistribution, x) = logpdf(Beta(
    max(dist.skill + dist.learningrate - dist.difficulty + 1, 0.1),
    max(dist.difficulty - (dist.skill + dist.learningrate) + 1, 0.1)
), x)
BayesBase.insupport(dist::TransformedBetaDistribution, x) = true

@node TransformedBetaDistribution Stochastic [out, skill, learningrate, difficulty]

# Define the model
@model function skill_progress(performance, prior_skill, prior_learning_rate, prior_difficulty)
    skill ~ prior_skill

    learning_rate ~ prior_learning_rate

    difficulty ~ prior_difficulty

    performance ~ TransformedBetaDistribution(skill, learning_rate, difficulty)
end

# Define constraints
@constraints function non_conjugate_model_constraints()
    q(skill) :: ProjectedTo(NormalMeanVariance)
    q(learning_rate) :: ProjectedTo(Beta)
    q(difficulty) :: ProjectedTo(NormalMeanVariance)
    q(skill, learning_rate, difficulty) = q(skill)q(learning_rate)q(difficulty)
end

# Define initialization
@initialization function model_initialization(prior_skill, prior_learning_rate, prior_difficulty)
    q(skill) = prior_skill
    q(learning_rate) = prior_learning_rate
    q(difficulty) = prior_difficulty
end

function analyze_progress(performance, prior_skill, prior_learning_rate, prior_difficulty)
    performance = Float64(performance) / 10
    performance = clamp(performance, 1e-6, 1 - 1e-6)

    result = infer(
        model = skill_progress(prior_skill=prior_skill, prior_learning_rate=prior_learning_rate, prior_difficulty=prior_difficulty),
        data = (performance = performance,),
        constraints = non_conjugate_model_constraints(),
        initialization = model_initialization(prior_skill, prior_learning_rate, prior_difficulty),
        options = (rulefallback = NodeFunctionRuleFallback(),),
        showprogress = true,
        iterations = 10
    )

    function compute_posterior(result)
        skill_posterior = result.posteriors[:skill][end]
        learning_rate_posterior = result.posteriors[:learning_rate][end]
        difficulty_posterior = result.posteriors[:difficulty][end]
        return skill_posterior, learning_rate_posterior, difficulty_posterior
    end

    posterior = compute_posterior(result)

    json_string = JSON3.write(Dict(
        "latest_performance" => 10*performance,
        "prior_stats" => Dict(
            "skill" => (mean(prior_skill), std(prior_skill)),
            "learning_rate" => params(prior_learning_rate),
            "difficulty" => (mean(prior_difficulty), std(prior_difficulty))
        ),
        "posterior_stats" => Dict(
            "skill" => (mean(posterior[1]), std(posterior[1])),
            "learning_rate" => params(posterior[2]),
            "difficulty" => (mean(posterior[3]), std(posterior[3]))
        )
    ))

    # Write the JSON string to a file
    open("prior_skill.json", "w") do io
        write(io, json_string)
    end

    return json_string
end

function test_inference()
    # Read and parse the JSON file
    data = JSON3.read(read("prior_skill.json", String), Dict)

    # Extract data and create prior distributions
    performance = data["latest_performance"]
    # Today posterior is the prior for the next session
    priors = Dict(
        :skill => NormalMeanVariance(data["posterior_stats"]["skill"]...),
        :learning_rate => Beta(data["posterior_stats"]["learning_rate"]...),
        :difficulty => NormalMeanVariance(data["posterior_stats"]["difficulty"]...)
    )

    # Run analysis and parse results
    result = analyze_progress(performance, priors[:skill], priors[:learning_rate], priors[:difficulty])
    result = JSON3.read(result, Dict)

    # Helper function to format output
    format_stat = (name, stats) -> "  $name: mean = $(round(stats[1], digits=2)), std = $(round(stats[2], digits=2))"
    format_lr = (stats) -> "  learning_rate: α = $(round(stats[1], digits=2)), β = $(round(stats[2], digits=2))"

    # Print results
    println("Test Results:")
    for stage in ["Prior", "Posterior"]
        println("$stage:")
        for (key, value) in result["$(lowercase(stage))_stats"]
            println(key == "learning_rate" ? format_lr(value) : format_stat(key, value))
        end
    end
end

## Uncomment to test inference
# test_inference()
