using Flux 
using MLJ: partition
import Chain: @chain

vars2matrix(x::NamedTuple, vars::Array{Symbol}) = mapreduce(v -> x[v], hcat, vars) |> transpose
vars2matrix(x::NamedTuple, vars::Array{Symbol}, n::Nothing) = vars2matrix(x, vars)
vars2matrix(x::NamedTuple, vars::Array{Symbol}, transfuns::NamedTuple ) = 
    mapreduce(v -> transfuns[v].(x[v]), hcat, vars) |> transpose

makelinfun(a::Number, b::Number) = x -> a * x + b
##


function plotfit(screen, losses, model, allTrain, allVali, trace_param)

    !isnothing(trace_param) && length(trace_param) > 1 && println("WARNING: only first parameter trace plotted!")

    fig=Figure(resolution=(1920,1080))
    scat1 = Axis(fig[1, 1:2], title="Training -- y: observed, x: $(typeof(model))")
    scat2 = Axis(fig[1, 3:4], title="Validation -- y: observed, x: $(typeof(model))")
    line = Axis(fig[2, 1:2], ylabel="log(Loss)")
    lineZoom = Axis(fig[2,3], ylabel=L"log(0.01 + loss - min(losses))")
    if !isnothing(trace_param) lineTrace = Axis(fig[2,4], ylabel=string(trace_param[1].first)) end

    lines!(line, losses.train .|> log10)
    lines!(line, losses.vali .|> log10)

    zoomlen=min(length(losses.train)-1, 200)
    zoomidx = length(losses.train)-zoomlen:length(losses.train)
    lines!(lineZoom, zoomidx, log10.(1e-2 .+ losses.train[zoomidx] .- minimum(losses.train[zoomidx])))
    lines!(lineZoom, zoomidx, log10.(1e-2 .+ losses.vali[zoomidx] .- minimum(losses.vali[zoomidx])))
    if !isnothing(trace_param) lines!(lineTrace, zoomidx, trace_param[1].second[zoomidx]) end
    
    len=length(allTrain.x[1])
    plotidx = range(1, len, step=1 + len ÷ 1000)
    #print(plotidx |> length)
    scatter!(scat1, vec(model(allTrain.x))[plotidx], vec(allTrain.y)[plotidx], color=(:black, 0.2))
    ablines!(scat1, 0, 1, color=:red, linewidth=5 )
    scatter!(scat2, model(allVali.x) |> vec, allVali.y |> vec, color=(:black, 0.2))
    ablines!(scat2, 0, 1, color=:red, linewidth=5 )

    display(screen, fig)
    sleep(0.2)
    #return fig

end

"""
fit_df!(data, predictors, target, lossfun; modeltype=nothing, model=nothing, lr=0.001, opt=Adam(lr), opt_state=Flux.setup(opt, model),
n_epoch=1000, patience=100, batchsize=30, latents2record=nothing, printevery=10, plotevery=50)

Fit a semi-parametric model to tabular data using Flux.jl.

Arguments:
- `data`: a named tuple of input data, where the keys are column names and the values are vectors of equal length.
- `predictors`: an array of column names to use as predictors in the model.
- `target`: the name of the column to use as the target variable.
- `lossfun`: the loss function to optimize, taking in the model and a data batch.

Optional keyword arguments:
- `modeltype`: a function that creates an untrained model, which is passed to the `model` argument if provided.
- `model`: a pre-trained model to use. If not provided, `modeltype` is used to create one.
- `lr`: the learning rate to use for the optimizer.
- `opt`: the optimizer to use.
- `opt_state`: the initial state of the optimizer.
- `n_epoch`: the number of epochs to train for.
- `patience`: the number of epochs to wait for improvement on the validation set before stopping training.
- `batchsize`: the size of each mini-batch during training.
- `latents2record`: an optional dictionary of model parameters to record during training.
- `printevery`: how often to print out loss values during training.
- `plotevery`: how often to plot the loss curves and other diagnostics during training.

Returns a named tuple containing the training and validation losses, the best model found during training, the final optimizer state, and any recorded model parameters.
"""

function fit_df!(data, predictors, target, lossfun; 
    modeltype=nothing, model=nothing, lr=0.001, opt=Adam(lr), opt_state=Flux.setup(opt, model), 
    n_epoch=1000, patience=100, batchsize=30, latents2record=nothing, printevery=10, plotevery=50) 

# This creating of the model on the fly did not work.
#    if isnothing(model)  
#     Flux.@functor(modeltype)
#     model=modeltype(predictors) 
#    end
   isnothing(model) && error("Does not work, because model is not trainable. Model needs to be explicitly defined it seems...")
    #opt_state = Flux.setup(opt, model)   
    #@show model

    # Partition data into training and validation sets
    d_train, d_vali = partition(data, 0.8, shuffle=true)

    # Helper function to create input matrix for model, d is the data, p the predictors to use
    # If no specific predcitors are indicated then just the data is returned (it has keys that are used in the model)
    # Predictors can be specified if the model "wants" a tensor directly (e.g. a normal chain)
    create_X(d::NamedTuple, p::AbstractArray) = vars2matrix(d, p)
    create_X(d::NamedTuple, p::Nothing) = d
    
    # Create data loaders for training and validation sets
    trainAll = Flux.DataLoader((; x= create_X(d_train, predictors), y=reshape(d_train[target], 1, :)), batchsize=length(d_train[1]), shuffle=true, partial=true)
    trainloader = Flux.DataLoader((; x= create_X(d_train, predictors), y=reshape(d_train[target], 1, :)), batchsize=batchsize, shuffle=true, partial=true)
    valiloader = Flux.DataLoader((; x= create_X(d_vali, predictors), y=reshape(d_vali[target], 1, :)), batchsize=length(d_vali[target]), shuffle=false, partial=false)
   
    # Save a copy of the initial model and initialize loss values
    bestModel=deepcopy(model)
    vali_losses=[lossfun(model, valiloader |> first)]
    best_vali_loss = vali_losses[1]
    patience_cnt = 0 
    train_losses=[lossfun(model, trainAll |> first)]

    # Record latent variables if specified
    if isnothing(latents2record) 
        trace_param=nothing
    else 
        trace_param = map(latents2record) do x x => copy(getfield(model, x)) end
    end

    # Initialize plotting windows
    trainScreen=GLMakie.Screen()
    inferScreen=GLMakie.Screen()

    # Train the model for n_epoch epochs
    for e in 1:n_epoch
        for d in trainloader
            ∂L∂m = gradient(lossfun, model, d)[1]
            Flux.update!(opt_state, model, ∂L∂m)
        end
        # Record loss values and latent variables (if specified) after each epoch      
        push!(vali_losses, lossfun(model, valiloader |> first))
        push!(train_losses, lossfun(model, trainAll |> first))
        if !isnothing(latents2record) foreach(trace_param) do x push!(x.second, copy(getfield(model, x.first)[1])) end end

        # Print and plot progress at specified intervals
        e % printevery == 1 && println("Epoch: $e, Train: $(train_losses[end]), Vali: $(vali_losses[end]), Pat_cnt: $patience_cnt")
        e % plotevery == 1 && plotfit(trainScreen, (; train=train_losses, vali=vali_losses), model, trainAll |> first, valiloader |> first, trace_param)
        #println(valiloader |> first |> keys)
        e % plotevery == 1 && eval_all(inferScreen, model, d_vali) ## Plots the inferred variables. Will break with normal model

        # Update best model accordings to validation loss and track patience
        if last(vali_losses) < best_vali_loss
            patience_cnt = 0
            best_vali_loss = last(vali_losses)
            bestModel=deepcopy(model)
        else
            patience_cnt += 1
            patience_cnt > patience && break
        end
       
    end #epoch

    # Plot final results:
    println("Final Plotting: Best model")
    println("Best validation losses: $(best_vali_loss)")
    plotfit(trainScreen, (; train=train_losses, vali=vali_losses), bestModel, trainAll |> first, valiloader |> first, trace_param)
        #println(valiloader |> first |> keys)
    eval_all(inferScreen, bestModel, d_vali)

    # Return named tuple storing losses, bestmodel, state of optimizer and the trace of the (physical) parameters
    return (; train_losses, vali_losses, bestModel, opt_state, trace_param)
end


predict_all(model, data) = model(data, :infer)

function eval_all(screen, model, data, vars2plot=nothing)
    
    pred=predict_all(model, data)
    nvar=length(pred)
    if isnothing(vars2plot)
        vars2plot = [v => Symbol(v, "_syn")  for v in keys(pred)]
    end

    fig=Figure(resolution=(1920,1080))
    axes=[]
    for (i, p) in enumerate(vars2plot)
        col=(i-1) % 3 + 1
        row= (i-1) ÷ 3 + 1
        push!(axes, Axis(fig[row, col], xlabel=p.first |> string, ylabel=p.second |> string))
        scatter!(axes[end], pred[p.first], data[p.second], color=(:black, 0.1))
        ablines!(0, 1, color=:red, linestyle=:dash)
    end

    display(screen, fig)

end

##