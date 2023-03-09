include("hybrid_mr.jl")
GLMakie.activate!(;float=true)
update_theme!(fontsize=2 * 14)


struct FluxPartModel_Q10
    RUE_chain::Flux.Chain
    RUE_predictors::AbstractArray{Symbol}
    Rb_chain::Flux.Chain
    Rb_predictors::AbstractArray{Symbol}
    Q10
end

function FluxPartModel_Q10(RUE_predictors::AbstractArray{Symbol}, Rb_predictors::AbstractArray{Symbol}, Q10=[1.5f0])
    RUE_ch = chain4RUEfun(length(RUE_predictors))
    Rb_ch = chain4Rbfun(length(Rb_predictors))
    FluxPartModel_Q10(RUE_ch, RUE_predictors, Rb_ch, Rb_predictors, Q10)
end

chain4RUEfun(nInVar) = Flux.Chain(
    BatchNorm(nInVar, affine=true),
    Dense(nInVar => 15, relu),
    #GRU(5 => 5),
    #Dense(12 => 12, σ), 
    Dense(15 => 1, σ),
)
chain4Rbfun(nInVar) = Flux.Chain(
    BatchNorm(nInVar, affine=true),
    Dense(nInVar => 15, relu),
    #GRU(5 => 5),
    #Dense(9 => 5, σ), 
    Dense(15 => 1, σ),
)
  
function (m::FluxPartModel_Q10)(x)

    # RUE_input4Chain = vars2matrix(x, m.RUE_predictors) #mapreduce(v -> x[v], hcat, m.RUE_predictors) |> transpose
    # Rb_input4Chain = vars2matrix(x, m.Rb_predictors) #mapreduce(v -> x[v], hcat, m.Rb_predictors) |> transpose
    # RUE = 1f0 * vec(m.RUE_chain(RUE_input4Chain)) ## 0.1 to make GPP between 0 and 100
    # GPP = reshape(x.SW_IN .* RUE ./ 12.011, 1, :)
    # Rb = 100f0 * vec(m.Rb_chain(Rb_input4Chain)) ## 100 to make Rb between 0 and 100
    # Reco = @c Rb .* m.Q10[1] .^ (0.1f0(x.TA .- 15.0f0)) reshape(1, :)
    # return Reco - GPP

    res = m(x, :infer)
    return reshape(res.RECO - res.GPP, 1, :)
end

function (m::FluxPartModel_Q10)(x, mode::Symbol)

    if mode == :infer
        RUE_input4Chain = vars2matrix(x, m.RUE_predictors) #mapreduce(v -> x[v], hcat, m.RUE_predictors) |> transpose
        Rb_input4Chain = vars2matrix(x, m.Rb_predictors) #mapreduce(v -> x[v], hcat, m.Rb_predictors) |> transpose
    
        RUE = 1f0 * vec(m.RUE_chain(RUE_input4Chain))
        GPP = reshape(x.SW_IN .* RUE ./ 12.011, 1, :)  # mol/g * g / J * J /s / m2 =mol m-2 s-1
        Rb = 100f0 * vec(m.Rb_chain(Rb_input4Chain))
        Reco = @c Rb .* m.Q10[1] .^ (0.1f0(x.TA .- 15.0f0)) reshape(1,:)

        return((; RUE, Rb, GPP=GPP |> vec, RECO=Reco |> vec, NEE=vec(Reco - GPP)))
    end

end
  
  # Call @functor to allow for training the custom model
Flux.@functor FluxPartModel_Q10

##


#fpmod = FluxPartModel_Q10([:SW_IN, :TA, :VPD], [:SW_POT_sm_diff, :SW_POT_sm])
fpmod = FluxPartModel_Q10([:TA, :VPD], [:SW_POT_sm_diff, :SW_POT_sm],[1.5f0])
fpmod = FluxPartModel_Q10(chain4RUEfun(2), [:TA, :VPD], resRb.bestModel, [:SW_POT_sm, :SW_POT_sm_diff],[1.5f0])
##
df = @c "D:/markusr\\_FLUXNET\\AT-Neu.HH.csv" CSV.read(DataFrame, missingstring="NA") @transform(:NEE = coalesce(:NEE, :RECO_NT-:GPP_NT))
#select(df, names(df, Number))
transform!(df, names(df, Number) .=> ByRow(Float32), renamecols=false)
#subset(df, :doy => x-> 100 .< x .< 200)
#@subset! df 100 < :doy < 200
# mapcols(df) do col
#     eltype(col) === Int ? Float64.(col) : col
# end
df = @chain df begin 
    @transform :Rb_syn = max(:Rb_syn, 0.0f0)
    @transform :RECO_syn = :Rb_syn * 1.2f0 ^ (0.1f0(:TA - 15.0f0))
    @transform :NEE_syn = :RECO_syn - :GPP_syn
end

dfs =@c df begin 
    @transform :NEE_SWin=@bycol mapwindow2(regBivar, :SW_IN, :NEE, 49)
    unnest("NEE_SWin")
    @transform :NEE_SWin=@bycol mapwindow2((x,y)->regBivar(Float64.(x),Float64.(y), quantreg), :SW_IN, :NEE, 49)
    unnest("NEE_SWin")
    @transform {} = Float32({r"^NEE_SWin"})
    #transform(:lmres => AsTable)
    #@transform :NEE_slope=:lmres[2]
    #@transform :NEE_inter=:lmres[1]
    #select(Not(:lmres))
   end

dt = dfs |> eachcol |> pairs |> NamedTuple;
##
NEE = fpmod(dt) |> vec;
scatter(NEE)
allinfo = fpmod(dt,:infer)

##
## First try a "normal" hybrid fit to the synthetic data (works well)
##
fpmod = FluxPartModel_Q10([:TA, :VPD, :NEE_SWin_quantreg_slope], [:NEE_SWin_quantreg_inter, :TA],[1.0f0])
fpmod = FluxPartModel_Q10([:TA, :VPD], [:SW_POT_sm_diff, :SW_POT_sm],[2.0f0])
NEE = fpmod(dt) |> vec;
scatter(NEE)
allinfo = fpmod(dt,:infer);
dt=merge(dt, (; NEE_syn=dt.NEE_syn + 2f0 * randn(Float32, length(dt.NEE_syn))));

res=fit_df!(dt, nothing, :NEE_syn, (m, d) -> Flux.mse(m(d.x), d.y), model=fpmod, n_epoch=200, batchsize=480, opt=Adam(0.01), latents2record=[:Q10])
infer = res.bestModel(dt, :infer)
scatter(infer.Rb, color=(:black, 0.05))
scatter!(dt.Rb_syn, color=(:red, 0.05))

scatter(infer.Reco, dt.RECO_syn, color=(:black, 0.05))
ablines!(0,1, color=:red, linewidth=3, linestyle=:dash)

println(res.bestModel.Q10)

### Fit Rb directly with a FF Neural net
resRb=fit_df!(dt, [:SW_POT_sm, :SW_POT_sm_diff], :Rb_syn, (m, d) -> Flux.mse(m(d.x), d.y), model=chain4Rbfun(2), n_epoch=200, batchsize=480)




##
ok=findall(dt.SW_IN .< 0.1)
scatter(infer.Reco[ok], dt.NEE[ok])
scatter!(dt.RECO_NT[ok], dt.NEE[ok], color=(:red, 0.2))
ablines!(0,1, color=:black, linewidth=3, linestyle=:dash)
current_figure()
##

lines(dt.SW_POT_sm_diff)




ncd = NCDataset("DE-Tha_2010_gf.nc")
data = @chain begin
    mapreduce(x -> DataFrame(x => ncd[x][:]), hcat, keys(ncd)[2:end])
    @transform begin
        :Tair = :Tair - 273.15
        :Tair_min = :Tair_min - 273.15
        :PAR = 0.45 * :SW_IN * 86400e-6
        :GPP = :GPP * 86400e3
        :VPDday = :VPDday * 100
        #"{}" = {r"^Tair"} * 1000

    end
    NamedTuple.(eachrow(_))
    invert
    NamedTuple{keys(_)}(Float32.(d) for d in _)
end

##
chain4RUE = Flux.Chain(
    #BatchNorm(2, affine=true),
    Dense(2 => 5, σ),
    #GRU(5 => 5),
    Dense(5 => 5), 
    Dense(5 => 1, softplus),
)

chain4Rb = Flux.Chain(
    Dense(5 => 5, relu),
    Dense(5 => 5, relu),
    Dense(5 => 1, sigmoid)
)
##
hybridRUEmodel = CustomRUEModel([:VPDday, :Tair_min, :SW_IN])

GPPmod = hybridRUEmodel(data)
#RUEmod = hybridRUEmodel(data, :infer)

lines(GPPmod |> vec)
##
# lines(RUEmod |> vec)

# loss(m, d) = Flux.mse(m(d.x), d.y)

# ##
# opt_state = Flux.setup(Adam(0.001), hybridRUEmodel)   # explicit setup of optimiser momenta

# dloader = Flux.DataLoader((; x= data, y=reshape(data.GPP, 1, :)), batchsize=30, shuffle=true, partial=true)
# dloaderAll = Flux.DataLoader((; x= data, y=reshape(data.GPP, 1, :)), batchsize=length(data.GPP), shuffle=true, partial=true)
# ##
# epochs=100
# losses=[loss(hybridRUEmodel, dloaderAll |> first)]
# for e in 1:epochs
#     for d in dloaderAll
#         ∂L∂m = gradient(loss, hybridRUEmodel, d)[1]
#         Flux.update!(opt_state, hybridRUEmodel, ∂L∂m)
#     end
#     push!(losses, loss(hybridRUEmodel, dloaderAll |> first) )
#    # Flux.train!(loss, hybridRUEmodel, dloader, opt_state)
# end


struct CustomRUEModel
    RUEchain::Flux.Chain
    RUEpredictors::AbstractArray{Symbol}
    #Rbchain::Flux.Chain
end

function CustomRUEModel(predictors::AbstractArray{Symbol})
    ch = chain4RUEfun(length(predictors))
    CustomRUEModel(ch, predictors)
end

chain4RUEfun(nInVar) = Flux.Chain(
    #BatchNorm(2, affine=true),
    Dense(nInVar => 5, σ),
    #GRU(5 => 5),
    Dense(5 => 5), 
    Dense(5 => 1, softplus),
)
  
function (m::CustomRUEModel)(x)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    #return x.PAR .*  x.fPAR .* vec(m.RUEchain([x.PAR x.VPDday x.Tair_min] |> transpose))

    input4Chain = mapreduce(v->x[v], hcat, m.RUEpredictors) |> transpose

    return reshape(x.PAR .*  x.fPAR .* vec(m.RUEchain(input4Chain)), 1, :)
end

function (m::CustomRUEModel)(x, stateful=true)
    # Arbitrary code can go here, but note that everything will be differentiated.
    # Zygote does not allow some operations, like mutating arrays.
    #return x.PAR .*  x.fPAR .* vec(m.RUEchain([x.PAR x.VPDday x.Tair_min] |> transpose))
    #!stateful && return x.PAR .*  x.fPAR .* vec(m.RUEchain([ x.VPDday x.Tair_min] |> transpose))
    out = reduce(hcat, [m.RUEchain(s) for s in eachslice([ x.VPDday x.Tair_min], dims=1)])
    

end

function (m::CustomRUEModel)(x, mode::Symbol)
    mode == :infer && vec(m.RUEchain([ x.VPDday x.Tair_min] |> transpose))
end
  
  # Call @functor to allow for training. Described below in more detail.
Flux.@functor CustomRUEModel

# fit_df  <- function(df=NULL, batchSize=1000L, lr=0.05, n_epoch=100L, model=NULL, startFromResult=NULL, 
#                     predictors=c("WS", "VPD","TA", "SW_IN", "SW_POT_sm", "SW_POT_sm_diff"), target="NEE", seqID=NULL, seqLen=NA, 
#                     weights=NULL, checkpoint="R_checkpoint.pt", DictBased=F,
#                     patience=50, lossFunc=lossFuncs$trimmedLoss, justKeepOnTraining=FALSE, ...) {

##