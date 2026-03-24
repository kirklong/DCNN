#!/usr/bin/env julia
using Flux, ChainRulesCore, Random, Term
import UnicodePlots: lineplot

include("fakeDataGen.jl")
include("memecho_util.jl")

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@layer Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

# custom join layer
struct Join{T, F}
  combine::F
  paths::T
end

# allow Join(op, m1, m2, ...) as a constructor
Join(combine, paths...) = Join(combine, paths)
(m::Join)(xs::Tuple) = m.combine(map((f, x) -> f(x), m.paths, xs)...)
(m::Join)(xs...) = m(xs)

function CNN_ensemble(inputShape; maxChannels=32, dropout_rate=0.1, nPool=4, nDim=1) #this version works pretty well (train = 0.65, val = 1.07, ~30-40 epochs, learningRate ~1e-4)
    if nDim == 1
        nT = inputShape[1]
        nLC = inputShape[2] 
        nBatch = inputShape[3]
        smallFilter = max(2, convert(Int, floor(nT/100)))
        mediumFilter = max(3, convert(Int, floor(nT/50)))
        mediumFilter2 = max(5, convert(Int, floor(nT/10)))
        largeFilter = max(7, convert(Int, floor(nT/5)))
        
        CNN_model = Chain(
            BatchNorm(nLC, relu),
            Split(
                Chain(
                    Conv((smallFilter,), nLC=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                ),
                Chain(
                    Conv((mediumFilter,), nLC=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate),
                    Conv((mediumFilter2,), maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                ),
                Chain(
                    Conv((largeFilter,), nLC=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                )
            ),
            # Join the parallel paths
            Join((a, b, c) -> cat(a, b, c, dims=2),
                Chain(x -> x),
                Chain(x -> x),
                Chain(x -> x)
            ),
            SkipConnection(
                Chain(
                    Conv((smallFilter,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((mediumFilter,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((mediumFilter2,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((largeFilter,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu
                ),
                +
            ),
            Dropout(dropout_rate),
            
            # do it again, experiment with pooling layer to get rid of extra channels? (train = ~0.3, test = ~0.8 after ~30-40 epochs)
            Split(
                Chain(
                    Conv((smallFilter,), 3*maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                ),
                Chain(
                    Conv((mediumFilter,), 3*maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate),
                    Conv((mediumFilter2,), maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                ),
                Chain(
                    Conv((largeFilter,), 3*maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                )
            ),
            # Join the parallel paths
            Join((a, b, c) -> cat(a, b, c, dims=2),
                Chain(x -> x),
                Chain(x -> x),
                Chain(x -> x)
            ),
            SkipConnection(
                Chain(
                    Conv((smallFilter,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((mediumFilter,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((mediumFilter2,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((largeFilter,), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu
                ),
                +
            ),
            Dropout(dropout_rate),
            
            #final output layer
            Conv((1,), 3*maxChannels=>nLC, relu, pad=Flux.SamePad(), bias=true)
        )
    else
        nT = inputShape[1]
        nLC = inputShape[2] 
        nC = inputShape[3] 
        nBatch = inputShape[4]
        smallFilter_t = max(2, convert(Int, floor(nT/100)))
        mediumFilter_t = max(5, convert(Int, floor(nT/50)))
        largeFilter_t = max(7, convert(Int, floor(nT/5)))
        smallFilter_v = floor(Int, nLC/6) #max(2, convert(Int, floor(nLC/6)))
        mediumFilter_v = floor(Int, nLC/4) #max(5, convert(Int, floor(nLC/4)))
        largeFilter_v = floor(Int, nLC/2) #max(7, convert(Int, floor(nLC/2)))
        CNN_model = Chain(
            BatchNorm(nC, relu),
            Split(
                Chain(
                    Conv((smallFilter_t,1), nC=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate),
                    Conv((1,smallFilter_v), maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu
                ),
                Chain(
                    Conv((mediumFilter_t,1), nC=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate),
                    Conv((1,mediumFilter_v), maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                ),
                Chain(
                    Conv((largeFilter_t,1), nC=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate),
                    Conv((1,largeFilter_v), maxChannels=>maxChannels, pad=Flux.SamePad(),init=Flux.glorot_normal,bias=false),
                    BatchNorm(maxChannels),
                    relu,
                    Dropout(dropout_rate)
                )
            ),
            # Join the parallel paths
            Join((a, b, c) -> cat(a, b, c, dims=3),
                Chain(x -> x),
                Chain(x -> x),
                Chain(x -> x)
            ),
            SkipConnection(
                Chain(
                    Conv((smallFilter_t,1), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((mediumFilter_t,1), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((largeFilter_t,1), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((1,smallFilter_v), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((1,mediumFilter_v), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                    Dropout(dropout_rate),

                    Conv((1,largeFilter_v), 3*maxChannels=>3*maxChannels, pad=Flux.SamePad(), bias=false, init=Flux.glorot_normal),
                    BatchNorm(3*maxChannels),
                    relu,
                ),
                +
            ),
            Dropout(dropout_rate),

            #final output layer
            Conv((1,1), 3*maxChannels=>nC, relu, pad=Flux.SamePad(), bias=true)
        )
    end
    return CNN_model
end

function loss_alternative(m, x, y, Pc = 0.68, λ = 1.0) 
    ŷmin, ŷμ, ŷmax = m(x) # get model predictions
    yμ = view(y, :, 1, :) 
    yμNoisey = view(y, :, 2:size(y, 2), :) 
    f_inBounds = 0.0
    for i=1:size(yμNoisey,2)
        f_inBounds += @ignore_derivatives sum((ŷmin .<= view(yμNoisey, :, i, :)) .& (view(yμNoisey, :, i, :) .<= ŷmax)) # count how many y values are within bounds
    end
    f_inBounds = f_inBounds / (size(yμNoisey,1) * size(yμNoisey,2) * size(yμNoisey,3)) # fraction of y values within bounds
    l1 = Flux.mse(ŷμ, yμ)/Flux.mse(ŷmax, ŷmin) # mean squared error for mean prediction weighted by "σ"
    l2 = maximum([0.0, Pc - f_inBounds]) # penalty if fraction of y values outside of bounds exceeds Pc

    return l1 + λ*l2 #combine losses, hyperparamer λ controls relative weight of penalty term
end

function formatData(d;format="tuple")
    # format data for model
    t = d[1].t
    dXOut = [Array{Float32}(undef,length(t),1,1) for i in 1:length(d)]
    dYOut = [Array{Float32}(undef,length(t),1,1) for i in 1:length(d)]
    for i in 1:length(d)
        for ti in 1:length(t)
            #dXOut[i][ti,:,1] = [d[i].L[ti],d[i].C[ti]]
            dXOut[i][ti,1,1] = d[i].L[ti]
            dYOut[i][ti,1,1] = d[i].Ψ[ti]
        end
    end
    if format == "tuple"
        return [(dx,dy) for (dx,dy) in zip(dXOut,dYOut)]
    else
        return dXOut, dYOut
    end
end

function formatData(d)
    #format data for Flux.DataLoader
    t = d[1].t; nt = length(t)
    X = zeros(Float32, nt, 1, length(d)) #time, channels, obs
    Y = zeros(Float32, nt, 1, length(d))
    for (i,d) in enumerate(d)
        Y[:,1,i] .= convert.(Float32,d.Ψ)
        X[:,1,i] .= convert.(Float32,d.L) 
    end
    return (X,Y)
end

function formatData_NEW(d,nPerΨ)
    t = d[1].t; nt = length(t)
    X = zeros(Float32, nt, nPerΨ, length(d))
    Y = zeros(Float32, nt, nPerΨ+1, length(d)) #+1 for mean
    for (i,d) in enumerate(d)
        for j in 1:nPerΨ
            Y[:,j,i] .= convert.(Float32,d.Ψ[j])
        end
        Y[:,nPerΨ+1,i] .= convert.(Float32,d.Ψ[1]) #mean
        X[:,1,i] .= convert.(Float32,d.L)
    end
    return (X,Y)
end

function getData_NEW(;ν=0.0,expΨ=true,ΨtMaxRange=1:0.1:10,sineC=false,trainFrac=0.5,τ=1.0,μ=0.0,σ=0.1,
    t=collect(range(0,stop=100,length=1001)),PC=50.0,nPerΨ=10,noiseLevel=0.1,shuffle=true,writeMEMecho=true,C=nothing,
    ensemble=true,tOffset = 100., useδc=false, KDM=true, useBLRDisk=true, gaussian=true, ellipse=true, useBLRClouds=true, 
    tPulse=10.0, missingFrac=0.1,include_MEM_outputs=true,saveDir="",generate_MEM_outputs=true,nCombinedSamples=0)

    println("Generating data")
    noisy = noiseLevel > 0.0 #if noiseLevel is 0.0 then no noise is added
    tC = vcat(-reverse(t[2:end]),t) # negative times for convolution
    if isnothing(C)
        if sineC
            C = @. sin(2*pi*tC/PC)
        elseif useδc
            C = δC(t=tC,tPulse=tPulse)
        else
            C = DRW(t=tC,μ=μ,τ=τ,σ=σ)
        end
    end
    if writeMEMecho
        #remove existing files 
        if isdir(saveDir*"MEMechoSamples/")
            files = readdir(saveDir*"MEMechoSamples/")
            for file in files
                rm(saveDir*"MEMechoSamples/"*file)
            end
        else
            mkpath(saveDir*"MEMechoSamples/") #create directory if it doesn't exist
        end
    end
    # generate data
    samples = length(ΨtMaxRange) == 0 ? nPerΨ : length(ΨtMaxRange)*nPerΨ
    nt = length(t)
    X = zeros(Float32, nt, 1, samples) #time, channels, obs
    Y = zeros(Float32, nt, 1, samples)
    Clist = zeros(nt, samples)
    choices = []
    if expΨ
        push!(choices, "expΨ")
    end
    if KDM
        push!(choices, "KDM")
    end
    if useBLRDisk
        push!(choices, "useBLRDisk")
        push!(choices, "useBLRDisk")
    end
    if gaussian
        push!(choices, "gaussian")
    end
    if ellipse
        push!(choices, "ellipse")
    end
    if useBLRClouds
        push!(choices, "useBLRClouds")
        push!(choices, "useBLRClouds")
    end
    choiceTags = Array{String}(undef, samples) #store choice for each sample
    ΨtMaxTags = zeros(samples)
    strLen = 0; tStart = time()
    for (j, ΨtMax) in enumerate(ΨtMaxRange)
        Ctmp = deepcopy(C) #make a copy of C for each sample
        # Y[:, 1, j] .= convert.(Float32, truth.Ψ) #analytic truth
        choice = rand(choices)
        expΨ = choice == "expΨ"
        KDM = choice == "KDM"
        useBLRDisk = choice == "useBLRDisk"
        gaussian = choice == "gaussian"
        ellipse = choice == "ellipse"
        useBLRClouds = choice == "useBLRClouds"
        truth = genData(C,tC,t,ν=ν,expΨ=expΨ,KDM=KDM,useBLRDisk=useBLRDisk,gaussian=gaussian,ellipse=ellipse,useBLRClouds=useBLRClouds,ΨtMax=ΨtMax,noisy=false) #generate truth data
        for i in 1:nPerΨ
            if noisy
                noise = rand(Normal(0,noiseLevel*maximum(abs.(C))),length(C))
                Ctmp .+= noise
                Clist[:, (j-1)*nPerΨ+i] .= Ctmp[length(t):end] #store the noisy C for this sample, for MEMecho
            end
            choiceTags[(j-1)*nPerΨ+i] = choice
            ΨtMaxTags[(j-1)*nPerΨ+i] = ΨtMax
            d = genData(C,tC,t,ν=ν,Ψ=truth.Ψ,noisy=noisy,noiseLevel=noiseLevel,missingFrac=missingFrac) #generate data with noise
            Y[:, 1, (j-1)*nPerΨ+i] .= convert.(Float32, d.Ψ)
            X[:, 1, (j-1)*nPerΨ+i] .= convert.(Float32, d.L)
        end
        if writeMEMecho
            μ = mean(X[:, :, (j-1)*nPerΨ+1:j*nPerΨ], dims=3)
            σ = std(X[:, :, (j-1)*nPerΨ+1:j*nPerΨ], dims=3)
            # p1=plot(vec(μ),ribbon=vec(σ),title="debug $j with noiseLevel $noiseLevel")
            # p2=plot(X[:,1,(j-1)*nPerΨ+1:j*nPerΨ],title="individual light curves")
            # p2=plot!(truth.L,label="truth",color=:black,linewidth=2,linestyle=:dash)
            # plot(p1,p2,layout=@layout([a;b]),size=(800,600))
            # png("debug_$j.png")
            write_dat(saveDir*"MEMechoSamples/"*"MEMecho_$(ΨtMax).dat", t .+ tOffset, μ, σ) #write MEMecho data using mean and standard deviation of fake light curves
        end
        strLen = progress!(j, length(ΨtMaxRange), tStart, strLen)
    end
    println("distribution statistics:")
    for choice in choices
        count = sum(choiceTags .== choice)
        println("\t $choice: $count/$(samples) ($(round(count/samples*100, digits=2))%)")
    end
    if writeMEMecho
        μ = mean(Clist, dims=2)
        σ = std(Clist, dims=2)
        write_dat(saveDir*"MEMechoSamples/"*"MEMecho_C.dat", t .+ tOffset, μ, σ) #write MEMecho data using mean and standard deviation of C
    end
    if include_MEM_outputs
        if generate_MEM_outputs
            run(`julia --threads=8 ../STORM/MEMecho/kirkScripts/iterateMEMecho.jl $saveDir`)
        end
        save(saveDir*"trainTestC.jld2", "C", C, "tC", tC)
        MEMoutX,MEMoutY,MEMtags = getMEMSamples(saveDir,nPerΨ=nPerΨ,noiseLevel=noiseLevel,missingFrac=missingFrac)
        X = cat(X, MEMoutX, dims=3) #add MEMecho samples to X
        Y = cat(Y, MEMoutY, dims=3) #add MEMecho samples to Y
        ΨtMaxTags = vcat(ΨtMaxTags, MEMtags) #add MEMecho tags to ΨtMaxTags
        choiceTags = vcat(choiceTags, fill("MEMecho", size(MEMoutX, 3)))
    end
    if nCombinedSamples > 0
        newX = Array{Float32}(undef, size(X, 1), size(X, 2), nCombinedSamples*nPerΨ)
        newY = Array{Float32}(undef, size(Y, 1), size(Y, 2), nCombinedSamples*nPerΨ)
        newΨtMaxTags = zeros(nCombinedSamples*nPerΨ)
        for i in 1:nCombinedSamples
            idxs = rand(1:size(X,3), 2) #randomly select 2 samples to combine
            Ynew = mean(Y[:,:,idxs], dims=3)
            for j in 1:nPerΨ
                d = genData(C,tC,t,ν=ν,Ψ=vec(Ynew),noisy=noisy,noiseLevel=noiseLevel,missingFrac=missingFrac) #generate new light curve for combined Ψ
                newX[:,:,(i-1)*nPerΨ+j] = convert.(Float32, d.L)
                newY[:,:,(i-1)*nPerΨ+j] = convert.(Float32, d.Ψ)
                newΨtMaxTags[(i-1)*nPerΨ+j] = maximum(ΨtMaxTags) + i #unique identifier for each combined sample
            end
        end
        Y = cat(Y, newY, dims=3)
        X = cat(X, newX, dims=3)
        choiceTags = vcat(choiceTags, fill("combined", nCombinedSamples*nPerΨ))
        ΨtMaxTags = vcat(ΨtMaxTags, newΨtMaxTags)
    end
    # Group samples by ΨtMax and shuffle groups, then split into train/test (this way ensures that no group of nPerΨ is split between sets)
    uniqueΨtMax = unique(ΨtMaxTags)
    nGroups = length(uniqueΨtMax)
    if shuffle
        shuffledGroups = Random.shuffle(uniqueΨtMax)
    else
        shuffledGroups = uniqueΨtMax
    end
    nTrain = floor(Int, nGroups * trainFrac)
    trainGroups = shuffledGroups[1:nTrain]
    testGroups = shuffledGroups[nTrain+1:end]

    # Create masks for train and test samples
    trainMask = [tag in trainGroups for tag in ΨtMaxTags]
    testMask = .!trainMask

    # Split the data
    trainX = X[:, :, trainMask]
    testX = X[:, :, testMask]
    trainY = Y[:, :, trainMask]
    testY = Y[:, :, testMask]
    trainΨtMaxTags = ΨtMaxTags[trainMask]
    testΨtMaxTags = ΨtMaxTags[testMask]
    trainChoiceTags = choiceTags[trainMask]
    testChoiceTags = choiceTags[testMask]
    if shuffle 
        train_idx = Random.shuffle(1:size(trainX, 3))
        trainX = trainX[:, :, train_idx]
        trainY = trainY[:, :, train_idx]
        trainΨtMaxTags = trainΨtMaxTags[train_idx]
        trainChoiceTags = trainChoiceTags[train_idx]
        test_idx = Random.shuffle(1:size(testX, 3))
        testX = testX[:, :, test_idx]
        testY = testY[:, :, test_idx]
        testΨtMaxTags = testΨtMaxTags[test_idx]
        testChoiceTags = testChoiceTags[test_idx]
    end

    train = (trainX, trainY)
    test = (testX, testY)

    return train, test, tC, C, (trainΨtMaxTags, testΨtMaxTags), (trainChoiceTags, testChoiceTags)
end

function saveModel(m;fname="model_state.jld2")
    m = cpu(m)
    state = Flux.state(m)
    jldsave(fname; state)
end

function load_CNN_ensemble(inputShape;nPool=2,maxChannels=4,fname="model_state.jld2",nDim=1) #FIX
    state = load(fname,"state")
    m = CNN_ensemble(inputShape,nPool=nPool,maxChannels=maxChannels,nDim=nDim)
    return Flux.loadmodel!(m,state)
end

function loss(m,x,y) #simplest loss function, works fine
    return Flux.mse(m(x), y)
end

function trainModel!(model,loss,trainingData,testData;epochs=10,lossRecord=true,progress=true,autostop=true,shuffle=true,
    batchsize=1,autostopMemory=10,learningRate=1e-3,partial=true,schedule=false,nPool=2,maxChannels=4,title="",nDim=1,saveDir="")
    opt_state = Flux.setup(Flux.Adam(learningRate),model)
    lastInd = epochs+1
    tStart = time(); strLen = 0
    dataLoader = Flux.DataLoader(trainingData,batchsize=batchsize,shuffle=shuffle,partial=partial)
    testDataLoader = isnothing(testData) ? nothing : Flux.DataLoader(testData,batchsize=batchsize,shuffle=shuffle,partial=partial)
        
    if lossRecord
        losses = zeros(2,epochs+1) 
        losses[1,1] = sum(loss(model, xi, yi) for (xi,yi) in dataLoader)
        losses[2,1] = sum(loss(model, xi, yi) for (xi,yi) in testDataLoader)
        title = title*"\n"*"model initial loss (train, validation): ($(losses[1,1]), $(losses[2,1]))"
    end
    MaxScheduleAttempts = 2; nScheduleAttempts = 0
    learningRates = schedule ? [t*exp(-t/10) for t in 1:epochs].*(learningRate/(10*exp(-1))) : nothing #rise then exponential decay with peak at 10 epochs
    lastCheckPointInd = 1
    for epoch in 1:epochs
        l = 0.0 
        for (xi,yi) in dataLoader
            ltmp, ∇ = Flux.withgradient(model) do m
                loss(m, xi, yi)
            end
            l += ltmp
            if !isfinite(l)
                @warn "loss is $l for epoch $epoch" epoch
                continue
            else
                Flux.update!(opt_state, model, ∇[1])
            end
        end
        if lossRecord
            losses[1,epoch+1] = l
            testmode!(model)
            losses[2,epoch+1] = sum(loss(model, xi, yi) for (xi,yi) in testDataLoader)
            trainmode!(model)
        end
        if progress
            lOld = nothing
            if lossRecord
                l = (losses[1,epoch+1],losses[2,epoch+1])
                lOld = (losses[1,epoch],losses[2,epoch])
            end
            strLen = progress!(epoch,epochs,tStart,strLen,l,lOld=lOld,makePlot=true,losses=losses,title=title)
        end
        if schedule
            learningRate = learningRates[epoch]
            Flux.adjust!(opt_state, learningRate)
        end
        if autostop && epoch > autostopMemory && lossRecord
            meanWindowSize = round(Int, autostopMemory/2)
            if (mean(losses[1,epoch-meanWindowSize+1:epoch+1]) - losses[1,epoch+1-autostopMemory] > 1e-5) && (losses[1,epoch+1] - losses[1,epoch+1-autostopMemory] > 1e-5)
                if nScheduleAttempts < MaxScheduleAttempts # && schedule
                    nScheduleAttempts += 1
                    println("\nautostop: train loss has not changed in $(autostopMemory) epochs, scheduling a learning rate reduction")
                    title = title*"\n"*"autostop: train loss has not changed in $(autostopMemory) epochs, scheduling a learning rate reduction"
                    # Reduce learning rate
                    learningRate /= 2
                    Flux.adjust!(opt_state, learningRate)
                    autostopMemory += maximum([5,autostopMemory]) #allow the model to train a little longer
                end
                println("\nautostop: train loss has not changed in $(autostopMemory) epochs, stopping training and reverting state to epoch $(lastCheckPointInd-1) with loss (train, validation): ($(losses[1,lastCheckPointInd]), $(losses[2,lastCheckPointInd]))")
                title = title*"\n"*"autostop: train loss has not changed in $(autostopMemory) epochs, stopping training and reverting state to epoch $(lastCheckPointInd-1) with loss (train, validation): ($(losses[1,lastCheckPointInd]), $(losses[2,lastCheckPointInd]))"
                lastInd = lastCheckPointInd
                model = nDim == 1 ? load_CNN_ensemble((size(trainingData[1],1),size(trainingData[1],2),batchsize),nPool=nPool,maxChannels=maxChannels,fname=saveDir*"model_autostop_checkpoint.jld2",nDim=nDim) : load_CNN_ensemble((size(trainingData[1],1),size(trainingData[1],2),size(trainingData[1],3), batchsize),nPool=nPool,maxChannels=maxChannels,fname=saveDir*"model_autostop_checkpoint.jld2",nDim=nDim) #revert to checkpoint
                losses = losses[:,1:lastInd]
                break
            elseif (mean(losses[2,epoch-meanWindowSize:epoch]) - losses[2,epoch-autostopMemory] > 1e-5) && (losses[2,epoch+1] - losses[2,epoch+1-autostopMemory] > 1e-5)
                println("\nautostop: validation loss has not changed in $(autostopMemory) epochs, stopping training and reverting state to epoch $(lastCheckPointInd-1) with loss (train, validation): ($(losses[1,lastCheckPointInd]), $(losses[2,lastCheckPointInd]))")
                title = title*"\n"*"autostop: validation loss has not changed in $(autostopMemory) epochs, stopping training and reverting state to epoch $(lastCheckPointInd-1) with loss (train, validation): ($(losses[1,lastCheckPointInd]), $(losses[2,lastCheckPointInd]))"
                lastInd = lastCheckPointInd
                model = nDim == 1 ? load_CNN_ensemble((size(trainingData[1],1),size(trainingData[1],2),batchsize),nPool=nPool,maxChannels=maxChannels,fname=saveDir*"model_autostop_checkpoint.jld2",nDim=nDim) : load_CNN_ensemble((size(trainingData[1],1),size(trainingData[1],2),size(trainingData[1],3), batchsize),nPool=nPool,maxChannels=maxChannels,fname=saveDir*"model_autostop_checkpoint.jld2",nDim=nDim) #revert to checkpoint
                losses = losses[:,1:lastInd]
                break
            end
        end
        if lossRecord
            if losses[2,epoch+1] < minimum(losses[2,1:epoch]) #&& losses[1,epoch+1] < minimum(losses[1,1:epoch]) # actually let's just checkpoint based on validation loss
                saveModel(model,fname=saveDir*"model_autostop_checkpoint.jld2") #checkpoint if loss has improved in both train and test
                lastCheckPointInd = deepcopy(epoch+1)
            elseif !isfile(saveDir*"model_autostop_checkpoint.jld2")
                saveModel(model,fname=saveDir*"model_autostop_checkpoint.jld2") #make sure there is at least one checkpoint
                lastCheckPointInd = deepcopy(epoch+1)
            end
        end
        GC.gc(true); CUDA.reclaim() #force garbage collection to avoid memory issues
    end
    if lossRecord
        return model, losses
    else
        return model
    end
end

function ensemblePredict(mList, test, nBatch=1, nT=201, nLC=1, nC=1; nPool=2, maxChannels=4, partial=false, dev=nothing,nDim=1, returnAll=false)
    testDL = Flux.DataLoader(test, batchsize=nBatch, shuffle=false, partial=partial) 
    predictions = nDim == 1 ? zeros(Float32, nT, nLC, length(testDL)*nBatch) : zeros(Float32, nT, nLC, nC, length(testDL)*nBatch)
    if returnAll
        predictions = nDim == 1 ? zeros(Float32, nT, nLC, length(testDL)*nBatch, length(mList)) : zeros(Float32, nT, nLC, nC, length(testDL)*nBatch, length(mList))
    end
    σs = nDim == 1 ? zeros(Float32, nT, nLC, length(testDL)*nBatch) : zeros(Float32, nT, nLC, nC, length(testDL)*nBatch)
    lowers = nDim == 1 ? zeros(Float32, nT, nLC, length(testDL)*nBatch) : zeros(Float32, nT, nLC, nC, length(testDL)*nBatch)
    uppers = nDim == 1 ? zeros(Float32, nT, nLC, length(testDL)*nBatch) : zeros(Float32, nT, nLC, nC, length(testDL)*nBatch)
    if !isnothing(dev)
        testDL = testDL |> dev
        # predictions = predictions |> dev
        # σs = σs |> dev
    end
    for (j,(x, y)) in enumerate(testDL)
        predTmp = nDim == 1 ? zeros(Float32, nT, nLC, nBatch, length(mList)) : zeros(Float32, nT, nLC, nC, nBatch, length(mList))
        for (i, m) in enumerate(mList)
            if typeof(m) == String
                m = nDim == 1 ? load_CNN_ensemble((nT, nLC, nBatch), fname=m, nPool=nPool, maxChannels=maxChannels,nDim=nDim) : load_CNN_ensemble((nT, nLC, nC, nBatch), fname=m, nPool=nPool, maxChannels=maxChannels,nDim=nDim)
            end
            if !isnothing(dev)
                m = m |> dev
            end
            pred = m(x) |> cpu
            if nDim == 1 
                predTmp[:,:,:,i] .= pred
            else
                predTmp[:,:,:,:,i] .= pred
            end
        end
        if nDim == 1 && !returnAll
            predictions[:, :, (j-1)*nBatch+1:j*nBatch] .= mean(predTmp, dims=4) # average prediction across ensemble
            σs[:, :, (j-1)*nBatch+1:j*nBatch] .= std(predTmp, dims=4) # standard deviation across ensemble
            for ti = 1:nT
                for bi = 1:nBatch
                    for ni = 1:nLC
                        μi = predictions[ti,ni,(j-1)*nBatch+bi]
                        lowMask = (predTmp[ti, ni, bi, :] .< μi)
                        highMask = .!lowMask
                        if sum(lowMask) > 1
                            lowers[ti, ni, (j-1)*nBatch+bi] = std(predTmp[ti, ni, bi, lowMask]) # lower bound across ensemble
                        elseif sum(lowMask) == 1
                            lowers[ti, ni, (j-1)*nBatch+bi] = μi-predTmp[ti,ni,bi,lowMask][1]
                        end
                        if sum(highMask) > 1
                            uppers[ti, ni, (j-1)*nBatch+bi] = std(predTmp[ti, ni, bi, highMask]) # upper bound across ensemble
                        elseif sum(highMask) == 1
                            uppers[ti, ni, (j-1)*nBatch+bi] = predTmp[ti,ni,bi,highMask][1]-μi 
                        end
                    end
                end
            end
            # lowers[:, :, (j-1)*nBatch+1:j*nBatch] .= minimum(predTmp, dims=4) # lower bound across ensemble
            # uppers[:, :, (j-1)*nBatch+1:j*nBatch] .= maximum(predTmp, dims=4) # upper bound across ensemble
        elseif nDim == 2 && !returnAll
            predictions[:, :, :, (j-1)*nBatch+1:j*nBatch] .= mean(predTmp, dims=5) # average prediction across ensemble
            σs[:, :, :, (j-1)*nBatch+1:j*nBatch] .= std(predTmp, dims=5) # standard deviation across ensemble
            for ti = 1:nT
                for ci = 1:nC
                    for bi = 1:nBatch
                        for ni = 1:nLC
                            μi = predictions[ti, ni, ci, (j-1)*nBatch+bi]
                            lowMask = nothing
                            lowMask = (predTmp[ti, ni, ci, bi, :] .< μi)
                            highMask = .!lowMask
                            if sum(lowMask) > 1
                                lowers[ti, ni, ci, (j-1)*nBatch+bi] = std(predTmp[ti, ni, ci, bi, lowMask]) # lower bound across ensemble
                            elseif sum(lowMask) == 1
                                lowers[ti, ni, ci, (j-1)*nBatch+bi] = μi - predTmp[ti,ni,ci,bi,lowMask][1]
                            end
                            if sum(highMask) > 1
                                uppers[ti, ni, ci, (j-1)*nBatch+bi] = std(predTmp[ti, ni, ci, bi, highMask]) # upper bound across ensemble
                            elseif sum(highMask) == 1
                                uppers[ti, ni, ci, (j-1)*nBatch+bi] = predTmp[ti,ni,ci,bi,highMask][1] - μi
                            end
                        end
                    end
                end
            end
        else
            #return all predictions from ensemble
            if nDim == 1
                predictions[:, :, (j-1)*nBatch+1:j*nBatch, :] .= predTmp
            else
                predictions[:, :, :, (j-1)*nBatch+1:j*nBatch, :] .= predTmp
            end
        end
        GC.gc(true); CUDA.reclaim() #force garbage collection to avoid memory issues
    end
    if returnAll
        return predictions  # Return all predictions from ensemble
    else
        return predictions, σs, lowers, uppers  # Average predictions across ensemble
    end
end

function visualizeModel(m,losses,testData,C;index=1,t::Array{Float64,1}=collect(range(0,stop=100,length=1001)),batchsize=1)
    testX,testY = testData
    upper = index+batchsize-1
    lower = deepcopy(index)
    subIndex = 1
    if upper > size(testX,3)
        upper = size(testX,3)
        lower = upper - batchsize + 1
        subIndex = index - lower + 1
    end

    l = loss(m,testX[:,:,lower:upper],testY[:,:,lower:upper])
    p2 = plot(xlabel="epoch",ylabel="loss [custom]")
    if length(size(losses)) == 1
        p2 = plot!(losses,label="loss",c=:crimson,lw=2)
        p2max = length(losses)
    else
        p2 = plot!(losses[1,:]./length(testX),label="train loss",c=:crimson,lw=2)
        p2 = plot!(losses[2,:]./length(testX),label="validation loss",c=:blue,lw=2)
    end
    lNorm = maximum(abs.(testX[:,1,index]))
    CNorm = maximum(abs.(C))
    ΨNorm = maximum(abs.(testY[:,1,index]))

    p1 = plot(t,testX[:,1,index]./lNorm,label="L",lw=1.5)
    p1 = plot!(t,C./CNorm,label="C",lw=1.5)
    p1 = plot!(t,testY[:,1,index]./ΨNorm,label="Ψ",lw=2)
    μ, lnσ = m(testX[:,:,lower:upper])
    μ = vec(μ[:,:,subIndex])
    lnσ = vec(lnσ[:,:,subIndex])
    p1 = plot!(t,μ./ΨNorm,ribbon=exp.(lnσ)./ΨNorm,label="Predicted Ψ and σ (ribbon)",lw=2)
    xcorr_t = range(-maximum(t),stop=maximum(t),length=2*length(t)-1)
    CCF = DSP.xcorr(testX[:,1,index],C)
    p1 = plot!(xcorr_t,CCF./maximum(CCF),label="CCF",lw=1)
    τgiven = sum(testY[:,1,index].*t)/sum(testY[:,1,index])
    τpred = sum(μ.*t)/sum(μ)
    p1 = vline!(p1,[τgiven],label="true τ",c=3,ls=:dash,lw=1.5)
    p1 = vline!(p1,[τpred],label="predicted τ",c=4,ls=:dash,lw=1.5)
    p1 = plot!(legend=:outertop,legend_columns=3,xlims=(0,maximum(t)),xlabel="time [days]",ylabel="normalized value")

    return plot(p1,p2,layout=@layout([a{0.7h}; b{0.3h}]),size=(600,450))
end

function progress!(i,n,tStart,strLen,l;nOut=100,lOld=nothing,makePlot=false,losses=nothing,title="") #add live unicode plot of loss function? could be fun lol
    if i == 1
        modelString = "["*" "^25 *"]"*" XX% complete -- "*"estimated 00:00:00 left (0.0s/it) | loss = 0.00"
        if !isnothing(lOld)
            modelString = modelString * " ↑"
        end
        strLen = length(modelString)
    end
    if i % ceil(Int,n/nOut) == 0 #output every 1% of iterations
        tNew = time()
        timeLeft = (tNew-tStart)*(n/i - 1) #( iterLeft / iterPerSec ) =  (n-k) / (k/(tNew-tStart)) = (tNew-tStart)*(n/k - 1)
        hours = floor(Int,timeLeft/3600)
        hours = hours < 10 ? "0$(hours)" : string(hours)
        minutes = floor(Int,(timeLeft % 3600)/60)
        minutes = minutes < 10 ? "0$(minutes)" : string(minutes)
        seconds = round(Int,(timeLeft % 3600) % 60)
        seconds = seconds < 10 ? "0$(seconds)" : string(seconds)
        place = Int(ceil(nOut*i/n)) 
        percent = place < 10 ? "0$(place)" : string(place)
        place = ceil(Int,place/4) #scale bar to 25 characters to prevent overflow
        progressBar = "["*"="^(place-1)*">"*" "^(25-place)*"]"
        percentStr = "$(percent)% complete"
        timeStr = i > 0 ? (parse(Int,hours) > 24 ? "> $(hours)" : "$(hours):$(minutes):$(seconds)") : "N/A"
        if length(l) == 1
            str = progressBar * " " * percentStr * " — estimated " * timeStr * " left ($(round(i/(tNew-tStart),sigdigits=2))it/s)" * " | loss = $(round(l,sigdigits=3))"
        else
            str = progressBar * " " * percentStr * " — estimated " * timeStr * " left ($(round(i/(tNew-tStart),sigdigits=2))it/s)" * " | losses (train, validation) = ($(round(l[1],sigdigits=3)), $(round(l[2],sigdigits=3)))"
        end
        if !isnothing(lOld)
            if length(l) == 1
                str = str* "    " #4 spaces
            else
                str = str* "      " #8 spaces
            end
        end
        if makePlot
            # print("\033]H\033]2J") #clear screen    
            run(`printf "\033c"`) #clear screen in bash terminal
            println(title)
        else
            print("\r"*" "^strLen*"\r")
        end
        printstyled(split(progressBar,">")[1],color=:cyan)
        printstyled(">",color=:magenta,blink=true,bold=true)
        printstyled(split(progressBar,">")[2],color=:cyan)
        print(" ")
        printstyled("$(percent)%",color=:magenta,bold=true)
        print(" complete — estimated ")
        printstyled(timeStr,color=:red,bold=true)
        print(" left (")
        printstyled("$(round(i/(time()-tStart),sigdigits=3))",color=:red,bold=true)
        if length(l) == 1
            print(" it/s) | loss = $(round(l,sigdigits=3))")
            if !isnothing(lOld)
                if l < lOld 
                    printstyled(" ↓",color=:green,bold=true)
                elseif l > lOld
                    printstyled(" ↑",color=:red,bold=true)
                else
                    printstyled(" –",color=:yellow,bold=true)
                end
            end
        else
            print(" it/s) | losses (train, validation) = ($(round(l[1], sigdigits=3))")
            if !isnothing(lOld)
                if l[1] < lOld[1]
                    printstyled(" ↓",color=:green,bold=true)
                elseif l[1] > lOld[1]
                    printstyled(" ↑",color=:red,bold=true)
                else
                    printstyled(" –",color=:yellow,bold=true)
                end
            end
            print(", $(round(l[2], sigdigits=3))")
            if !isnothing(lOld)
                if l[2] < lOld[2]
                    printstyled(" ↓",color=:green,bold=true)
                elseif l[2] > lOld[2]
                    printstyled(" ↑",color=:red,bold=true)
                else
                    printstyled(" –",color=:yellow,bold=true)
                end
            end
            print(")")
        end
        strLen = length(str) + 10 #for redundancy
        if makePlot && !isnothing(losses)
            print("\n")
            x = 0:i
            window = i <= 10 ? 1 : i-10
            p=lineplot(x, [losses[1,1:i+1] losses[2,1:i+1]],name=["train" "validation"],color=[:blue :red],xlabel="epoch", ylabel="loss [arbitrary]",
                compact=true, ylim=(0,maximum(losses[:,window:i+1])*1.1))
            Base.show(stdout,p)
        end
        flush(stdout)
    end
    if i == n
        print("\r"*" "^strLen*"\r")
        progressBar = "["*"="^24*">"*"]"
        printstyled(split(progressBar,">")[1],color=:cyan)
        printstyled(">",color=:green,bold=true)
        printstyled(split(progressBar,">")[2],color=:cyan)
        print(" ")
        printstyled("100% complete",color=:green,bold=true)
        tElapsed = time()-tStart
        hours = floor(Int,tElapsed/3600)
        hours = hours < 10 ? "0$(hours)" : string(hours)
        minutes = floor(Int,(tElapsed % 3600)/60)
        minutes = minutes < 10 ? "0$(minutes)" : string(minutes)
        seconds = round(Int,(tElapsed % 3600) % 60)
        seconds = seconds < 10 ? "0$(seconds)" : string(seconds)
        print(" — total time = $(hours):$(minutes):$(seconds) (")
        printstyled("$(round(i/(tNew-tStart),sigdigits=2))",color=:red,bold=true)
        if length(l) == 1
            print(" it/s) | final loss = $(round(l,sigdigits=3))")
        else
            print(" it/s) | final losses (train, validation) = ($(round(l[1],sigdigits=3)), $(round(l[2],sigdigits=3)))")
        end
        if !isnothing(lOld)
            if l < lOld 
                printstyled(" ↓",color=:green,bold=true)
            elseif l > lOld
                printstyled(" ↑",color=:red,bold=true)
            else
                printstyled(" –",color=:yellow,bold=true)
            end
        end
        println("\n")
        if makePlot && !isnothing(losses)
            x = 0:i
            window = i < 10 ? 1 : i-10
            p=lineplot(x, [losses[1,1:i+1] losses[2,1:i+1]],name=["train" "validation"],color=[:blue :red],xlabel="epoch", ylabel="loss [arbitrary]",
                compact=true, ylim=(0,maximum(losses[:,window:i+1])*1.1))
            Base.show(stdout,p)
        end
        flush(stdout)
    end
    return strLen
end
