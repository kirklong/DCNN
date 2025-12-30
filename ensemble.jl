#!/usr/bin/env julia
using CUDA, cuDNN, JLD2, Plots
#ensemble uncertainty method 

include("DCNN.jl")
dev = gpu_device()

score(trainLoss,testLoss) = testLoss #(trainLoss + testLoss) / 2 #actually we just want models with the best validation loss 

function ensembleStats(pred, σs, testData; mode="chi2", dim=1)
    testX, testY = testData
    ret = nothing
    if mode == "chi2"
        χ²s = dim == 1 ? zeros(size(pred,3)) : zeros(size(pred,4))
        n = dim == 1 ? size(pred,3) : size(pred,4)
        for i in 1:n
            mask = dim == 1 ? vec((σs[:,:,i] .> 0.0) .& (testY[:,:,i] .> 0.0)) : vec((σs[:,:,:,i] .> 0.0) .& (testY[:,:,:,i] .> 0.0))
            χ²s[i] = dim == 1 ? sum((vec(pred[:,:,i])[mask] .- vec(testY[:,1,i])[mask]).^2 ./ vec(σs[:,:,i])[mask].^2)/sum(mask) : sum((vec(pred[:,:,:,i])[mask] .- vec(testY[:,:,:,i])[mask]).^2 ./ vec(σs[:,:,:,i])[mask].^2)/sum(mask)
        end
        ret = χ²s
    else
        ret = dim == 1 ? zeros(size(pred,3)) : zeros(size(pred,4))
        n = dim == 1 ? size(pred,3) : size(pred,4)
        for i in 1:n
            ret[i] = dim == 1 ? sum((vec(pred[:,:,i]) .- vec(testY[:,1,i])).^2)/size(testY[:,1,i], 1) : sum((vec(pred[:,:,:,i]) .- vec(testY[:,:,:,i])).^2)/size(testY[:,:,:,i], 1)
        end
    end
    return ret
end

function ensembleConfidenceHist(mList, test, index=1)
    x,y = test
    nT, nLC, nBatch = size(x)
    predTmp = zeros(Float32, nT, nLC, nBatch, length(mList))
    for (i, m) in enumerate(mList)
        predTmp[:,:,:,i] .= m(x)
    end
    μ = mean(predTmp, dims=4) # average prediction across ensemble
    σ = std(predTmp, dims=4) # standard deviation across ensemble
    σZeroMask = σ .== 0.0
    σ[σZeroMask] .= 1.0 # replace zeros with 1.0 for division
    yi = vec(y[:,1,index])
    μi = vec(μ[:,1,index])
    σi = vec(σ[:,1,index])
    mask = σi .> 0.0 # avoid division by zero
    thing1 = sum((yi[mask] .- μi[mask]).^2 ./ σi[mask].^2)/sum(mask)
    thing2 = zeros(length(mList))
    for (i, m) in enumerate(mList)
        predi = vec(predTmp[:,1,index,i])
        thing2[i] = sum((yi[mask] .- predi[mask]).^2 ./ σi[mask].^2)/sum(mask)
    end
    return thing1, thing2
end

function keepTopModels(lossList, mList, keepThreshold=1.0, useLoss=true, train=nothing, test=nothing, nDim=1, nBatch=2^6, maxChannels=32)
    if !useLoss
        for (i,m) in enumerate(mList)
            println("generating score for model $i of $(length(mList))")
            if typeof(m) == String
                m = nDim == 1 ? load_CNN_ensemble((size(train[1],1),size(train[1],2),nBatch), fname=m, nDim=nDim, maxChannels=maxChannels) : load_CNN_ensemble((size(train[1],1),size(train[1],2),size(train[1],3),nBatch), fname=m, nDim=nDim, maxChannels=maxChannels)
            end
            m = m |> dev
            trainDL = Flux.DataLoader(train, batchsize=nBatch, shuffle=false, partial=false) 
            trainLoss = 0.0
            for (x,y) in trainDL
                trainLoss += loss(m, x, y)
            end
            testDL = Flux.DataLoader(test, batchsize=nBatch, shuffle=false, partial=false)
            testLoss = 0.0
            for (x,y) in testDL
                testLoss += loss(m, x, y)
            end
            lossList[1,end,i] = trainLoss
            lossList[2,end,i] = testLoss
            m = m |> cpu
            GC.gc(true); CUDA.reclaim() #force garbage collection to avoid memory issues
        end
    end
    scores = [score(lossList[1,end,i], lossList[2,end,i]) for i in 1:size(lossList,3)] #average final epoch train/test loss
    sortedIndices = sortperm(scores) 
    σ = std(scores) # standard deviation of scores
    cutoffScore = scores[sortedIndices[1]] + σ * keepThreshold # keep models with scores within this threshold of the best model
    topModels = []; inds = []; 
    for i in sortedIndices
        if scores[i] <= cutoffScore
            push!(topModels, mList[i])
            push!(inds, i)
        end
    end
    return topModels, inds, scores, lossList
end

function visualizeEnsemble(pred, σs, losses, σlosses, testData, C; 
    index=1, t::Array{Float64,1}=collect(range(0,stop=100,length=1001)),
    batchsize=1,tags=nothing,includeMEMecho=false,saveDir="",stats=nothing,p2yScale=:log)
    
    testX,testY = testData
    p2 = Plots.plot(xlabel="epoch",ylabel="loss [custom]")
    if p2yScale == :log
        plotLosses = log10.(losses)
        σPlotLosses = log10.(σlosses)
    else
        plotLosses = losses
    end
    if length(size(losses)) == 1
        p2 = Plots.plot!(plotLosses,label="loss",c=:crimson,lw=2)
        p2max = length(losses)
    else
        x = 0:size(losses,2)-1
        p2 = Plots.plot!(x,plotLosses[1,:,:],ribbon=σPlotLosses[1,:,:],label="train",c=:dodgerblue,lw=2)
        p2 = Plots.plot!(x,plotLosses[2,:,:],ribbon=σPlotLosses[2,:,:],label="validation",c=:crimson,lw=2)
    end
    lNorm = maximum(abs.(testX[:,1,index]))
    CNorm = maximum(abs.(C))
    ΨNorm = maximum(abs.(testY[:,1,index]))

    p1 = Plots.plot(t,testX[:,1,index]./lNorm,label="L",lw=1.5)
    p1 = Plots.plot!(t,C./CNorm,label="C",lw=1.5)
    p1 = Plots.plot!(t,testY[:,1,index]./ΨNorm,label="Ψ",lw=2)
    μ = vec(pred[:,:,index])
    σ = vec(σs[:,:,index])
    p1 = Plots.plot!(t,μ./ΨNorm,ribbon=σ./ΨNorm,label="Predicted Ψ and σ (ribbon)",lw=2)
    xcorr_t = range(-maximum(t),stop=maximum(t),length=2*length(t)-1)
    CCF = DSP.xcorr(testX[:,1,index],C)
    p1 = Plots.plot!(xcorr_t,CCF./maximum(CCF),label="CCF",lw=1)
    dt = t[2]-t[1]
    τgiven = sum(testY[:,1,index].*t.*dt)/sum(testY[:,1,index].*dt)
    τpred = sum(μ.*t.*dt)/sum(μ.*dt)
    τu = sum((μ.+σ).*t.*dt)/sum((μ.+σ).*dt) # upper bound
    τl = sum((μ.-σ).*t.*dt)/sum((μ.-σ).*dt) # lower bound
    p1 = Plots.vline!(p1,[τgiven],label="true τ ($(round(τgiven, sigdigits=3)))",c=3,ls=:dash,lw=1.5)
    p1 = Plots.vline!(p1,[τpred],label="predicted τ ($(round(τpred, sigdigits=3)) + $(round(τu-τpred, sigdigits=3))/-$(round(τpred-τl, sigdigits=3)))",c=4,ls=:dash,lw=1.5)
    p1 = Plots.plot!(legend=:outertop,legend_columns=3,xlims=(0,maximum(t)),xlabel="time [days]",ylabel="normalized value")
    if includeMEMecho && !isnothing(tags)
        ΨtMax = tags[index]
        t,Ψ,col3 = read_dat("../STORM/MEMecho/kirkFakeMEMechoResults/"*saveDir*"out_$(ΨtMax)_Psi.dat")
        mask = t .>= 0.0
        t = t[mask]; Ψ = Ψ[mask]
        p1 = Plots.plot!(t,Ψ./maximum(Ψ),label="MEMecho Ψ",lw=2,c=7)
        τMEM = sum(Ψ.*t.*dt)/sum(Ψ.*dt)
        p1 = Plots.vline!(p1,[τMEM],label="MEMecho τ ($(round(τMEM, sigdigits=3)))",c=7,ls=:dash,lw=1.5)
    end
    if !isnothing(stats)
        p1 = Plots.plot!(title="reduced χ² = $(round(stats[index], sigdigits=3))")
    end
    return Plots.plot(p1,p2,layout=@layout([a{0.7h}; b{0.3h}]),size=(600,450))
end

function main(saveDir=""; t=collect(range(0, stop=400, length=401)), ΨtMaxRange=range(5,stop=25,length=2^12), nPerΨ=2^3, nLC=1,
                nBatch=2^6, nEnsemble=20, maxEpochs=100, autostopMemoryFrac=10, loadData=false, useδc=false, transfer=false,
                nPool=10,maxChannels=4,trainModels=true,include_MEM_outputs=true,generate_MEM_outputs=true, keepModelsInMemory=true,
                topModelσCut=1.0,nCombinedSamples=0,schedule=false,nDim=1,nC=1,ensembleOffset=0,C=nothing,thin=1.0,useLoss=false)

    nT=length(t)
    autostopMemory = round(Int, maxEpochs/autostopMemoryFrac) #autostop memory size
    #DRW args
    σ = 0.2
    μ = 1.0
    τ = 50.0
    #noise args
    noiseLevel = 5e-3
    #δC args
    tPulse = 10.0 #place where delta function is applied, if using δC
    #meta args
    # loadData = true
    # useδc = true
    # transfer = false #use transfer learning from previous model?

    if loadData
        println("loading train/test/C data from file")
        train = load(saveDir*"trainTestC.jld2", "train")
        test = load(saveDir*"trainTestC.jld2", "test")
        Cfull = load(saveDir*"trainTestC.jld2", "C")
        Ψtags = load(saveDir*"trainTestC.jld2", "Ψtags")
        tC = load(saveDir*"trainTestC.jld2", "tC")
        choiceTags = load(saveDir*"trainTestC.jld2", "choiceTags")
        C = Cfull[length(t):end]
    else
        if !isdir(saveDir*"MEMechoSamples/")
            mkdir(saveDir*"MEMechoSamples/")
        end
        if !isdir(saveDir) && saveDir != ""
            mkdir(saveDir)
        end
        oldFiles = readdir(saveDir*"MEMechoSamples/")
        for f in oldFiles
            rm(saveDir*"MEMechoSamples/"*f)
        end
        if isfile(saveDir*"MEMechoSamples/"*"trainTestC.jld2")
            rm(saveDir*"MEMechoSamples/"*"trainTestC.jld2") #remove old data file 
        end
        train,test,tC,Cfull,Ψtags,choiceTags = getData_NEW(ΨtMaxRange=ΨtMaxRange,nPerΨ=nPerΨ,t=t,ensemble=true,writeMEMecho=true,σ=σ,μ=μ,τ=τ,C=C,
                                            noiseLevel=noiseLevel,useδc=useδc,tPulse=tPulse,expΨ=true,KDM=true,useBLRDisk=true,gaussian=true,ellipse=true,useBLRClouds=true,
                                            saveDir=saveDir,include_MEM_outputs=include_MEM_outputs,generate_MEM_outputs=generate_MEM_outputs,nCombinedSamples=nCombinedSamples)
        C = Cfull[length(t):end]
        println("saving train/test/C/tags data to file")
        save(saveDir*"trainTestC.jld2", "train", train, "test", test, "C", Cfull, "Ψtags", Ψtags, "tC", tC, "choiceTags", choiceTags)
    end
    if thin > 0.0 && thin < 1.0
        println("randomly selecting $((thin)*100)% of training samples")
        if nDim == 1
            randKeepTrain = rand(1:size(train[1],3),round(Int, size(train[1],3)*(thin)))
            randKeepTest = rand(1:size(test[1],3),round(Int, size(test[1],3)*(thin)))
            train = (train[1][:,:,randKeepTrain], train[2][:,:,randKeepTrain])
            test = (test[1][:,:,randKeepTest], test[2][:,:,randKeepTest])
            Ψtags = (Ψtags[1][randKeepTrain], Ψtags[2][randKeepTest])
            choiceTags = (choiceTags[1][randKeepTrain], choiceTags[2][randKeepTest])
        else
            randKeepTrain = rand(1:size(train[1],4), round(Int, size(train[1],4)*(thin)))
            randKeepTest = rand(1:size(test[1],4), round(Int, size(test[1],4)*(thin)))
            train = (train[1][:,:,:,randKeepTrain], train[2][:,:,:,randKeepTrain])
            test = (test[1][:,:,:,randKeepTest], test[2][:,:,:,randKeepTest])
            Ψtags = (Ψtags[1][randKeepTrain], Ψtags[2][randKeepTest])
            choiceTags = (choiceTags[1][randKeepTrain], choiceTags[2][randKeepTest])
        end
    end

    train = train |> dev
    test = test |> dev

    mList = []; 
    lossList = zeros(2, maxEpochs+1, nEnsemble) # store losses for each model in ensemble
    rateList = zeros(nEnsemble) # store learning rates for each model in ensemble
    fullInds = collect(1:nEnsemble)
    rateList = zeros(nEnsemble) # store learning rates for each model in ensemble
    stopList = zeros(nEnsemble) # store autostop status for each model in ensemble
    for i in 1:nEnsemble
        # model = CNN((nT,nLC,nBatch))
        mID = i+ensembleOffset
        if trainModels
            if nDim == 1
                model = transfer ? load_CNN_ensemble((nT,nC,nBatch),fname=saveDir*"ensemble_$mID.jld2",nPool=nPool,maxChannels=maxChannels,nDim=nDim) : CNN_ensemble((nT,nC,nBatch),nPool=nPool,maxChannels=maxChannels,nDim=nDim)
            else
                model = transfer ? load_CNN_ensemble((nT,nLC,nC,nBatch),fname=saveDir*"ensemble_$mID.jld2",nPool=nPool,maxChannels=maxChannels,nDim=nDim) : CNN_ensemble((nT,nLC,nC,nBatch),nPool=nPool,maxChannels=maxChannels,nDim=nDim)
            end
            # model = CNN_simple_baseline((nT,nLC,nBatch)) #simple baseline for now for debugging
            m = model |> dev
            learningRate = 3e-3/20 #3e-3/20 #good baseline, was 3e-3/20
            title="Training model $i of $nEnsemble with learning rate $learningRate"
            println(title)
            rateList[i] = learningRate
            m,losses = trainModel!(m,loss,train,test,epochs=maxEpochs,autostop=true,autostopMemory=autostopMemory,batchsize=nBatch,learningRate=learningRate,partial=false,
                        nPool=nPool,maxChannels=maxChannels,schedule=schedule,title=title,nDim=nDim,saveDir=saveDir)          
            m = m |> cpu
            fname = transfer ? "ensemble_$(mID)_transfer.jld2" : "ensemble_$(mID).jld2"
            saveModel(m, fname=saveDir*fname)
            if keepModelsInMemory
                push!(mList, m)
            else
                push!(mList, saveDir*fname)
            end
            stopList[i] = size(losses)[2] #how many epochs were needed before autostop?
            stopInd = Int(stopList[i])
            if stopInd < maxEpochs+1
                lossList[:,1:stopInd,i] .= losses
                lossList[:,stopInd+1:end,i] .= losses[end] # fill remaining epochs with last loss
            else
                lossList[:,:,i] .= losses
            end
        else
            fname = transfer ? "ensemble_$(mID)_transfer.jld2" : "ensemble_$mID.jld2"
            if keepModelsInMemory
                println("Loading model $i of $nEnsemble from file")
                m = nDim == 1 ? load_CNN_ensemble((nT,nLC,nBatch),nPool=nPool,maxChannels=maxChannels,fname=saveDir*fname,nDim=nDim) : load_CNN_ensemble((nT,nLC,nC,nBatch),nPool=nPool,maxChannels=maxChannels,fname=saveDir*"ensemble_$mID.jld2",nDim=nDim) #revert to checkpoint
                push!(mList, m)
            else
                push!(mList, saveDir*fname) # store file name instead of model
            end
        end
    end
    if !trainModels
        fname = transfer ? "results_dict_transfer.jld2" : "results_dict.jld2"
        if isfile(saveDir*fname)
            try
                println("Loading previous results from $(saveDir*fname)")
                d = load(saveDir*fname,"dict")
                lossListTmp = d["lossList"] # load losses from file
                rateListTmp = d["learningRate"] # load learning rate from file
                if size(lossListTmp,3) == nEnsemble
                    lossList = lossListTmp
                    rateList = rateListTmp
                else
                    println("Previous results have $(size(lossListTmp,3)) ensemble members, but nEnsemble=$nEnsemble. Losses for $(nEnsemble-size(lossListTmp,3))/$nEnsemble models will be set to final values.")
                    fullInds = [parse(Int,split(s,"_")[end][1:end-5]) for s in d["mList"]]
                    for (i,iFull) in enumerate(fullInds)
                        lossList[:,:,iFull] .= lossListTmp[:,:,i]
                        rateList[iFull] = rateListTmp[i]
                    end
                end
            catch e
                println("Error loading previous results: $e")
                lossList = zeros(2, maxEpochs+1, nEnsemble) # initialize losses if not loading
                rateList = zeros(nEnsemble) # initialize learning rates if not loading
            end
        else
            println("No previous results found, losses and learning rates will be zeros")
            lossList = zeros(2, maxEpochs+1, nEnsemble) # initialize losses if not loading
            rateList = zeros(nEnsemble) # initialize learning rates if not loading
        end
    end
    topModels, topInds, scores, lossList = keepTopModels(lossList, mList, topModelσCut, useLoss, train, test, nDim, nBatch, maxChannels) # keep models within 1 std deviation of the best model
    pred, σs, lowers, uppers = ensemblePredict(topModels, test, nBatch, nT, nLC, nC, maxChannels=maxChannels, nPool=nPool, partial=false, dev=dev,nDim=nDim) # predict with top models
    topScores = [score(lossList[1,end,i], lossList[2,end,i]) for i in topInds] # scores of top models
    test = test |> cpu
    train = train |> cpu
    losses = mean(lossList, dims=3) # average losses across ensembles
    σlosses = std(lossList, dims=3) # standard deviation of losses across ensembles
    topLosses = mean(lossList[:,:,topInds], dims=3) # average losses of top models
    topσLosses = std(lossList[:,:,topInds], dims=3) # standard deviation of losses of top models
    topRates = rateList[topInds] # learning rates of top models
    allScores = [score(lossList[1,end,i], lossList[2,end,i]) for i in 1:size(lossList,3)] # all model scores
    notTopMask = .!([i in topInds for i in 1:size(lossList,3)])
    println("selected $(length(topModels)) top models")
    println("\ttop model score distribution (l, μ, σ, u): ($(round(minimum(topScores), sigdigits=2)), $(round(mean(topScores), sigdigits=2)), $(round(std(topScores), sigdigits=2)), $(round(maximum(topScores), sigdigits=2)))")
    println("\toverall model score distribution (l, μ, σ, u): ($(round(minimum(allScores), sigdigits=2)), $(round(mean(allScores), sigdigits=2)), $(round(std(allScores), sigdigits=2)), $(round(maximum(allScores), sigdigits=2)))")
    println("\tnon-top model score distribution (l, μ, σ, u): ($(round(minimum(allScores[notTopMask]), sigdigits=2)), $(round(mean(allScores[notTopMask]), sigdigits=2)), $(round(std(allScores[notTopMask]), sigdigits=2)), $(round(maximum(allScores[notTopMask]), sigdigits=2)))")
    println("\ttop model learning rate distribution (l, μ, σ, u): ($(round(minimum(topRates), sigdigits=2)), $(round(mean(topRates), sigdigits=2)), $(round(std(topRates), sigdigits=2)), $(round(maximum(topRates), sigdigits=2)))")
    if sum(notTopMask) > 0 
        println("\tnon-top model learning rate distribution (l, μ, σ, u): ($(round(minimum(rateList[notTopMask]), sigdigits=2)), $(round(mean(rateList[notTopMask]), sigdigits=2)), $(round(std(rateList[notTopMask]), sigdigits=2)), $(round(maximum(rateList[notTopMask]), sigdigits=2)))")
    end
    d = Dict("topModels" => topModels, "topInds" => topInds, "topScores" => topScores, 
            "pred" => pred, "σs" => σs, "losses" => losses, "σlosses" => σlosses,"learningRate" => rateList,
            "topLosses" => topLosses, "topσLosses" => topσLosses, "train" => train, "lossList" => lossList,
            "test" => test, "Cfull" => Cfull, "Ψtags" => Ψtags, "tC" => tC, "C" => C,"lowers" => lowers,
            "uppers" => uppers, "mList" => mList, "rateList" => rateList, "choiceTags" => choiceTags, "t" => t,
            "nBatch" => nBatch, "nLC" => nLC, "nT" => nT, "nPerΨ" => nPerΨ, "nPool" => nPool, "maxChannels" => maxChannels)
    fname = transfer ? "results_dict_transfer.jld2" : "results_dict.jld2"
    println("saving dictionary of results to $fname")
    save(saveDir*fname, "dict", d)
    return d
end
