#!/usr/bin/env julia
using Printf
using Interpolations

function read_dat(f)
    lines = readlines(f)
    skip = 0
    for line in lines
        if occursin("#", line)
            skip += 1
        else
            break
        end
    end
    lines = lines[skip+1:end] # skip header lines
    t = zeros(Float64, length(lines))
    LC = zeros(Float64, length(lines))
    err = zeros(Float64, length(lines))
    for (i, line) in enumerate(lines)
        parts = split(line)
        t[i] = parse(Float64, parts[1])
        LC[i] = parse(Float64, parts[2])
        err[i] = parse(Float64, parts[3])
    end
    return t, LC, err
end

function write_dat(f, t, LC, err)
    function formatLine(t,LC,err)
        preSpace = " "^3
        t = @sprintf("%.7g", t)
        if length(t) < 7 
            if occursin('.', t)
                t = t * "0"^(8 - length(t))
            else
                t = t * "." * "0"^(7 - length(t))
            end
        end
        LC = @sprintf("%.7E", LC)
        if length(LC) < 8 
            LC = LC * "0"^(8 - length(LC))
        end
        tLCSpace = length(LC) == 9 ? " "^6 : " "^7
        err = @sprintf("%.7E", err)
        LCerrSpace = " "^6
        return preSpace * t * tLCSpace * LC * LCerrSpace * err
    end
    open(f, "w") do io 
        for i in 1:length(t)
            println(io, formatLine(t[i], LC[i], err[i]))
        end
    end
end

function getMEMSamples(saveDir="";nPerΨ=2^3,noiseLevel=5e-3,missingFrac=0.0)
    Cfull = load(saveDir*"trainTestC.jld2", "C")
    tC = load(saveDir*"trainTestC.jld2", "tC")
    t = tC[tC .>= 0.0]
    dir = "../STORM/MEMecho/kirkFakeMEMechoResults/"*saveDir
    Ψfiles = filter(f -> (contains(f, "Psi")), readdir(dir))
    X = zeros(Float32, length(t), 1, nPerΨ*length(Ψfiles))
    Y = zeros(Float32, length(t), 1, nPerΨ*length(Ψfiles))
    tags = zeros(nPerΨ*length(Ψfiles))
    keepMask = ones(Bool, nPerΨ*length(Ψfiles))
    for (i,Ψfile) in enumerate(Ψfiles)
        tΨ,Ψ,col3 = read_dat(dir*Ψfile)
        nanMask = isnan.(Ψ)
        tΨ = tΨ[.!nanMask]
        Ψ = Ψ[.!nanMask]
        if length(tΨ) < 2
            @warn "Not enough data in $Ψfile to generate samples"
            keepMask[(i-1)*nPerΨ+1:i*nPerΨ] .= false
        else
            tMask = tΨ .>= 0.0
            tΨ = tΨ[tMask]
            Ψ = Ψ[tMask]
            nt = length(t)
            if length(tΨ) < nt
                nOriginal = length(tΨ)
                Ψ = vcat(Ψ, [0.0 for i=nOriginal+1:nt])
            end
            extraBeginning = sum(tC .< 0)
            L = DSP.conv(Cfull, Ψ)[extraBeginning+1:extraBeginning+nt]./sqrt(length(Ψ)) #need to index by something else
            for j in 1:nPerΨ
                Ltmp = genErroneousLC(t, L, missingFrac; method="line", scatterErr=noiseLevel) #generate line light curve with noise
                X[:,1,(i-1)*nPerΨ+j] .= Float32.(Ltmp)
                Y[:,1,(i-1)*nPerΨ+j] .= Float32.(Ψ)
                tags[(i-1)*nPerΨ+j] = parse(Float64, split(Ψfile, "_")[2]) # extract ΨtMax from filename
            end
        end
    end
    return X[:,:,keepMask],Y[:,:,keepMask],tags[keepMask]
end

function visualizeMEM_MLpred(m; saveDir="", index=1, tOffset=100.,nBatch=5, nPerΨ=20, C = nothing, tC = nothing, t = nothing, noiseLevel=5e-3, missingFrac=0.0)
    dir = "../STORM/MEMecho/kirkFakeMEMechoResults/"*saveDir
    Cref = "../STORM/MEMecho/kirkFakeData/"*saveDir*"MEMecho_C.dat"
    Cfile = dir*filter(f -> (contains(f, "C")), readdir(dir))[index]
    Lfile = dir*filter(f -> (contains(f, "L")), readdir(dir))[index]
    Ψfile = dir*filter(f -> (contains(f, "Psi")), readdir(dir))[index]
    ΨtMax = parse(Float64,split(Ψfile, "_")[2])
    tΨ, Ψ, col3 = read_dat(Ψfile)
    tMask = tΨ .>= 0.0
    tΨ = tΨ[tMask]
    Ψ = Ψ[tMask]
    nt = length(t)
    if length(tΨ) < nt
        nOriginal = length(tΨ)
        tΨ = vcat(tΨ, [tΨ[end] + i*(tΨ[2] - tΨ[1]) for i=nOriginal+1:nt])
        Ψ = vcat(Ψ, [0.0 for i=nOriginal+1:nt])
        # ΨErr = vcat(ΨErr, [0.0 for i=nOriginal+1:nt])
    end
    X = zeros(Float32, nt, 1, nBatch) #time, channels, obs
    for i=1:nBatch
        extraBeginning = sum(tC .< 0) #should have length(t) -1 
        L = DSP.conv(C, Ψ)[extraBeginning:extraBeginning+nt+1]./sqrt(length(Ψ)) #need to index by something else
        L = genErroneousLC(t, L, missingFrac; method="line", scatterErr=noiseLevel) #generate line light curve with noise
        X[:,1,i] .= Float32.(L)
    end
    pred = m(X)
    Lσ= std(X, dims=3)
    Lμ = mean(X, dims=3)
    p=plot(tC,C,label="C",lw=1.5)
    p=plot!(t,vec(Lμ),label="L",lw=1.5,ribbon=vec(Lσ))
    p=plot!(t,Ψ./maximum(Ψ),label="Ψ",lw=1.5)
    Ψpredμ = mean(pred, dims=3)
    Ψpredσ = std(pred, dims=3)
    p=plot!(t,vec(Ψpredμ),label="Ψ (prediction)",lw=1.5,ribbon=vec(Ψpredσ))
    return p
end

function getMEMPredData(C,tC,Ψ,t,nPerΨ=16,noiseLevel=5e-3,missingFrac=0.0;bootStrap=false,nBootJack=0,jackknife=false,randMissing=false,useSavedC=false,CMissingFrac=0.0)
    saveCount = 1
    for i=1:nPerΨ
        if randMissing
            missingFrac = rand()*missingFrac+0.1
        end
        d = genData(C,tC,t,ν=0.0,noisy=true,noiseLevel=noiseLevel,Ψ=Ψ,missingFrac=missingFrac,dropMissing=true) 
        nanmask = isnan.(d.L)
        t_tmp = d.t[.!nanmask]
        L_tmp = d.L[.!nanmask]
        if useSavedC
            Cnanmask = (isnan.(C)) .| (tC .< 0.0) .| (rand(length(C)) .< CMissingFrac)
        else
            Cnanmask = nanmask
        end
        t_ctmp = useSavedC ? tC[.!Cnanmask] : d.t[.!Cnanmask]
        C_tmp = useSavedC ? C[.!Cnanmask] : d.C[.!Cnanmask]
        # Ctmp = C[tC.>=0.0]
        extent = maximum(C_tmp)-minimum(C_tmp)
        Cnoise= rand(Normal(0.0,extent*noiseLevel),length(t_ctmp))
        Ctmp = C_tmp.+Cnoise
        Cnoise = ones(length(t_ctmp)).*extent*noiseLevel
        extent = maximum(L_tmp)-minimum(L_tmp)
        Lnoise = ones(length(t_tmp)).*extent*noiseLevel
        if bootStrap
            for j in 1:nBoot
                Lnoise = ones(length(t_tmp)).*extent*noiseLevel
                Cnoise = ones(length(t_ctmp)).*extent*noiseLevel
                bootInds = sort(rand(1:length(t_tmp), length(t_tmp)))
                t_save = copy(t_tmp[unique(bootInds)])
                t_Csave = copy(t_ctmp[unique(bootInds)])
                L_save = copy(L_tmp[unique(bootInds)])
                C_save = copy(C_tmp[unique(bootInds)])
                for b in bootInds
                    if sum(bootInds .== b) > 1
                        Cnoise[b] /= sqrt(sum(bootInds .== b)) #decrease noise for duplicated points
                        Lnoise[b] /= sqrt(sum(bootInds .== b))
                    end
                end
                C_noise_save = Cnoise[unique(bootInds)]
                L_noise_save = Lnoise[unique(bootInds)]
                write_dat("MEMechoSamples/C_$saveCount.dat",t_Csave,C_save,C_noise_save)
                write_dat("MEMechoSamples/L_$saveCount.dat",t_save,L_save,L_noise_save)
                saveCount += 1
            end
        elseif jackknife
            for j in 1:nBootJack #should I randomly drop more than one point? i.e. if length(t_tmp) is large
                randomInd = rand(1:length(t_tmp))
                t_save = copy(t_tmp[setdiff(1:end, randomInd)]) #randomly drop a point
                L_save = copy(L_tmp[setdiff(1:end, randomInd)])
                C_save = copy(C_tmp[setdiff(1:end, randomInd)])
                t_Csave = copy(t_ctmp[setdiff(1:end, randomInd)])
                L_noise_save = copy(Lnoise[setdiff(1:end, randomInd)])
                C_noise_save = copy(Cnoise[setdiff(1:end, randomInd)])
                write_dat("MEMechoSamples/C_$saveCount.dat",t_Csave,C_save,C_noise_save)
                write_dat("MEMechoSamples/L_$saveCount.dat",t_save,L_save,L_noise_save)
                saveCount += 1
            end
        else
            write_dat("MEMechoSamples/L_$saveCount.dat",t_tmp,L_tmp,Lnoise)
            write_dat("MEMechoSamples/C_$saveCount.dat",t_ctmp,Ctmp,Cnoise)
            saveCount += 1
        end
        #run(`/home/kirk/Documents/research/Dexter/STORM/MEMecho/kirkScripts/run_MEMecho.sh -d $Cfile -e $Lfile -l 200 -u 200 -i 1000 -s MEMechoSamples/out_$saveCount`
    end
end

function prepFilesForDCNN(t; dir="MEMechoSamples/", n=10^4, case="normal") #need to restore shape of files to be compatible with DCNN architecture
    files = readdir(dir)
    Cfiles = filter(f -> (contains(f, "C_")), files)
    Lfiles = filter(f -> (contains(f, "L_")), files)
    for i in 1:n
        Cfile = dir*Cfiles[i]
        Lfile = dir*Lfiles[i]
        tC, C, CErr = read_dat(Cfile)
        tL, L, LErr = read_dat(Lfile)
        Linterp = LinearInterpolation(tL, L, extrapolation_bc=Line())
        Cinterp = LinearInterpolation(tC, C, extrapolation_bc=Line())
        CErr = LinearInterpolation(tC, CErr, extrapolation_bc=Line())
        LErr = LinearInterpolation(tL, LErr, extrapolation_bc=Line())
        CErr = CErr.(t)
        LErr = LErr.(t)
        C = Cinterp.(t) #sometimes ti not exactly the same as t
        L = Linterp.(t)
        write_dat(dir*"DCNN_$(case)_C_$(i).dat", t, C, CErr)
        write_dat(dir*"DCNN_$(case)_L_$(i).dat", t, L, LErr)
    end
end

function readMEMPredData(tData,nPerΨ=16)
    t,Ψi,_ = read_dat("MEMechoSamples/Psi_out_1")
    Ψ = zeros(length(t),nPerΨ)
    for i=1:nPerΨ
        t,Ψi,_ = read_dat("MEMechoSamples/Psi_out_$i")
        Ψ[:,i] .= Ψi 
    end
    mask = t.>=tData[1]
    p=mean(Ψ[mask,:],dims=2)
    σ=std(Ψ[mask,:],dims=2)
    p=vcat(p,zeros(length(tData)-sum(mask)))
    σ=vcat(σ,zeros(length(tData)-sum(mask)))
    return p,σ
end

#Ψ0=load("combined_example.jld2","Ψ"); t=load("combined_example.jld2","t")
function generateMEMUncertainty(t;nPerΨ=16,dir="MEMechoSamples/",lhood=1.0)
    Ψout = zeros(length(t),nPerΨ)
    for i=1:nPerΨ
        run(`/home/kirk/Documents/research/Dexter/STORM/MEMecho/kirkScripts/run_MEMecho.sh -d $(dir*"C_$i") -e $(dir*"L_$i") -l-10 -u 100 -i 1000 -t $lhood -s $(dir*"uncertainty_$i")`)
        ti,Ψ,_=read_dat(dir*"uncertainty_$(i)_Psi.dat")
        Ψinterp = LinearInterpolation(ti,Ψ,extrapolation_bc=Line())
        Ψ = Ψinterp.(t) #sometimes ti not exactly the same as t
        Ψ[t .> maximum(ti)] .= 0.0 #set extrapolated values to 0
        #Ψ = Ψ[ti .>= 0.0]
        #Ψ=vcat(Ψ,zeros(length(t)-length(Ψ))) #append zeros to end
        Ψout[:,i].=Ψ
    end
    μ = mean(Ψout,dims=2)
    lowers = zeros(length(t))
    uppers = zeros(length(t))
    for i in 1:length(t) #get standard deviation above and below mean at each time
        lowMask = Ψout[i,:] .< μ[i]
        highMask = Ψout[i,:] .> μ[i]
        if sum(lowMask) == 0
            lowers[i] = 0.0
        else
            lowers[i] = std(Ψout[i,lowMask]) 
        end
        if sum(highMask) == 0
            uppers[i] = 0.0
        else
            uppers[i] = std(Ψout[i,highMask])
        end
    end
    σ = std(Ψout,dims=2)
    #lowMask = Ψout[] -- do low and high for real
    return μ,σ,Ψout,lowers,uppers
end

function visualizeMEMechoResult(saveDir="";ΨtMax=nothing,index=1,tOffset=100.,choice)
    Cfile = ""
    Lfile = ""
    Ψfile = ""
    dir = "../STORM/MEMecho/kirkFakeMEMechoResults/"*saveDir
    if isnothing(ΨtMax)
        files = readdir(dir)
        Cfile = dir*filter(f -> (contains(f, "C")), files)[index]
        Lfile = dir*filter(f -> (contains(f, "L")), files)[index]
        Ψfile = dir*filter(f -> (contains(f, "Psi")), files)[index]
        ΨtMax = parse(Float64,split(Ψfile, "_")[2])
    else
        Lfile = dir*"out_$(ΨtMax)_L.dat"
        Ψfile = dir*"out_$(ΨtMax)_Psi.dat"
        Cfile = dir*"out_$(ΨtMax)_C.dat"
    end
    println("ΨtMax: $ΨtMax")
    tC, C, CErr = read_dat(Cfile)
    tC = tC .- tOffset
    tMask = tC .>= 0.0
    tC = tC[tMask] # only keep positive times
    C = C[tMask]
    CErr = CErr[tMask]
    tL, L, LErr = read_dat(Lfile)
    tL = tL .- tOffset
    tMask = tL .>= 0.0
    tL = tL[tMask] # only keep positive times
    L = L[tMask]
    LErr = LErr[tMask]
    tΨPred, ΨPred, col3 = read_dat(Ψfile)
    tMask = tΨPred .>= 0.0
    tΨPred = tΨPred[tMask]
    ΨPred = ΨPred[tMask]
    shift = 0.0
    if choice=="expΨ" #decayRate is actually where max height is for t*e^(-ct/decayrate), c controls how fast it decays
        Ψ = expModel(tΨPred,shift,ΨtMax)
    elseif choice=="useBLRDisk"
        m = BLR.DiskWindModel(3000.,100.,1.,75/180*π,
            nr=1024,nϕ=512,scale=:log,f1=0.0,f2=0.0,f3=0.0,f4=1.0,
            I=BLR.DiskWindIntensity,v=BLR.vCircularDisk,τ=5.0,reflect=false)
        
        M₀ = 1e8*2e30 #kg
        ΨtMax₀=29.775541764107576/4 #days, when using M₀ as above
        M = M₀*ΨtMax/ΨtMax₀ #multiply by scaling factor
        rs = 2*M*6.67e-11/9e16 #2GM/c^2
        rsDay = rs/3e8/3600/24 #days
        tCenters = tΨPred./rsDay #convert to rs
        Δt = tCenters[2]-tCenters[1]
        tEdges = vcat([tCenters[1]-Δt/2],tCenters.+Δt/2)
        Ψ = BLR.getΨt(m,tEdges)
        Ψ = Ψ./maximum(Ψ)
    elseif choice=="KDM"
        Ψ = KeithDiskModel(tΨPred,t₀=ΨtMax)
    end
    lNorm = maximum(abs.(L))
    CNorm = maximum(abs.(C))
    ΨNorm = maximum(abs.(Ψ))

    p1 = plot(tL, L./lNorm, label="L", lw=1.5)
    p1 = plot!(tC, C./CNorm, label="C", lw=1.5)
    p1 = plot!(tΨPred, Ψ./ΨNorm, label="Ψ", lw=2)

    p1 = plot!(tΨPred, ΨPred./maximum(ΨPred), label="Predicted Ψ", lw=2,c=7)
    xcorr_t = range(-maximum(tL), stop=maximum(tL), length=2*length(tL)-1)
    CCF = DSP.xcorr(L, C)
    p1 = plot!(xcorr_t, CCF./maximum(CCF), label="CCF", lw=1)
    dt = tΨPred[2] - tΨPred[1]
    τgiven = sum(Ψ.*tΨPred.*dt)/sum(Ψ.*dt)
    τpred = sum(ΨPred.*tΨPred.*dt)/sum(ΨPred.*dt)
    p1 = vline!(p1, [τgiven], label="true τ", c=3, ls=:dash, lw=1.5)
    p1 = vline!(p1, [τpred], label="predicted τ", c=7, ls=:dash, lw=1.5)
    p1 = plot!(legend=:outertop, legend_columns=3, xlims=(0, maximum(tL)), xlabel="time [days]", ylabel="normalized value")
    return p1
end
    