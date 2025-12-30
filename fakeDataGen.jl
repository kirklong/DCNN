using Distributions, DSP, Plots, FFTW, Interpolations, Optim, BroadLineRegions
# using GaussianProcesses #removed because of conflicts with Pigeons

@kwdef struct data
    ν::Float64 = 0.0
    τ::Float64
    Ψ::Array{Float64}
    C::Array{Float64}
    L::Array{Float64}
    t::Array{Float64}
end

@kwdef struct echoMap
    ν::Array{Float64}
    t::Array{Float64}
    Ψ::Matrix{Float64}
end

@kwdef struct data2D
    ν::Array{Float64}
    t::Array{Float64}
    Ψ::Matrix{Float64}
    C::Array{Float64}
    L::Matrix{Float64}
end

function δC(;t::Array{Float64}=collect(range(0,stop=100,length=1001)), tPulse=1.0)
    C = zeros(length(t))
    C[findfirst(t.>=tPulse)] = 1.0 #delta function at tPulse
    return C
end

function randWalk(;t::Array{Float64}=collect(range(0,stop=100,length=1001)), σ::Float64=0.1, maxFreq::Float64=1.0, pad::Int64=50)
    """
    Generate random walk
    parameters:
        t: time array (days)
        σ: standard deviation of random walk
    returns:
        random walk array
    """
    t0i = findfirst(t.==0.0) #index of t=0, which we want to normalize continuum value here to 0 (each measurement relative to first)
    w = zeros(length(t)+pad)

    for i in (t0i+1):length(t)+pad
        dir = rand(Bool) ? 1 : -1
        w[i] = w[i-1] + dir*rand(Normal(0,σ))
    end
    for i in (t0i-1):-1:1 #negative times
        dir = rand(Bool) ? 1 : -1
        w[i] = w[i+1] + dir*rand(Normal(0,σ))
    end
    ft = fft(w)
    freq = fftfreq(length(t)+pad)
    ft[abs.(freq).>maxFreq] .= 0.0 #set all frequencies above maxFreq to zero, smooths output
    w = real.(ifft(ft))[1:length(t)] #take inverse fft and remove padding (padding is to avoid edge effects)
    w = w.-w[t0i] #normalize continuum value at t=0 to 0

    rescale = maximum(abs.(w[t.>=0.0])) #rescale so that max/min is 1
    w = w./rescale
    return w
end

function DRW(;t::Array{Float64}=collect(range(0,stop=100,length=1001)), μ::Float64=0.0, τ::Float64=1.0, σ::Float64=0.1)
    """
    Generate damped random walk
    parameters:
        t: time array (days)
        μ: mean of random walk
        τ: characteristic time scale of random walk
        σ: standard deviation of random walk
    returns:
        damped random walk array corresponding to times t
    """
    SF∞ = √2*σ #eq 4 of https://iopscience.iop.org/article/10.1088/0004-637X/721/2/1014/pdf
    C = zeros(length(t)) .+ μ
    for i in 2:length(t)
        Δt = t[i] - t[i-1]
        #below from eq 5 of https://iopscience.iop.org/article/10.1088/0004-637X/721/2/1014/pdf
        E = exp(-Δt/τ)*C[i-1]+μ*(1-exp(-Δt/τ))
        V = 0.5*SF∞^2*(1-exp(-2*Δt/τ))
        C[i] = E + rand(Normal(0,sqrt(V)))
    end
    return C
end

function zeroPad(x::Array{Float64},n::Int)
    """
    Zero pad array x to length n
    """
    return vcat(x,zeros(n))
end

function genData2D(;ΨtMaxRange=1:0.1:10, ΨνMaxRange=0:0.5:3,t::Array{Float64}=collect(range(0,stop=100,length=1001)),ν::Array{Float64}=collect(range(-10,stop=10,length=11)))
    tC = vcat(-reverse(t[2:end]),t) # negative times for convolution
    C = randWalk(t=tC,maxFreq=1.0)
    dList = Array{data2D}(undef,length(ΨνMaxRange)*length(ΨtMaxRange))
    d = 1
    for ΨtMax in ΨtMaxRange
        for ΨνMax in ΨνMaxRange
            t_decayRate = rand(Uniform(0.1*ΨtMax,2*ΨtMax)) #decay rate of exponential in t direction, randomly decay between 0.1 (shorter) and 2 (longer) times ΨtMax
            ν_decayRate = ΨνMax != 0 ? rand(Uniform(0.1*ΨνMax,2*ΨνMax)) : rand(Uniform(0.01*maximum(ν),maximum(ν))) #decay rate of exponential in ν direction, randomly decay between 0.1 (smaller) and 2 (larger) times ΨνMax
            Ψ = zeros(length(ν),length(t))
            L = zeros(length(ν),length(t))
            for i in 1:length(ν)
                for j in 1:length(t)
                    if (t[j] >= ΨtMax) && (abs(ν[i]) >= ΨνMax) 
                        Ψ[i,j] = exp(-(t[j]-ΨtMax)/t_decayRate)*exp(-(abs(ν[i])-ΨνMax)/ν_decayRate)
                    end
                end
            end
            Ψ = Ψ./maximum(Ψ) #normalize to max of 1
            for i in 1:length(ν)
                L[i,:] = DSP.conv(C,Ψ[i,:])[length(t)-1:end-length(t)]./sqrt(length(Ψ)) #convolve with continuum and remove negative/extra times
            end
            dList[d] = data2D(ν=ν,t=t,Ψ=Ψ,C=C[length(t):end],L=L)
            d += 1
        end
    end
    return dList
end

function genData2D_synthLC(;t::Array{Float64},ν::Array{Float64},C::Array{Float64},ΨtMaxRange=1:0.1:10,ΨνMaxRange=0:0.5e3:3e3)
    dList = Array{data2D}(undef,length(ΨνMaxRange)*length(ΨtMaxRange))
    d = 1
    for ΨtMax in ΨtMaxRange
        for ΨνMax in ΨνMaxRange
            t_decayRate = rand(Uniform(0.1*ΨtMax,2*ΨtMax)) #decay rate of exponential in t direction, randomly decay between 0.1 (shorter) and 2 (longer) times ΨtMax
            ν_decayRate = ΨνMax != 0 ? rand(Uniform(0.1*ΨνMax,2*ΨνMax)) : rand(Uniform(0.01*maximum(ν),maximum(ν))) #decay rate of exponential in ν direction, randomly decay between 0.1 (smaller) and 2 (larger) times ΨνMax
            Ψ = zeros(length(ν),length(t))
            L = zeros(length(ν),length(t))
            for i in 1:length(ν)
                for j in 1:length(t)
                    if (t[j] >= ΨtMax) && (abs(ν[i]) >= ΨνMax) 
                        Ψ[i,j] = exp(-(t[j]-ΨtMax)/t_decayRate)*exp(-(abs(ν[i])-ΨνMax)/ν_decayRate)
                    end
                end
            end
            Ψ = Ψ./maximum(Ψ) #normalize to max of 1
            Ψ1D = [sum(Ψ[:,i]) for i in 1:length(t)]
            τ = BLR.trapint(t,Ψ1D.*t)/BLR.trapint(t,Ψ1D)
            if 2*τ > t[end] || isnan(τ)
                for i in 1:length(ν)
                    L[i,:] .= NaN
                end
                dList[d] = data2D(ν=ν,t=t,Ψ=Ψ,C=C[length(t):end],L=L)
                d += 1
                continue
            else
                for i in 1:length(ν)
                    synthLC = BLR.syntheticLC(t,C,Ψ[i,:],tStart=0.0)
                    L[i,:] .= synthLC
                end
                dList[d] = data2D(ν=ν,t=t,Ψ=Ψ,C=C[length(t):end],L=L)
                d += 1
            end
        end
    end
    return dList
end

function expModel(t,shift,ΨtMax,normalize=true)
    m = t.*exp.(-(t./ΨtMax))
    shiftInd = findfirst(t.>=shift)
    m = circshift(m,shiftInd-1)
    m[t.<=shift].=0.0
    if normalize
        m = m./maximum(m)
    end
    return m
end

function recoverExpΨParams(t,Ψ)
    f(x,Ψ,t) = sum(abs.(Ψ .- expModel(t,x[1],x[2])))
    x0 = [t[findfirst(Ψ.>0.0)], t[findmax(Ψ)[2]]] #initial guess for [shift, ΨtMax]
    res = optimize(x -> f(x, Ψ, t), x0)
    return res
end

function KeithDiskModel(t;λ=1.0,b=3/4,λ₀=1.0,t₀=1.0,Ψ₀=1.0,normalize=true)
    x = (λ₀/λ).*(t./t₀).^b
    W = @. x^2/2/(cosh(x)-1)
    Ψ = Ψ₀/t₀*(λ/λ₀)^2 .*(t./t₀).^(3*b-2).*W 
    Ψ[isnan.(Ψ)] .= 0.0 #set NaN values to 0
    if normalize
        Ψ = Ψ./maximum(Ψ)
    end
    return Ψ
end

function genData(C,tC,t::Array{Float64}=collect(range(0,stop=100,length=1001));expΨ::Bool=true, ΨtMax::Float64=1.0, 
    shift::Float64=0.0, ν::Float64=0.0,noisy::Bool=true,noiseLevel::Float64=0.1, useBLRDisk::Bool=false, KDM::Bool=false,
    missingFrac::Float64=0.0, gaussian::Bool=false, useBLRClouds::Bool=false, ellipse::Bool=false, Ψ=nothing, dropMissing::Bool=false)
    """
    Generate fake data for testing model
    parameters:
        expΨ: if true, Ψ is exponential decay model (where it decays to zero at rougly 1/2 maximum time), else random, but only to half the maximum time (days)
        ΨtMax: time that corresponds to maximum Ψ (days)
        sineC: if true, continuum is a sine wave with period of PC, else random
        PC: period of continuum (days) -- future work: add this as characteristic random walk time scale from edge to edge 
        t: time array (days)
        ν: frequency (for velocity delay map construct multiple datasets at many ν)
    returns:
        data object
    """

    # generate transfer function
    if isnothing(Ψ)
        stop = false
        Ψ = zeros(length(t))
        tryCount = 0
        while !stop
            if expΨ #decayRate is actually where max height is for t*e^(-ct/decayrate), c controls how fast it decays
                Ψ = expModel(t,shift,ΨtMax)
            elseif useBLRDisk #update this to randomly pick parameters
                iMin = 5.; iMax = 85.
                r̄Min = 500.; r̄Max = 2000.
                MfacMin = 0.1; MfacMax = 2.0
                rFacMin = 2.; rFacMax = 100.
                f1Min = 0.0; f1Max = 1.0
                f2Min = 0.0; f2Max = 1.0
                f3Min = 0.0; f3Max = 1.0
                f4Min = 0.0; f4Max = 1.0
                SαMin = 0.0; SαMax = 2.0
                scale = 1.; cenShift = 0.; ϕMin = -3.14; ϕMax = 3.14 #fixed parameters
                chooser̄ = rand(Bool)
                if chooser̄
                    r̄ = rand(Uniform(r̄Min,r̄Max)) #mean radius of BLR (in code units of rs)
                    M = ΨtMax/r̄*2.7e25*3600*24/2/6.67e-11 #mass in kg, from scaling relation in code units
                    Mfac = M/(1e8*2e30) #mass factor
                else
                    Mfac = rand(Uniform(MfacMin,MfacMax)) #mass factor
                    M = 1e8*2e30*Mfac
                    rs = 2*M*6.67e-11/9e16 #2GM/c^2
                    r̄ = ΨtMax/rs*3e8*3600*24 #mean radius of BLR (in code units of rs)
                end
                #r̄ = rand(Uniform(r̄Min,r̄Max)) #mean radius of BLR (in code units of rs)
                #Mfac = rand(Uniform(MfacMin,MfacMax)) #mass factor
                rFac = rand(Uniform(rFacMin,rFacMax)) #radius factor
                f1 = rand(Uniform(f1Min,f1Max))
                f2 = rand(Uniform(f2Min,f2Max))
                f3 = rand(Uniform(f3Min,f3Max))
                f4 = rand(Uniform(f4Min,f4Max))
                Sα = rand(Uniform(SαMin,SαMax))
                i = rand(Uniform(iMin,iMax)) #inclination angle in degrees

                m = BLR.DiskWindModel(r̄,rFac,1.0,i/180*π,
                    nr=1024,nϕ=512,scale=:log,f1=f1,f2=f2,f3=f3,f4=f4,
                    I=BLR.DiskWindIntensity,v=BLR.vCircularDisk,τ=5.0,reflect=false)
                
                M = 1e8*2e30*Mfac
                rs = 2*M*6.67e-11/9e16 #2GM/c^2
                rsDay = rs/3e8/3600/24 #days
                tCenters = t./rsDay #convert to rs
                Δt = tCenters[2]-tCenters[1]
                tEdges = vcat([tCenters[1]-Δt/2],tCenters.+Δt/2)
                Ψ = BLR.getΨt(m,tEdges)
            elseif KDM
                Ψ = KeithDiskModel(t,t₀=ΨtMax)
            elseif gaussian
                #make guassian Ψ centered at Ψtmax with some random width
                width = minimum([rand(Uniform(1,10)),ΨtMax]) #1-10 days
                Ψ = exp.(-0.5*((t.-ΨtMax)./width).^2)
            elseif useBLRClouds
                iMin = 5.; iMax = 85.
                r̄Min = 500.; r̄Max = 2000.
                MfacMin = 0.1; MfacMax = 2.0
                FMin = 0.; FMax = 1.
                κMin = -0.5; κMax = 0.5
                ξMin = 0.; ξMax = 1.
                γMin = 0.; γMax = 5.
                fEllipseMin = 0.; fEllipseMax = 1.
                fFlowMin = 0.; fFlowMax = 1.
                θₑMin = 0.; θₑMax = 90.
                σρᵣMin = 0.1; σρᵣMax = 1.0
                σρcMin = 0.1; σρcMax = 1.0
                σθᵣMin = 0.1; σθᵣMax = 1.0
                σθcMin = 0.1; σθcMax = 1.0
                σₜMin = 0.1; σₜMax = 1.0
                βMin = 0.; βMax = 3.
                θₒMin = 0.; θₒMax = 90.

                nClouds = 500_000

                i = rand(Uniform(iMin,iMax))
                chooser̄ = rand(Bool)
                if chooser̄
                    r̄ = rand(Uniform(r̄Min,r̄Max)) #mean radius of BLR (in code units of rs)
                    M = ΨtMax/r̄*2.7e25*3600*24/2/6.67e-11 #mass in kg, from scaling relation in code units
                    Mfac = M/(1e8*2e30) #mass factor
                else
                    Mfac = rand(Uniform(MfacMin,MfacMax)) #mass factor
                    M = 1e8*2e30*Mfac
                    rs = 2*M*6.67e-11/9e16 #2GM/c^2
                    r̄ = ΨtMax/rs*3e8*3600*24 #mean radius of BLR (in code units of rs)
                end
                θₒ = rand(Uniform(θₒMin,θₒMax))
                β = rand(Uniform(βMin,βMax))
                F = rand(Uniform(FMin,FMax))
                κ = rand(Uniform(κMin,κMax))
                ξ = rand(Uniform(ξMin,ξMax))
                γ = rand(Uniform(γMin,γMax))
                fEllipse = rand(Uniform(fEllipseMin,fEllipseMax))
                fFlow = rand(Uniform(fFlowMin,fFlowMax))
                θₑ = rand(Uniform(θₑMin,θₑMax))
                σρᵣ = rand(Uniform(σρᵣMin,σρᵣMax))
                σρc = rand(Uniform(σρcMin,σρcMax))
                σΘᵣ = rand(Uniform(σθᵣMin,σθᵣMax))
                σΘc = rand(Uniform(σθcMin,σθcMax))
                σₜ = rand(Uniform(σₜMin,σₜMax))

                m = BLR.cloudModel(nClouds; I=BLR.cloudIntensity, v=BLR.vCloudTurbulentEllipticalFlow,
                i=i/180*π, θₒ=θₒ/180*π, β=β, F=F, μ=r̄, κ=κ, ξ=ξ, γ=γ,
                fEllipse=fEllipse, fFlow=fFlow, θₑ=θₑ/180*π, σρᵣ=σρᵣ, σρc=σρc, σΘᵣ=σΘᵣ, σΘc=σΘc, σₜ=σₜ,
                τ=0.0)

                M = 1e8*2e30*Mfac
                rs = 2*M*6.67e-11/9e16 #2GM/c^2
                rsDay = rs/3e8/3600/24 #days
                tCenters = t./rsDay #convert to rs
                Δt = tCenters[2]-tCenters[1]
                tEdges = vcat([tCenters[1]-Δt/2],tCenters.+Δt/2)
                Ψ = BLR.getΨt(m,tEdges)
            elseif ellipse
                width = rand(Uniform(1,10)) #1-10 days
                Ψ = exp.(-0.5*((t.-ΨtMax)./width).^2)
                secondΨtMax = rand(Uniform(ΨtMax,ΨtMax*3)) #second peak between 1 and 3 times ΨtMax
                secondScale = rand(Uniform(0.0,5*exp(-(secondΨtMax/ΨtMax)))) #scale of second peak between 0.0 and 1.0
                Ψ += exp.(-0.5*((t.-secondΨtMax)./width).^2)*secondScale
                Ψ = Ψ./maximum(Ψ)
            else
                @error "Ψ not modified from zeros default, received expΨ = $expΨ, KDM = $KDM, useBLRDisk = $useBLRDisk, useBLRClouds = $useBLRClouds, gaussian = $gaussian, ellipse = $ellipse"
                exit()
            end
            Ψ[isnan.(Ψ)] .= 0.0
            if sum(Ψ) != 0.0
                stop = true
            else
                tryCount += 1
                println("Ψ = all 0 after attempt $tryCount")
            end
        end
    end
    τ = sum(Ψ.*t)/sum(Ψ) #weighted average time delay
    Ψ = Ψ./maximum(Ψ) #normalize to max of 1

    L = DSP.conv(C,Ψ)[length(t):end-(length(Ψ)-1)]/sqrt(length(Ψ)) # convolve continuum LC with transfer function and remove negative/extra times, apply sqrt(n) normalization: https://dsp.stackexchange.com/questions/72397/normalization-factor-in-the-convolution-theorem

    L = genErroneousLC(t,L,missingFrac;method="line",scatterErr=noiseLevel,dropMissing=dropMissing)

    data(ν=ν, τ=τ, Ψ=Ψ, C=C[length(t):end], L=L, t=t)
end

function genData2D_NEW(C,tC,t::Array{Float64}=collect(range(0,stop=400,length=401));expΨ::Bool=true, ΨtMax::Float64=1.0, 
    shift::Float64=0.0, ν::Float64=0.0,noisy::Bool=true,noiseLevel::Float64=5e-3, useBLRDisk::Bool=false, KDM::Bool=false,
    missingFrac::Float64=0.0, gaussian::Bool=false, useBLRClouds::Bool=false, ellipse::Bool=false, Ψ=nothing,vCenters=[0.0])
    """
    Generate fake data for testing model
    parameters:
        expΨ: if true, Ψ is exponential decay model (where it decays to zero at rougly 1/2 maximum time), else random, but only to half the maximum time (days)
        ΨtMax: time that corresponds to maximum Ψ (days)
        sineC: if true, continuum is a sine wave with period of PC, else random
        PC: period of continuum (days) -- future work: add this as characteristic random walk time scale from edge to edge 
        t: time array (days)
        ν: frequency (for velocity delay map construct multiple datasets at many ν)
    returns:
        data object
    """
    nv = length(vCenters); nt = length(t)
    L = zeros(nv, nt)
    # generate transfer function
    if isnothing(Ψ)
        stop = false
        Ψ = zeros(nv, nt)
        tryCount = 0
        while !stop
            ΨvMax = rand(Uniform(minimum(abs.(vCenters))/10,3*maximum(abs.(vCenters))/4)) #randomly pick a velocity scale for exponential decay in velocity direction
            if expΨ #decayRate is actually where max height is for t*e^(-ct/decayrate), c controls how fast it decays
                for i in 1:nv
                    Ψ[i, :] = expModel(t, shift, ΨtMax).*exp(-abs(vCenters[i])/ΨvMax) #make Ψ decay with velocity as well
                end
            elseif useBLRDisk #update this to randomly pick parameters
                iMin = 5.; iMax = 85.
                r̄Min = 500.; r̄Max = 1200.
                MfacMin = 0.1; MfacMax = 2.0
                rFacMin = 2.; rFacMax = 100.
                f1Min = 0.0; f1Max = 1.0
                f2Min = 0.0; f2Max = 1.0
                f3Min = 0.0; f3Max = 1.0
                f4Min = 0.0; f4Max = 1.0
                SαMin = 0.0; SαMax = 2.0
                scale = 1.; cenShift = 0.; ϕMin = -3.14; ϕMax = 3.14 #fixed parameters
                chooser̄ = rand(Bool)
                if chooser̄
                    r̄ = rand(Uniform(r̄Min,r̄Max)) #mean radius of BLR (in code units of rs)
                    M = ΨtMax/r̄*2.7e25*3600*24/2/6.67e-11 #mass in kg, from scaling relation in code units
                    Mfac = M/(1e8*2e30) #mass factor
                else
                    Mfac = rand(Uniform(MfacMin,MfacMax)) #mass factor
                    M = 1e8*2e30*Mfac
                    rs = 2*M*6.67e-11/9e16 #2GM/c^2
                    r̄ = ΨtMax/rs*3e8*3600*24 #mean radius of BLR (in code units of rs)
                end
                rFac = rand(Uniform(rFacMin,rFacMax)) #radius factor
                f1 = rand(Uniform(f1Min,f1Max))
                f2 = rand(Uniform(f2Min,f2Max))
                f3 = rand(Uniform(f3Min,f3Max))
                f4 = rand(Uniform(f4Min,f4Max))
                Sα = rand(Uniform(SαMin,SαMax))
                i = rand(Uniform(iMin,iMax)) #inclination angle in degrees

                m = BLR.DiskWindModel(r̄,rFac,Sα,i/180*π,scale=:log,f1=f1,f2=f2,f3=f3,f4=f4,
                    I=BLR.DiskWindIntensity,v=BLR.vCircularDisk,τ=5.0,reflect=false)
                
                M = 1e8*2e30*Mfac
                rs = 2*M*6.67e-11/9e16 #2GM/c^2
                rsDay = rs/3e8/3600/24 #days
                tCenters = t./rsDay #convert to rs
                Δt = tCenters[2]-tCenters[1]
                tEdges = vcat([tCenters[1]-Δt/2],tCenters.+Δt/2)
                vBinWidths = diff(vCenters)
                vEdges = [vCenters[1]-vBinWidths[1]/2; vCenters[1:end-1].+vBinWidths/2; vCenters[end]+vBinWidths[end]/2]./3e5 #code units in v/c
                Ψ = BLR.getΨ(m,vEdges,tEdges)
            elseif KDM #problem -- not really actually, it's just that the λ range implied by BLR velocities is small? so not much wavelength dependence
                for (i,v) in enumerate(vCenters)
                    # Δλ/λ = v/c = v′ -> λ = Δλ/v′ = (λ - λ₀)/v′ -> λ = λ₀/(1 - v′) #if v′ != 0.0
                    v = v/3e5 #convert to v/c
                    Ψ[i, :] = KeithDiskModel(t,t₀=ΨtMax,λ=1/(1 - v),normalize=false)
                end
            elseif gaussian 
                #make guassian Ψ centered at Ψtmax with some random width
                tWidth = minimum([rand(Uniform(1,10)),ΨtMax]) #1-10 days
                vWidth = minimum([rand(Uniform(0.1e3,2e3)),ΨvMax]) #0.1-2 1e3 km/s
                for (i,vi) in enumerate(vCenters)
                    for (j,tj) in enumerate(t)
                        Ψ[i, j] = exp(-0.5*((tj-ΨtMax)/tWidth)^2) * exp(-0.5*((vi-ΨvMax)/vWidth)^2)
                    end
                end
            elseif useBLRClouds
                iMin = 5.; iMax = 85.
                r̄Min = 500.; r̄Max = 2000.
                MfacMin = 0.1; MfacMax = 2.0
                FMin = 0.; FMax = 1.
                κMin = -0.5; κMax = 0.5
                ξMin = 0.; ξMax = 1.
                γMin = 0.; γMax = 5.
                fEllipseMin = 0.; fEllipseMax = 1.
                fFlowMin = 0.; fFlowMax = 1.
                θₑMin = 0.; θₑMax = 90.
                σρᵣMin = 0.1; σρᵣMax = 1.0
                σρcMin = 0.1; σρcMax = 1.0
                σθᵣMin = 0.1; σθᵣMax = 1.0
                σθcMin = 0.1; σθcMax = 1.0
                σₜMin = 0.1; σₜMax = 1.0
                βMin = 0.; βMax = 3.
                θₒMin = 0.; θₒMax = 90.

                nClouds = 500_000

                i = rand(Uniform(iMin,iMax))
                chooser̄ = rand(Bool)
                if chooser̄
                    r̄ = rand(Uniform(r̄Min,r̄Max)) #mean radius of BLR (in code units of rs)
                    M = ΨtMax/r̄*2.7e25*3600*24/2/6.67e-11 #mass in kg, from scaling relation in code units
                    Mfac = M/(1e8*2e30) #mass factor
                else
                    Mfac = rand(Uniform(MfacMin,MfacMax)) #mass factor
                    M = 1e8*2e30*Mfac
                    rs = 2*M*6.67e-11/9e16 #2GM/c^2
                    r̄ = ΨtMax/rs*3e8*3600*24 #mean radius of BLR (in code units of rs)
                end
                θₒ = rand(Uniform(θₒMin,θₒMax))
                β = rand(Uniform(βMin,βMax))
                F = rand(Uniform(FMin,FMax))
                κ = rand(Uniform(κMin,κMax))
                ξ = rand(Uniform(ξMin,ξMax))
                γ = rand(Uniform(γMin,γMax))
                fEllipse = rand(Uniform(fEllipseMin,fEllipseMax))
                fFlow = rand(Uniform(fFlowMin,fFlowMax))
                θₑ = rand(Uniform(θₑMin,θₑMax))
                σρᵣ = rand(Uniform(σρᵣMin,σρᵣMax))
                σρc = rand(Uniform(σρcMin,σρcMax))
                σΘᵣ = rand(Uniform(σθᵣMin,σθᵣMax))
                σΘc = rand(Uniform(σθcMin,σθcMax))
                σₜ = rand(Uniform(σₜMin,σₜMax))

                m = BLR.cloudModel(nClouds; I=BLR.cloudIntensity, v=BLR.vCloudTurbulentEllipticalFlow,
                i=i/180*π, θₒ=θₒ/180*π, β=β, F=F, μ=r̄, κ=κ, ξ=ξ, γ=γ,
                fEllipse=fEllipse, fFlow=fFlow, θₑ=θₑ/180*π, σρᵣ=σρᵣ, σρc=σρc, σΘᵣ=σΘᵣ, σΘc=σΘc, σₜ=σₜ,
                τ=0.0)

                M = 1e8*2e30*Mfac
                rs = 2*M*6.67e-11/9e16 #2GM/c^2
                rsDay = rs/3e8/3600/24 #days
                tCenters = t./rsDay #convert to rs
                Δt = tCenters[2]-tCenters[1]
                tEdges = vcat([tCenters[1]-Δt/2],tCenters.+Δt/2)
                vBinWidths = diff(vCenters)
                vEdges = [vCenters[1]-vBinWidths[1]/2; vCenters[1:end-1].+vBinWidths/2; vCenters[end]+vBinWidths[end]/2]./3e5 #code units in v/c
                Ψ = BLR.getΨ(m,vEdges,tEdges)
            elseif ellipse
                tWidth = minimum([rand(Uniform(1,10)),ΨtMax]) #1-10 days
                vWidth = rand(Uniform(minimum(abs.(vCenters))/10, maximum(abs.(vCenters))/2)) #velocity width
                ellipseWidth = rand(Uniform(0.5, 5)) #1-5 cells
                tCenter = rand(Uniform(ΨtMax, ΨtMax*2)) #center of ellipse in time
                for (i,vi) in enumerate(vCenters)
                    for (j,tj) in enumerate(t)
                        # Create ellipse using standard ellipse equation: (x-h)²/a² + (y-k)²/b² = 1
                        t_term = ((tj - tCenter) / tWidth)^2
                        v_term = ((vi) / vWidth)^2
                        # set intensity based on how far from the ellipse we are (should = 1)
                        distance_factor = abs(sqrt(t_term + v_term) - 1)
                        Ψ[i, j] = exp(-distance_factor^2 / ellipseWidth)*exp(-sqrt(t_term)/ΨtMax) # Gaussian-like falloff from ellipse and in time
                    end
                end
            else
                @error "Ψ not modified from zeros default, received expΨ = $expΨ, KDM = $KDM, useBLRDisk = $useBLRDisk, useBLRClouds = $useBLRClouds, gaussian = $gaussian, ellipse = $ellipse"
                exit()
            end
            Ψ[isnan.(Ψ)] .= 0.0
            if sum(Ψ) != 0.0
                stop = true
            else
                tryCount += 1
                println("Ψ = all 0 after attempt $tryCount")
            end
        end
    end
    τ = sum(sum(Ψ,dims=1).*t)/sum(Ψ) #weighted average time delay
    Ψ = Ψ./maximum(Ψ) #normalize to max of 1
    for i in 1:nv
        Ltmp = DSP.conv(C,Ψ[i,:])[length(t):end-(length(Ψ[i,:])-1)] / sqrt(length(Ψ[i,:])) # convolve continuum LC with transfer function and remove negative/extra times, apply sqrt(n) normalization: https://dsp.stackexchange.com/questions/72397/normalization-factor-in-the-convolution-theorem
        Ltmp = genErroneousLC(t,Ltmp,missingFrac;method="line",scatterErr=noiseLevel)
        L[i,:] = Ltmp
    end
    data2D(ν=vCenters, Ψ=Ψ, C=C[length(t):end], L=L, t=t)
end

#### note to self: actually do want to generate data all in -1:1 space, so that training 
#### is consistent / doesn't penalize weirdly. But should think about how things are rescaled and keep track of it.
#### training is working again though right now (1/22)

function getParamFx(min,max)
    if min == max
        return fixed() = min
    else
        return uniform() = rand(Uniform(min,max))
    end
end

function genErroneousLC(t,LC,dropFrac;method="line",scatterErr=0.0,dropMissing=false)
    """
    Generate an "erroneous" light curve with "missing" data by dropping a fraction of the data and filling in the gaps and adding noise to data
    parameters:
        t: "perfect" time array
        LC: "perfect" light curve array
        dropFrac: fraction of data to replace
        method: method to fill in gaps (line, random, GP)
            - optional, defaults to linear interpolation (line)
            - random: random value between min and max of data
            - GP: Gaussian process interpolation
        scatterErr: standard deviation of Gaussian noise to add to data, passed as fraction of data extent 
            - optional, defaults to 0.0
    returns:
        LC: erroneous light curve array sampled at t
    """
    nDrop = floor(Int,dropFrac*length(LC))
    if scatterErr > 0.0
        extent = maximum(LC)-minimum(LC)
        LC = LC .+ rand(Normal(0,scatterErr*extent),length(LC))
    end
    if nDrop > 0
        dropInds = randperm(length(LC))[1:nDrop]
        sortedDrop = sort(dropInds)
        if dropMissing
            for i in dropInds
                LC[i] = NaN
            end
        else
            if method == "line"
                currentInd = 1; nextInd = 2
                while nextInd < length(sortedDrop)
                    if sortedDrop[nextInd] == sortedDrop[nextInd-1]+1
                        nextInd += 1
                    else
                        m = (LC[sortedDrop[nextInd]]-LC[sortedDrop[currentInd]])/(t[sortedDrop[nextInd]]-t[sortedDrop[currentInd]])
                        b = LC[sortedDrop[currentInd]]-m*t[sortedDrop[currentInd]]
                        LC[sortedDrop[currentInd]:sortedDrop[nextInd]] .= m.*t[sortedDrop[currentInd]:sortedDrop[nextInd]].+b
                        currentInd = deepcopy(nextInd)
                        nextInd += 1
                    end
                end            
            elseif method == "random"
                for i in dropInds
                    LC[i] = rand(Uniform(minimum(LC),maximum(LC)))
                end
            elseif method == "GP"
                kernel = SE(0.,0.) #length scale 1 day, sigma 1
                good_mask = [i for i in 1:length(t) if i ∉ dropInds]
                t_obs = [t[i] for i in good_mask]
                LC_obs = [LC[i] for i in good_mask]
                gp = GP(t_obs,LC_obs,MeanZero(),kernel)
                try
                    optimize!(gp)
                catch
                    println("GP optimization failed, using default parameters")
                end
                currentInd = 1; nextInd = 2
                while nextInd < length(sortedDrop)
                    if sortedDrop[nextInd] == sortedDrop[nextInd-1]+1
                        nextInd += 1
                    else
                        μ,σ² = predict_y(gp,t[sortedDrop[currentInd]:sortedDrop[nextInd]])
                        LC[sortedDrop[currentInd]:sortedDrop[nextInd]] .= μ
                        currentInd = deepcopy(nextInd)
                        nextInd += 1
                    end
                end
            else
                error("method must be line, random, or GP")
            end
        end
    end
    return LC
end

function genDiskWindSamples(n,C,tC,vCenters,progress=true,useConv=true;kwargs...)
    iMin = 5.; iMax = 85.
    r̄Min = 200.; r̄Max = 2000.
    MfacMin = 0.01; MfacMax = 10.
    rFacMin = 2.; rFacMax = 100.
    f1Min = 0.0; f1Max = 1.0
    f2Min = 0.0; f2Max = 1.0
    f3Min = 0.0; f3Max = 1.0
    f4Min = 0.0; f4Max = 1.0
    SαMin = 0.0; SαMax = 2.0
    scale = 1.; cenShift = 0.; ϕMin = -3.14; ϕMax = 3.14 #fixed parameters
    nv = 11; nT = 501; dList = nothing

    method="line"
    dropFrac=0.0
    scatterErr=0.0
    gridSearch = false

    if :iMin in keys(kwargs)
        iMin = kwargs[:iMin]
    end
    if :iMax in keys(kwargs)
        iMax = kwargs[:iMax]
    end
    if :r̄Min in keys(kwargs)
        r̄Min = kwargs[:r̄Min]
    end
    if :r̄Max in keys(kwargs)
        r̄Max = kwargs[:r̄Max]
    end
    if :MfacMin in keys(kwargs)
        MfacMin = kwargs[:MfacMin]
    end
    if :MfacMax in keys(kwargs)
        MfacMax = kwargs[:MfacMax]
    end
    if :rFacMin in keys(kwargs)
        rFacMin = kwargs[:rFacMin]
    end
    if :rFacMax in keys(kwargs)
        rFacMax = kwargs[:rFacMax]
    end
    if :f1Min in keys(kwargs)
        f1Min = kwargs[:f1Min]
    end
    if :f1Max in keys(kwargs)
        f1Max = kwargs[:f1Max]
    end
    if :f2Min in keys(kwargs)
        f2Min = kwargs[:f2Min]
    end
    if :f2Max in keys(kwargs)
        f2Max = kwargs[:f2Max]
    end
    if :f3Min in keys(kwargs)
        f3Min = kwargs[:f3Min]
    end
    if :f3Max in keys(kwargs)
        f3Max = kwargs[:f3Max]
    end
    if :f4Min in keys(kwargs)
        f4Min = kwargs[:f4Min]
    end
    if :f4Max in keys(kwargs)
        f4Max = kwargs[:f4Max]
    end
    if :SαMin in keys(kwargs)
        SαMin = kwargs[:SαMin]
    end
    if :SαMax in keys(kwargs)
        SαMax = kwargs[:SαMax]
    end
    if :scale in keys(kwargs)
        scale = kwargs[:scale]
    end
    if :cenShift in keys(kwargs)
        cenShift = kwargs[:cenShift]
    end
    if :ϕMin in keys(kwargs)
        ϕMin = kwargs[:ϕMin]
    end
    if :ϕMax in keys(kwargs)
        ϕMax = kwargs[:ϕMax]
    end
    if :nv in keys(kwargs)
        nv = kwargs[:nv]
    end
    if :nT in keys(kwargs)
        nT = kwargs[:nT]
    end
    if :method in keys(kwargs)
        method = kwargs[:method]
    end
    if :dropFrac in keys(kwargs)
        dropFrac = kwargs[:dropFrac]
    end
    if :scatterErr in keys(kwargs)
        scatterErr = kwargs[:scatterErr]
    end

    if gridSearch #not working right now, ignore
        println("grid search WIP")
        # indpParams = Dict{String,Array{Float64}}()
        # i = nothing; r̄ = nothing; Mfac = nothing; rFac = nothing; f1 = nothing; f2 = nothing; f3 = nothing; f4 = nothing; Sα = nothing
        # if iMin == iMax 
        #     i = iMin
        # else
        #     indpParams["i"] = [iMin,iMax]
        # end 
        # if r̄Min == r̄Max 
        #     r̄ = r̄Min
        # else
        #     indpParams["r̄"] = [r̄Min,r̄Max]
        # end
        # if MfacMin == MfacMax 
        #     Mfac = MfacMin
        # else
        #     indpParams["Mfac"] = [MfacMin,MfacMax]
        # end
        # if rFacMin == rFacMax 
        #     rFac = rFacMin
        # else
        #     indpParams["rFac"] = [rFacMin,rFacMax]
        # end
        # if f1Min == f1Max 
        #     f1 = f1Min
        # else
        #     indpParams["f1"] = [f1Min,f1Max]
        # end
        # if f2Min == f2Max 
        #     f2 = f2Min
        # else
        #     indpParams["f2"] = [f2Min,f2Max]
        # end
        # if f3Min == f3Max 
        #     f3 = f3Min
        # else
        #     indpParams["f3"] = [f3Min,f3Max]
        # end
        # if f4Min == f4Max 
        #     f4 = f4Min
        # else
        #     indpParams["f4"] = [f4Min,f4Max]
        # end
        # if SαMin == SαMax 
        #     Sα = SαMin
        # else
        #     indpParams["Sα"] = [SαMin,SαMax]
        # end

        # names = collect(keys(indpParams))
        # nIndp = length(names)
        # nSamples = floor(Int,n^(1/nIndp))
        # for (i,name) in enumerate(names)
        #     if i < nIndp
        #         for ()
        #fix this later...
    else
        params = Dict{String,Any}()

        params["i"] = getParamFx(iMin,iMax)
        params["r̄"] = getParamFx(r̄Min,r̄Max)
        params["Mfac"] = getParamFx(MfacMin,MfacMax)
        params["rFac"] = getParamFx(rFacMin,rFacMax)
        params["f1"] = getParamFx(f1Min,f1Max)
        params["f2"] = getParamFx(f2Min,f2Max)
        params["f3"] = getParamFx(f3Min,f3Max)
        params["f4"] = getParamFx(f4Min,f4Max)
        params["Sα"] = getParamFx(SαMin,SαMax)
        params["scale"] = scale
        params["cenShift"] = cenShift
        params["ϕMin"] = ϕMin
        params["ϕMax"] = ϕMax
        dList = Array{data2D}(undef,n)
        strLen = 0; tStart = time()
        t = deepcopy(tC[tC.>=0.0])
        for iter=1:n 
            m = nothing
            r̄ = params["r̄"](); rFac = params["rFac"](); Sα = params["Sα"](); i = params["i"]()/180*π
            f1 = params["f1"](); f2 = params["f2"](); f3 = params["f3"](); f4 = params["f4"]()
            Mfac = params["Mfac"]()
            if params["ϕMin"] == -3.14 && params["ϕMax"] == 3.14
                m = BLR.DiskWindModel(r̄,rFac,Sα,i,
                    f1=f1,f2=f2,f3=f3,f4=f4,I=BLR.DiskWindIntensity,τ=5.,reflect=false)
            else
                m = BLR.DiskWindModel(r̄,rFac,Sα,i,
                    f1=f1,f2=f2,f3=f3,f4=f4,I=BLR.IϕDiskWindMask,τ=5.,
                    ϕMin=params["ϕMin"],ϕMax=params["ϕMax"],reflect=false)
            end
            rs = 2*Mfac*3e8*2e30*6.67e-11/9e16
            rsDay = rs/3e8/24/3600
            tMax = tC[end]/rsDay #convert days to rs
            tEdges = collect(range(0,stop=tMax,length=length(tC)+1))
            vBinWidths = diff(vCenters)
            vEdges = [vCenters[1]-vBinWidths[1]/2; vCenters[1:end-1].+vBinWidths/2; vCenters[end]+vBinWidths[end]/2]./3e5 #code units in v/c
            tCenters = @. (tEdges[1:end-1] + tEdges[2:end])/2
            Ψ = BLR.getΨ(m,vEdges,tEdges)
            Ψ = Ψ./maximum(Ψ)
            L = zeros(length(vCenters),length(tCenters))
            τ = r̄*rsDay # in rs, *rsDay for days -- PROBLEM: sometimes this is too large for data set? check manually later
            if 2*τ > tC[end]
                # println("τ too large (τ = $τ) -- r̄ = $r̄, Mfac = $Mfac")
                for i in 1:length(vCenters)
                    L[i,:] .= NaN
                end
                if progress
                    strLen = progress!(iter,n,tStart,strLen)
                end
            else
                for i in 1:length(vCenters)
                    if useConv
                        synthcLC = DSP.conv(C,Ψ[i,:])[length(t):end-(length(Ψ[i,:])-1)]/sqrt(length(Ψ[i,:])) 
                    else
                        Ψt_interp = LinearInterpolation(tCenters.*rsDay,Ψ[i,:],extrapolation_bc=Line())
                        synthLC = BLR.syntheticLC(tC,C,Ψt_interp.(tC),tStart=0.0)
                    end
                    synthLC = genErroneousLC(t,synthLC,dropFrac,method=method,scatterErr=scatterErr)
                    L[i,:] .= synthLC
                    #interpolate synthetic LC to real measured LC data point times with GP interpolation? -- for HST visits have same t continuum and line (at least to day)
                end
            end
            dList[iter] = data2D(ν=vCenters,t=tCenters.*rsDay,Ψ=Ψ,C=C,L=L)
            if progress
                strLen = progress!(iter,n,tStart,strLen)
            end
        end
    end
    return dList
end

function genCloudSamples(n,C,tC,vCenters,progress=true,dropFrac=0.0,scatterErr=0.0,useConv=true;kwargs...)
    iMin = 5.; iMax = 85.
    MfacMin = 0.01; MfacMax = 10.
    θₒMin = 0.; θₒMax = 90.
    βMin = 0.; βMax = 3.
    FMin = 0.; FMax = 1.
    μMin = 200.; μMax = 2000.
    κMin = -0.5; κMax = 0.5
    ξMin = 0.; ξMax = 1.
    γMin = 0.; γMax = 5.
    fEllipseMin = 0.; fEllipseMax = 1.
    fFlowMin = 0.; fFlowMax = 1.
    θₑMin = 0.; θₑMax = 90.
    σρᵣMin = 0.1; σρᵣMax = 1.0
    σρcMin = 0.1; σρcMax = 1.0
    σθᵣMin = 0.1; σθᵣMax = 1.0
    σθcMin = 0.1; σθcMax = 1.0
    σₜMin = 0.1; σₜMax = 1.0
    scale = 1.; cenShift = 0.
    ϕMin = -3.14; ϕMax = 3.14 #fixed parameters
    nClouds = 100_000

    method="line"
    dropFrac=0.0
    scatterErr=0.0
    gridSearch = false
    
    if :iMin in keys(kwargs)
        iMin = kwargs[:iMin]
    end
    if :iMax in keys(kwargs)
        iMax = kwargs[:iMax]
    end
    if :MfacMin in keys(kwargs)
        MfacMin = kwargs[:MfacMin]
    end
    if :MfacMax in keys(kwargs)
        MfacMax = kwargs[:MfacMax]
    end
    if :θₒMin in keys(kwargs)
        θₒMin = kwargs[:θₒMin]
    end
    if :θₒMax in keys(kwargs)
        θₒMax = kwargs[:θₒMax]
    end
    if :βMin in keys(kwargs)
        βMin = kwargs[:βMin]
    end
    if :βMax in keys(kwargs)
        βMax = kwargs[:βMax]
    end
    if :FMin in keys(kwargs)
        FMin = kwargs[:FMin]
    end
    if :FMax in keys(kwargs)
        FMax = kwargs[:FMax]
    end
    if :μMin in keys(kwargs)
        μMin = kwargs[:μMin]
    end
    if :μMax in keys(kwargs)
        μMax = kwargs[:μMax]
    end
    if :κMin in keys(kwargs)
        κMin = kwargs[:κMin]
    end
    if :κMax in keys(kwargs)
        κMax = kwargs[:κMax]
    end
    if :ξMin in keys(kwargs)
        ξMin = kwargs[:ξMin]
    end
    if :ξMax in keys(kwargs)
        ξMax = kwargs[:ξMax]
    end
    if :γMin in keys(kwargs)
        γMin = kwargs[:γMin]
    end
    if :γMax in keys(kwargs)
        γMax = kwargs[:γMax]
    end
    if :fEllipseMin in keys(kwargs)
        fEllipseMin = kwargs[:fEllipseMin]
    end
    if :fEllipseMax in keys(kwargs)
        fEllipseMax = kwargs[:fEllipseMax]
    end
    if :fFlowMin in keys(kwargs)
        fFlowMin = kwargs[:fFlowMin]
    end
    if :fFlowMax in keys(kwargs)
        fFlowMax = kwargs[:fFlowMax]
    end
    if :θₑMin in keys(kwargs)
        θₑMin = kwargs[:θₑMin]
    end
    if :θₑMax in keys(kwargs)
        θₑMax = kwargs[:θₑMax]
    end
    if :σρᵣMin in keys(kwargs)
        σρᵣMin = kwargs[:σρᵣMin]
    end
    if :σρᵣMax in keys(kwargs)
        σρᵣMax = kwargs[:σρᵣMax]
    end
    if :σρcMin in keys(kwargs)
        σρcMin = kwargs[:σρcMin]
    end
    if :σρcMax in keys(kwargs)
        σρcMax = kwargs[:σρcMax]
    end
    if :σθᵣMin in keys(kwargs)
        σθᵣMin = kwargs[:σθᵣMin]
    end
    if :σθᵣMax in keys(kwargs)
        σθᵣMax = kwargs[:σθᵣMax]
    end
    if :σθcMin in keys(kwargs)
        σθcMin = kwargs[:σθcMin]
    end
    if :σθcMax in keys(kwargs)
        σθcMax = kwargs[:σθcMax]
    end
    if :σₜMin in keys(kwargs)
        σₜMin = kwargs[:σₜMin]
    end
    if :σₜMax in keys(kwargs)
        σₜMax = kwargs[:σₜMax]
    end
    if :scale in keys(kwargs)
        scale = kwargs[:scale]
    end
    if :cenShift in keys(kwargs)
        cenShift = kwargs[:cenShift]
    end
    if :ϕMin in keys(kwargs)
        ϕMin = kwargs[:ϕMin]
    end
    if :ϕMax in keys(kwargs)
        ϕMax = kwargs[:ϕMax]
    end
    if :nClouds in keys(kwargs)
        nClouds = kwargs[:nClouds]
    end
    if :method in keys(kwargs)
        method = kwargs[:method]
    end
    if :dropFrac in keys(kwargs)
        dropFrac = kwargs[:dropFrac]
    end
    if :scatterErr in keys(kwargs)
        scatterErr = kwargs[:scatterErr]
    end
    
    gridSearch = false
    dList = Array{data2D}(undef,n)
    strLen = 0; tStart = time()

    params = Dict{String,Any}()
    params["i"] = getParamFx(iMin,iMax)
    params["Mfac"] = getParamFx(MfacMin,MfacMax)
    params["θₒ"] = getParamFx(θₒMin,θₒMax)
    params["β"] = getParamFx(βMin,βMax)
    params["F"] = getParamFx(FMin,FMax)
    params["μ"] = getParamFx(μMin,μMax)
    params["κ"] = getParamFx(κMin,κMax)
    params["ξ"] = getParamFx(ξMin,ξMax)
    params["γ"] = getParamFx(γMin,γMax)
    params["fEllipse"] = getParamFx(fEllipseMin,fEllipseMax)
    params["fFlow"] = getParamFx(fFlowMin,fFlowMax)
    params["θₑ"] = getParamFx(θₑMin,θₑMax)
    params["σρᵣ"] = getParamFx(σρᵣMin,σρᵣMax)
    params["σρc"] = getParamFx(σρcMin,σρcMax)
    params["σθᵣ"] = getParamFx(σθᵣMin,σθᵣMax)
    params["σθc"] = getParamFx(σθcMin,σθcMax)
    params["σₜ"] = getParamFx(σₜMin,σₜMax)
    params["scale"] = scale
    params["cenShift"] = cenShift
    params["ϕMin"] = ϕMin
    params["ϕMax"] = ϕMax
    t = deepcopy(tC[tC.>=0.0])
    for iter=1:n
        m = nothing
        i = params["i"]()/180*π; Mfac = params["Mfac"](); θₒ = params["θₒ"]()/180*π; β = params["β"](); F = params["F"]()
        μ = params["μ"](); κ = params["κ"](); ξ = params["ξ"](); γ = params["γ"]()
        fEllipse = params["fEllipse"](); fFlow = params["fFlow"](); θₑ = params["θₑ"]()/180*π
        σρᵣ = params["σρᵣ"](); σρc = params["σρc"](); σΘᵣ = params["σθᵣ"](); σΘc = params["σθc"](); σₜ = params["σₜ"]()
        ϕMin = params["ϕMin"]; ϕMax = params["ϕMax"]
        if ϕMin == -3.14 && ϕMax == 3.14
            m = BLR.cloudModel(nClouds; I=BLR.cloudIntensity, v=BLR.vCloudTurbulentEllipticalFlow,
            i=i, θₒ=θₒ, β=β, F=F, μ=μ, κ=κ, ξ=ξ, γ=γ, 
            fEllipse=fEllipse, fFlow=fFlow, θₑ=θₑ, σρᵣ=σρᵣ, σρc=σρc, σΘᵣ=σΘᵣ, σΘc=σΘc, σₜ=σₜ,
            τ=0.0)
        else
            m = BLR.cloudModel(nClouds; I=BLR.IϕCloudMask, v=BLR.vCloudTurbulentEllipticalFlow,
            i=i, θₒ=θₒ, β=β, F=F, μ=μ, κ=κ, ξ=ξ, γ=γ, 
            fEllipse=fEllipse, fFlow=fFlow, θₑ=θₑ, σρᵣ=σρᵣ, σρc=σρc, σΘᵣ=σΘᵣ, σΘc=σΘc, σₜ=σₜ,ϕMin=ϕMin, ϕMax=ϕMax,
            τ=0.0)
        end
        rs = 2*Mfac*3e8*2e30*6.67e-11/9e16
        rsDay = rs/3e8/24/3600
        tMax = tC[end]/rsDay #convert days to rs
        tEdges = collect(range(0,stop=tMax,length=length(tC)+1))
        vBinWidths = diff(vCenters)
        vEdges = [vCenters[1]-vBinWidths[1]/2; vCenters[1:end-1].+vBinWidths/2; vCenters[end]+vBinWidths[end]/2]./3e5 #code units in v/c
        tCenters = @. (tEdges[1:end-1] + tEdges[2:end])/2
        Ψ = BLR.getΨ(m,vEdges,tEdges)
        Ψ = Ψ./maximum(Ψ)
        L = zeros(length(vCenters),length(tCenters))
        τ = μ*rsDay # in rs, *rsDay for days -- PROBLEM: sometimes this is too large for data set? check manually later
        if 2*τ > tC[end]
            for i in 1:length(vCenters)
                L[i,:] .= NaN
            end
            if progress
                strLen = progress!(iter,n,tStart,strLen)
            end
        else
            for i in 1:length(vCenters)
                if useConv
                    synthcLC = DSP.conv(C,Ψ[i,:])[length(t):end-(length(Ψ[i,:])-1)]/sqrt(length(Ψ[i,:])) 
                else
                    Ψt_interp = LinearInterpolation(tCenters.*rsDay,Ψ[i,:],extrapolation_bc=Line())
                    synthLC = BLR.syntheticLC(tC,C,Ψt_interp.(tC),tStart=0.0)
                end
                synthLC = genErroneousLC(t,synthLC,dropFrac,method=method,scatterErr=scatterErr)
                L[i,:] .= synthLC
                #interpolate synthetic LC to real measured LC data point times with GP interpolation? -- for HST visits have same t continuum and line (at least to day)
            end
        end
        if progress
            strLen = progress!(iter,n,tStart,strLen)
        end
        dList[iter] = data2D(ν=vCenters,t=tC,Ψ=Ψ,C=C,L=L)
    end
    return dList
end    

function fakeModelDataGen(contLC,tC,vData;n::Int=100,DiskWindFrac=1.,trainFrac=0.5,shuffle=true,kwargs...)
    nDiskWind = Int(n*DiskWindFrac)
    nClouds = n - nDiskWind
    diskWindSamples = genDiskWindSamples(nDiskWind,contLC,tC,vData;kwargs...)
    cloudSamples = genCloudSamples(nClouds,contLC,tC,vData;kwargs...)
    dList = vcat(diskWindSamples,cloudSamples)
    return dList
end

function visualizeTest(d::data)
    p = plot(d.t,d.C,label="C")
    p = plot!(d.t,(d.L),label="LC")
    p = plot!(d.t,(d.Ψ),label="Ψ",ls=:dash)
    CCF = xcorr(d.L,d.C)
    p = plot!(range(-maximum(d.t),stop=maximum(d.t),length=2*length(d.t)-1),CCF./maximum(CCF),label="CCF",ls=:dash)
    p = vline!(p,[d.τ],label="τ",ls=:dash,c=:crimson)
    p = plot!(xlims=(0.0,maximum(d.t)),ylims=(-1,1))
    return p
end

function progress!(i,n,tStart,strLen;nOut=100)
    if i == 1
        modelString = "["*" "^25 *"]"*" XX% complete -- "*"estimated 00:00:00 left (0.0s/it)"
        strLen = length(modelString)
    end
    tNew = time()
    if i % ceil(Int,n/nOut) == 0 #output every 1% of iterations
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
        timeStr = parse(Int,hours) > 24 ? "> $(hours)" : "$(hours):$(minutes):$(seconds)"
        str = progressBar * " " * percentStr * " — estimated " * timeStr * " left ($(round(i/(tNew-tStart),sigdigits=2))it/s)" 
        print("\r"*" "^strLen*"\r")
        printstyled(split(progressBar,">")[1],color=:cyan)
        printstyled(">",color=:magenta,blink=true,bold=true)
        printstyled(split(progressBar,">")[2],color=:cyan)
        print(" ")
        printstyled("$(percent)%",color=:magenta,bold=true)
        print(" complete — estimated ")
        printstyled(timeStr,color=:red,bold=true)
        print(" left (")
        printstyled("$(round(i/(time()-tStart),sigdigits=3))",color=:red,bold=true)
        print(" it/s)")
        flush(stdout)
        strLen = length(str)
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
        printstyled("($(round(i/(tNew-tStart),sigdigits=2))",color=:red,bold=true)
        print(" it/s)")
        flush(stdout)
        println("\n")
    end
    return strLen
end

