module SparseHamiltonianFlows

    using Base: StatusActive
    using LinearAlgebra, Distributions, Statistics, Flux, ProgressMeter, Logging
    using Zygote: Buffer, ignore, forwarddiff
    include("train.jl")

    export Args, sparse_flows, sampler, sb_refresh, adaptive_refresh, sample_from_adaptive_refresh, sample_from_appended_flow, appended_flow, est_elbo_trained_flow

    mutable struct Args
        # data
        # d: dimension of latent parameter
        # N: number of observations
        # xs: data matrix (N rows)
        # inds: indices of the points in the coreset,
        #       if not provided, a uniform
        #       subsample will be drawn 
        #       upon initialization
        # sub_xs: points in the coreset (according to inds)

        d::Int64 
        N::Int64 
        xs::Matrix{Float64} 
        inds::Union{Vector{Int64}, Nothing}
        sub_xs::Union{Matrix{Float64}, Nothing}
        
        # hyper params
        # M: size of coreset (<= N)
        # elbo_size: number of samples (of latent parameters) 
        #            used to estimate the ELBO
        # number_of_refresh: # quasi-refreshment
        # K: # leapfrog steps between quasi-refreshment
        # lf_n: total number of leapfrog steps (number_of_refresh * K)
        # iter: number of optimization iterations 
        #       (of coreset weights, leapfrog step sizes, and quasi-refreshment)
        # cond: Boolean indicating whether quasi-refreshment is
        #       conditional or marginal
        # n_subsample_elbo: number of observations used to estimate the ELBO
        # save: Boolean indicating whether to store the parameter values
        #       throughout optimization iterations
        # S_init: sample size used to obtain initial quasi-refreshment parameters
        # S_final: deprecated, not used

        M::Int64
        elbo_size::Int64
        number_of_refresh::Int64
        K::Int64
        lf_n::Int64
        iter::Int64
        cond::Bool
        n_subsample_elbo::Int64
        save::Bool
        S_init::Int64
        S_final::Int64
        
        # optimizer: Flux optimizer
        optimizer
        
        # sampling and likelihood functions
        log_prior::Function # prior distribution (in log space) 
        logq0::Function # initial distribution of flow (in log space)
        logp_lik::Function # sum over all individual log likelihoods
        sample_q0::Function # function to sample from initial distribution
        ???potential_by_hand::Function # gradient of the logged unnormalized target wrt latent parameters
        
        # for storing components of the ELBO 
        # and log determinants of quasi-refreshment
        # for debugging purposes
        logpbar
        logqbar
        log_det
    end

    function sparse_flows(a::Args, ??_unc::Vector{Float64})
        # @info "error checking"
        error_checking(a, ??_unc)

        # @info "sample coreset basis if not provided"
        if isnothing(a.inds)
            a.inds = sort(sample(1:a.N, a.M, replace = false))
            a.sub_xs = a.xs[a.inds, :]
        end

        # @info "initialize weights using uniform weights"
        w_unc = log.(a.N/a.M * ones(a.M))

        # @info "initial pass"
        a.cond = false
        r_states = sb_refresh(a, ??_unc, w_unc, a.S_init)

        # @info "extract component mean and variance"
        ??ps = zeros(a.number_of_refresh, a.d)
        log??p = zeros(a.number_of_refresh, a.d)
        for i in 1:a.number_of_refresh
            _, ??p, _, _, ??p, _ = get!(r_states[i], "key", 0)
            ??ps[i,:] = ??p
            log??p[i,:] = log.(sqrt.(diag(??p)))
        end

        # @info "initialize vectors storing step sizes and weights"
        if a.save
            ??_unc_hist = zeros(a.iter+1, a.d)
            w_unc_hist = zeros(a.iter+1, a.M)
            ??ps_hist = zeros(a.iter+1, a.d * a.number_of_refresh)
            log??p_hist = zeros(a.iter+1, a.d * a.number_of_refresh)
            ??_unc_hist[1,:] = ??_unc
            w_unc_hist[1,:] = w_unc
            ??ps_hist[1,:] = vec(??ps)
            log??p_hist[1,:] = vec(log??p)
        else
            ??_unc_hist = zeros(1, a.d)
            w_unc_hist = zeros(1, a.M)
            ??ps_hist = zeros(1, a.d * a.number_of_refresh)
            log??p_hist = zeros(1, a.d * a.number_of_refresh)
        end

        # @info "prepare loss"
        ??_u = copy(??_unc)
        w_u = copy(w_unc)
        ??ps_param = copy(vec(??ps))
        log??p_param = copy(vec(log??p))

        ps = Flux.params(??_u, w_u, ??ps_param, log??p_param)
        loss =() -> begin
            obj = est_negative_elbo(a, ??_u, w_u, ??ps_param, log??p_param)
            return obj
        end

        # @info "training"
        ls_hist, time_hist = sparse_flow_trainT!(a, loss, ps, ??_unc_hist, w_unc_hist, ??ps_hist, log??p_hist)
        ??_unc_final = ??_unc_hist[end,:]
        w_unc_final = w_unc_hist[end,:]
        ??ps_final = ??ps_hist[end,:]
        log??p_final = log??p_hist[end,:]

        # @info "prepare r_states"
        ??ps_final_mat = reshape(??ps_final, (a.number_of_refresh, a.d))
        log??p_final_mat = reshape(log??p_final, (a.number_of_refresh, a.d))
        r_states = []
        for i in 1:a.number_of_refresh
            push!(r_states, IdDict())
        end
        for i in 1:a.number_of_refresh
            ??p = ??ps_final_mat[i,:]
            ??p = diagm((exp.(log??p_final_mat[i,:])).^2)
            get!(r_states[i], "key")do
                (nothing, ??p, nothing, nothing, ??p, nothing)
            end
        end

        return ??_unc_hist, w_unc_hist, ??ps_hist, log??p_hist, ls_hist, time_hist, r_states
    end

    function error_checking(a::Args, ??_unc::Vector{Float64})
        if a.K * a.number_of_refresh != a.lf_n
            throw(ErrorException("number of total leap frogs must be the product of K and number_of_refresh."))
        elseif a.N != size(a.xs,1)
            throw(ErrorException("number of observations do not match the field N."))
        elseif size(??_unc,1) != a.d
            throw(ErrorException("dimension of step size ?? does not match the field d."))
        elseif a.n_subsample_elbo > a.N
            throw(ErrorException("cannot subsample more than the total number of observations."))
        elseif sum(isnothing.([a.inds, a.sub_xs])) == 1
            throw(ErrorException("one of inds and sub_xs is not defined."))
        elseif !isnothing(a.inds)
            if !(size(a.inds,1) == a.M && size(a.sub_xs,1) == a.M)
                throw(ErrorException("size of coreset does not match M"))
            end
        end
    end

    function oneleapfrog(z::Vector{Float64}, p::Vector{Float64}, ??_unc::Vector{Float64}, ???U::Function, w_unc::Vector{Float64})
        ?? = exp.(??_unc)
        w = exp.(w_unc)
        p = p .+ (?? ./ 2) .* ???U(z, w)
        z = z .+ ?? .* p
        p = p .+ (?? ./ 2) .* ???U(z, w)
        return z, p
    end

    function oneleapfrog!(z::SubArray{Float64}, p::SubArray{Float64}, ??_unc::Vector{Float64}, ???U::Function, w_unc::Vector{Float64})
        ?? = exp.(??_unc)
        w = exp.(w_unc)
        p .+= (?? ./ 2) .* ???U(z, w)
        z .+= ?? .* p
        p .+= (?? ./ 2) .* ???U(z, w)
    end

    function compute_mv(zs_ref::Matrix{Float64}, ps_ref::Matrix{Float64})
        ??z = vec(mean(zs_ref, dims=1))
        ??p = vec(mean(ps_ref, dims=1))
        ??pz = cov(ps_ref, zs_ref)
        ??z = cov(zs_ref, zs_ref)
        ??p = cov(ps_ref, ps_ref)
        return ??z, ??p, ??pz, ??z, ??p
    end

    function cov_by_hand(dat1::Matrix{Float64}, dat2::Matrix{Float64})
        m1 = mean(dat1, dims=1)
        m2 = mean(dat2, dims=1)
        d = length(m1)
        n = size(dat1,1)
        ret = zeros(d,d)
        for i in 1:n
            ret = ret + transpose(dat1[i,:]' .- m1) * (dat2[i,:]' .- m2)
        end
        ret = ret / (n-1)
        return ret
    end

    function sb_refresh(a::Args, ??_unc::Vector{Float64}, w_unc::Vector{Float64}, sample_size::Int64)
        # prep gradients
        ???U = (zz, ww) -> a.???potential_by_hand(a.sub_xs, zz, ww)

        # samples used to estimate refresh parameters
        zs_ref = a.sample_q0(sample_size)
        ps_ref = randn(sample_size, a.d)

        # prepare dictionaries holding parameters
        r_states = []
        for i in 1:a.number_of_refresh
            push!(r_states, IdDict())
        end

        prog_bar = ProgressMeter.Progress(a.lf_n, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        for n in 1:a.lf_n
            for i in 1:sample_size
                oneleapfrog!(@view(zs_ref[i,:]), @view(ps_ref[i,:]), ??_unc, ???U, w_unc)
            end
            if n % a.K == 0
                ??z, ??p, ??pz, ??z, ??p = compute_mv(zs_ref, ps_ref)
                get!(r_states[Int(n / a.K)], "key")do
                    (??z, ??p, ??pz, ??z, ??p, [sample_size])
                end
                batch_refresh!(a, ??z, ??p, ??pz, ??z, ??p, zs_ref, ps_ref)
            end
            ProgressMeter.next!(prog_bar)
        end

        return r_states
    end

    function batch_refresh!(a::Args, ??z::Vector{Float64}, ??p::Vector{Float64}, ??pz::Matrix{Float64}, ??z::Matrix{Float64}, ??p::Matrix{Float64}, zs_ref::Matrix{Float64}, ps_ref::Matrix{Float64})
        L_inv, ??_mean, _ = compute_transformation(??pz, ??z, ??p, a.cond)
        s = size(zs_ref,1)
        if a.cond
            for i in 1:s
                @view(ps_ref[i,:]) .= L_inv * (@view(ps_ref[i,:]) .- (??p .+ ??_mean * (@view(zs_ref[i,:]) .- ??z)))
            end
        else
            for i in 1:s
                @view(ps_ref[i,:]) .= L_inv * (@view(ps_ref[i,:]) .- ??p)
            end
        end
    end

    function est_negative_elbo(a::Args, ??_u::Vector{Float64}, w_u::Vector{Float64}, ??ps_param::Vector{Float64}, log??p_param::Vector{Float64})
        elbo = 0.
        for i in 1:a.elbo_size
            # init z, p and refreshing sample
            z_0 = zeros(a.d)
            ignore() do
                z_0 = a.sample_q0(1)
            end
            p_0 = randn(a.d)

            z_last, p_last, log_determinant = flow(a, z_0, p_0, ??_u, w_u, ??ps_param, log??p_param)

            elbo_i = single_elbo(a, z_0, z_last, p_0, p_last, log_determinant)
            elbo = elbo + (1. / a.elbo_size) * elbo_i
        end
        return -elbo
    end

    function flow(a::Args, z_0::Vector{Float64}, p_0::Vector{Float64}, ??_u::Vector{Float64}, w_u::Vector{Float64}, ??ps_param::Vector{Float64}, log??p_param::Vector{Float64})
        ???U = (zz, ww) -> a.???potential_by_hand(a.sub_xs, zz, ww)
        log_determinant = 0.
        ??ps = reshape(??ps_param, (a.number_of_refresh, a.d))
        log??p = reshape(log??p_param, (a.number_of_refresh, a.d))
        for n in 1:a.lf_n
            z_0, p_0 = oneleapfrog(z_0, p_0, ??_u, ???U, w_u)
            if n % a.K == 0
                r = Int(n / a.K)
                ??p = ??ps[r,:]
                ??_inv_sqrt = diagm(vec(1. ./ exp.(log??p[r,:])))
                p_0 = ??_inv_sqrt * (p_0 .- ??p)
                log_det_new = -sum(log??p[r,:])
                log_determinant = log_determinant + log_det_new
            end
        end
        return z_0, p_0, log_determinant
    end

    function single_elbo(a::Args, z_0::Vector{Float64}, z_last::Vector{Float64}, p_0::Vector{Float64}, p_last::Vector{Float64}, log_determinant::Float64)
        logp = zz -> log_joint_density_for_elbo(a, zz)
        logp_bar = logp(z_last) - 0.5 * (p_last' * p_last)
        logq_bar = a.logq0(z_0) - 0.5 * (p_0' * p_0) - log_determinant
        ignore() do
            push!(a.logpbar, logp_bar)
            push!(a.log_det, log_determinant)
            push!(a.logqbar, a.logq0(z_0) - 0.5 * (p_0' * p_0))
        end
        return logp_bar - logq_bar
    end

    function log_joint_density_for_elbo(a::Args, z::Vector{Float64})
        dx = size(a.xs,2)
        sub_xs_elbo = zeros(a.n_subsample_elbo, dx)
        ignore() do
            inds_elbo = sort(sample(1:a.N, a.n_subsample_elbo, replace = false))
            sub_xs_elbo = a.xs[inds_elbo, :]
        end
        ??s = a.log_prior(z) + a.logp_lik(z, sub_xs_elbo) * (a.N / a.n_subsample_elbo)
        return ??s
    end

    function compute_transformation(??pz, ??z, ??p, cond)
        d = size(??p,1)
        if cond
            ?? = ??p .- (??pz * (??z \ transpose(??pz)))
            ?? = 0.5 .* (?? .+ transpose(??))
            chol = cholesky(??)
            L = chol.L
            L_inv = L \ I(d)
            log_determinant = -sum(log.(diag(L)))
            ??_mean = ??pz * (??z \ I(d))
        else
            ??p = 0.5 .* (??p + transpose(??p))
            chol = cholesky(??p)
            L = chol.L
            L_inv = L \ I(d)
            log_determinant = -sum(log.(diag(L)))
            ??_mean = nothing
        end
        return L_inv, ??_mean, log_determinant
    end

    function sampler(a::Args, num_sample::Int64, ??_unc_final::Vector{Float64}, w_unc_final::Vector{Float64}, r_states::Vector{Any}, zs, ps)
        if isnothing(zs) && isnothing(ps)
            zs = a.sample_q0(num_sample)
            ps = randn(num_sample, a.d)
        end
        zs, ps, log_det = trained_flow(a, zs, ps, ??_unc_final, w_unc_final, r_states)
        return zs, ps, log_det
    end

    function trained_flow(a::Args, zs::Matrix{Float64}, ps::Matrix{Float64}, ??_unc_final::Vector{Float64}, w_unc_final::Vector{Float64}, r_states::Vector{Any}; i = nothing)
        ???U = (zz, ww) -> a.???potential_by_hand(a.sub_xs, zz, ww)
        num_sample = size(zs,1)
        log_det = 0.
        leapfrog_n = isnothing(i) ? a.lf_n : i
        for n in 1:leapfrog_n
            for i in 1:num_sample
                oneleapfrog!(@view(zs[i,:]), @view(ps[i,:]), ??_unc_final, ???U, w_unc_final)
            end

            if n % a.K == 0
                r = Int(n / a.K)
                r_state = r_states[r]
                ??z, ??p, ??pz, ??z, ??p, _ = get!(r_state, "key", 0)
                ??_inv_sqrt, ??_mean, log_det_new = compute_transformation(??pz, ??z, ??p, a.cond)
                log_det = log_det + log_det_new
                if a.cond
                    for i in 1:num_sample
                        @view(ps[i,:]) .= ??_inv_sqrt * (@view(ps[i,:]) .- (??p .+ ??_mean * (@view(zs[i,:]) .- ??z)))
                    end
                else
                    for i in 1:num_sample
                        @view(ps[i,:]) .= ??_inv_sqrt * (@view(ps[i,:]) .- ??p)
                    end
                end
            end
        end

        return zs, ps, log_det
    end

    function est_elbo_trained_flow(a, ??_unc, w_unc, r_states, sample_size_for_metric_computation)
        elbos = zeros(a.lf_n)
        z_0_begin = a.sample_q0(sample_size_for_metric_computation)
        p_0_begin = randn(sample_size_for_metric_computation, a.d)
        z_0_original = copy(z_0_begin)
        p_0_original = copy(p_0_begin)

        prog_bar = ProgressMeter.Progress(a.lf_n, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        for i in 1:a.lf_n
            zs_0 = copy(z_0_begin)
            ps_0 = copy(p_0_begin)
            zs_last, ps_last, log_determinant = trained_flow(a, zs_0, ps_0, ??_unc, w_unc, r_states; i=i)
            for j in 1:sample_size_for_metric_computation
                elbos[i] += single_elbo(a, z_0_original[j,:], zs_last[j,:], p_0_original[j,:], ps_last[j,:], log_determinant)
            end
            elbos[i] /= sample_size_for_metric_computation
            ProgressMeter.next!(prog_bar)
        end
        return elbos
    end

    function kl_gaussian(dat::Matrix{Float64})
        d = size(dat,2)
        ??q = vec(mean(dat, dims=1))
        ??q = cov(dat)
        ??q = 0.5 * (??q + ??q')
        ??p = zeros(d)
        ??p = Matrix(I(d) * 1.)
        return 0.5 * (logdet(??p) - logdet(??q) - d + tr(??p \ ??q) + transpose(??p - ??q) * (??p \ (??p - ??q)))
    end
end
