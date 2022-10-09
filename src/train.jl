using Flux, ProgressMeter, TickTock
using Zygote: Params, pullback


######################
## VI training functions (modify from Flux.optimise)
########################
#=
Perform update steps of the parameters `ps` (or the single parameter `p`)
according to optimizer `opt`  and the gradients `gs` (the gradient `g`).

As a result, the parameters are mutated and the optimizer's internal state may change.
The gradient could be mutated as well.
=#

# Callback niceties
call(f::Function, args...) = f(args...)
call(f::Function, args::Tuple) = f(args...)

# function trace the loss
function cb_loss!(logging_loss, ls_trace, ls, iter)
    if logging_loss
        ls_trace[iter] = ls
    else
        nothing
    end
end

# function that trace the updated params
function cb_ps!(logging_ps, ps_trace, ps::Params, iter::Int, niters::Int,  verbose_freq::Int)
    if logging_ps
        if iter % verbose_freq === 0
            # @info "training step $iter / $niters"
            # println(ps)
            pp = [copy(p) for p in ps]
            push!(ps_trace,  pp)
            println(pp)
        end
    else
        nothing
    end
end

function sparse_flow_trainT!(a, loss, ps::Params, ϵ_unc_hist, w_unc_hist, μps_hist, logσp_hist; logging_loss = true)
    # initialize ls_trace if logging_loss = true
    ls_trace = logging_loss ? Vector{Float64}(undef, a.iter) : nothing
    # progress bar
    prog_bar = ProgressMeter.Progress(a.iter, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    if a.save
        times = zeros(a.iter+1)
        tick()
        peek = peektimer()
        times[1] = peek
    end
    # optimization
    for i in 1:a.iter
        # compute loss, grad simultaneously
        ls, back = pullback(ps)do
            loss()
        end
        grads = back(1.0) # return grads::Array{Float64}
        Flux.update!(a.optimizer, ps, grads) # update parameters

        # logging and printing
        call(cb_loss!, logging_loss, ls_trace, ls, i)
        ProgressMeter.next!(prog_bar)

        if a.save
            peek = peektimer()
            times[i+1] = peek
            p1 = copy(ps[1])
            p2 = copy(ps[2])
            p3 = copy(ps[3])
            p4 = copy(ps[4])
            ϵ_unc_hist[i+1,:] = p1
            w_unc_hist[i+1,:] = p2
            μps_hist[i+1,:] = p3
            logσp_hist[i+1,:] = p4
        else
            if i == niters
                p1 = copy(ps[1])
                p2 = copy(ps[2])
                p3 = copy(ps[3])
                p4 = copy(ps[4])
                ϵ_unc_hist[1,:] = p1
                w_unc_hist[1,:] = p2
                μps_hist[i+1,:] = p3
                logσp_hist[i+1,:] = p4
            end
        end
    end
    if a.save
        return ls_trace, times
    else
        return ls_trace, []
    end
end