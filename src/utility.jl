# Helpers shared by the models

# Apply f(i, workspace) for i = 1,…,n in parallel over tasks, with dynamic load balancing. Each task creates its own
# workspace by calling `workspace()` (e.g., `workspace = PottsWorkspace` for the Potts sampler).
function _tforeach(f, n::Integer; ntasks::Integer = Threads.nthreads(), workspace = () -> nothing)
    n <= 0 && return nothing
    ntasks = max(1, min(ntasks, n))
    next = Threads.Atomic{Int}(1)
    if ntasks == 1
        _tforeach_worker(f, next, n, workspace)
    else
        @sync for _ = 1:ntasks
            Threads.@spawn _tforeach_worker(f, next, n, workspace)
        end
    end
    return nothing
end

function _tforeach_worker(f, next::Threads.Atomic{Int}, n::Integer, workspace)
    ws = workspace()
    while true
        i = Threads.atomic_add!(next, 1)
        i > n && break
        f(i, ws)
    end
    return nothing
end

# ---- Persistent worker tasks, synchronised by a barrier ----

# Barrier for a fixed group of n tasks. Waiting tasks spin, so that a barrier is passed within about a microsecond when
# every task has its own thread, and yield after a while, so that the barrier also works with more tasks than threads.
# The last task to arrive runs a sequential step before releasing the others. If a task fails, the barrier is aborted,
# which releases the tasks that are waiting at it (they then throw _BarrierAborted).
mutable struct _SpinBarrier
    @atomic arrived::Int
    @atomic generation::Int
    @atomic aborted::Bool
    const n::Int
end
_SpinBarrier(n::Integer) = _SpinBarrier(0, 0, false, Int(n))

struct _BarrierAborted <: Exception end

const _SPINS_BEFORE_YIELD = 1000

_abort!(barrier::_SpinBarrier) = (@atomic barrier.aborted = true; nothing)

# Waits until all n tasks have arrived at the barrier; the last task to arrive runs f() before releasing the others
function _wait!(f, barrier::_SpinBarrier)
    generation = @atomic barrier.generation
    if (@atomic barrier.arrived += 1) == barrier.n
        try
            f()
        catch
            _abort!(barrier)
            rethrow()
        finally
            @atomic barrier.arrived = 0
            @atomic barrier.generation = generation + 1
        end
    else
        spins = 0
        while (@atomic barrier.generation) == generation
            (@atomic barrier.aborted) && throw(_BarrierAborted())
            if spins < _SPINS_BEFORE_YIELD
                spins += 1
                ccall(:jl_cpu_pause, Cvoid, ())
                GC.safepoint()
            else
                yield()
            end
        end
    end
    (@atomic barrier.aborted) && throw(_BarrierAborted())
    return nothing
end

# Runs f(w, barrier) for w = 1, …, ntasks concurrently, with a barrier shared by the tasks. Every worker is a spawned
# task, and the calling task waits for them. (If the calling task were a worker, then whenever it waited at the barrier
# on its own thread, it could keep the other workers from running there: a sticky task, such as the main task, that
# yields is resumed before any other task of its thread. With as many workers as threads on Julia 1.11, where the main
# task shares the threads of the spawned tasks, this could leave a worker that has not yet arrived with no thread.)
# If any task fails, the barrier is aborted, so that no task is left waiting, and the first error is rethrown.
function _withworkers(f, ntasks::Integer)
    barrier = _SpinBarrier(ntasks)
    tasks = map(w -> Threads.@spawn(_runworker(f, w, barrier)), 1:ntasks)
    errors = map(fetch, tasks)
    for e in errors
        (isnothing(e) || e isa _BarrierAborted) || throw(e)
    end
    return nothing
end

function _runworker(f, w, barrier::_SpinBarrier)
    try
        f(w, barrier)
        return nothing
    catch e
        _abort!(barrier)
        return e
    end
end

# A scalar parameter, possibly given as a container with one element (e.g., a parameter vector)
function _scalar(β)
    length(β) == 1 || throw(ArgumentError("β must be a number, or a vector with one value per chain; got an object of length $(length(β))"))
    return first(β)
end

# Sorted, unique iterations at which to save the state of the chains
function _check_save_at(save_at, num_iterations::Integer)
    num_iterations >= 0 || throw(ArgumentError("num_iterations must be non-negative; got $num_iterations"))
    save_at = save_at isa Integer ? [Int(save_at)] : sort!(unique!(vec(collect(Int, save_at))))
    isempty(save_at) && throw(ArgumentError("save_at must contain at least one iteration"))
    all(s -> 0 <= s <= num_iterations, save_at) || throw(ArgumentError("save_at must contain iterations between 0 and num_iterations = $num_iterations; got $save_at"))
    return save_at
end
