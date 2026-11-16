# Automated quality checks: method ambiguities, unbound type parameters, undefined exports, stale dependencies,
# compat bounds, type piracy, and persistent tasks

using Aqua, HiddenMarkovRandomFields, Test

@testset "Aqua" begin
    Aqua.test_all(HiddenMarkovRandomFields)
end
