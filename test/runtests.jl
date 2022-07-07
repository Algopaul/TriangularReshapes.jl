using TriangularReshapes
using Test
using Random

@testset "TriangularReshapes.jl" begin
    @testset "Test vector_to_lower_triang" begin
        for dtype in [Float64, Float32]
            n = 5
            M = zeros(dtype, n, n)
            vl = n * (n + 1) ÷ 2
            v = Vector{dtype}(1:vl)
            vector_to_lower_triang!(M, v)
            M_target = Matrix{dtype}([
                1 0 0 0 0
                2 6 0 0 0
                3 7 10 0 0
                4 8 11 13 0
                5 9 12 14 15
            ])
            @test M == M_target
        end

        @testset "Test lower_triang_to_vector" begin
            for dtype in [Float64, Float32]
                n = 5
                M = Matrix{dtype}([
                    1 0 0 0 0
                    2 6 0 0 0
                    3 7 10 0 0
                    4 8 11 13 0
                    5 9 12 14 15
                ])
                vl = n * (n + 1) ÷ 2
                v = zeros(dtype, vl)
                lower_triang_to_vector!(v, M)
                @test v == Vector{dtype}(1:vl)
            end
        end

        @testset "Allocating Variants" begin
            for dtype in [Float64, Float32]
                n = 5
                v = rand(dtype, n * (n + 1) ÷ 2)
                M = vector_to_lower_triang(v)
                M_target = zeros(dtype, n, n)
                vector_to_lower_triang!(M_target, v)
                @test M == M_target
                v_test = lower_triang_to_vector(M)
                @test v == v_test
            end
        end

        @testset "Test lower_triang_to_vector, Rank1" begin
            for n in [5]
                for dtype in [Float64]
                    vl = n * (n + 1) ÷ 2
                    v = Vector{dtype}(undef, vl)
                    v1 = rand(MersenneTwister(0), dtype, n)
                    v2 = rand(MersenneTwister(1), dtype, n)
                    lower_triang_to_vector!(v, v1, v2)
                    V = v1 * v2'
                    v_test = lower_triang_to_vector(V)
                    @test v ≈ v_test
                end
            end
        end
    end
end
