using TriangularReshapes
using Test
using Random

@testset verbose = true "TriangularReshapes" begin

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
            b_alloc = @allocated vector_to_lower_triang!(M, v)
            @test b_alloc == 0
        end
    end

    @testset "Test vector_to_strictly_lower_triang" begin
        for dtype in [Float64, Float32]
            n = 6
            M = zeros(dtype, n, n)
            vl = n * (n - 1) ÷ 2
            v = Vector{dtype}(1:vl)
            vector_to_strictly_lower_triang!(M, v)
            M_target = Matrix{dtype}(
                [
                    0 0 0 0 0 0
                    1 0 0 0 0 0
                    2 6 0 0 0 0
                    3 7 10 0 0 0
                    4 8 11 13 0 0
                    5 9 12 14 15 0
                ],
            )
            @test M == M_target
            b_alloc = @allocated vector_to_strictly_lower_triang!(M, v)
            @test b_alloc == 0
        end
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
            b_alloc = @allocated lower_triang_to_vector!(v, M)
            @test b_alloc == 0
        end
    end

    @testset "Test vector_to_strictly_lower_triang" begin
        for dtype in [Float64, Float32]
            n = 6
            vl = n * (n - 1) ÷ 2
            v = zeros(dtype, vl)
            M = Matrix{dtype}(
                [
                    0 0 0 0 0 0
                    1 0 0 0 0 0
                    2 6 0 0 0 0
                    3 7 10 0 0 0
                    4 8 11 13 0 0
                    5 9 12 14 15 0
                ],
            )
            strictly_lower_triang_to_vector!(v, M)
            @test v == Vector(1:15)
            b_alloc = @allocated strictly_lower_triang_to_vector!(v, M)
            @test b_alloc == 0
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

    @testset "Allocating Variants (strictly)" begin
        for dtype in [Float64, Float32]
            n = 5
            v = rand(dtype, n * (n - 1) ÷ 2)
            M = vector_to_strictly_lower_triang(v)
            M_target = zeros(dtype, n, n)
            vector_to_strictly_lower_triang!(M_target, v)
            @test M == M_target
            v_test = strictly_lower_triang_to_vector(M)
            @test v == v_test
        end
    end

    function test_lower_triang_to_vector_rank1(n, c1, c2, adding, makereal)
        vl = n * (n + 1) ÷ 2
        v = makereal ? rand(Float64, vl) : rand(ComplexF64, vl)
        vc = deepcopy(v)
        v1 = rand(ComplexF64, n)
        v2 = rand(ComplexF64, n)
        v1c = c1 ? conj.(v1) : v1
        v2c = c2 ? conj.(v2) : v2
        V = v1c * transpose(v2c)
        lower_triang_to_vector!(v, v1, v2; c1, c2, adding)
        vt = lower_triang_to_vector(V)
        if makereal
            vt = real(vt)
        end
        if adding
            vt .+= vc
        end
        @test vt ≈ v
        b_alloc = @allocated lower_triang_to_vector!(v, v1, v2; c1, c2, adding)
        @test b_alloc == 0
    end

    @testset "Test lower_triang_to_vector, Rank1" begin
        for n in [5, 10]
            for c1 in [true, false]
                for c2 in [true, false]
                    for adding in [true, false]
                        for makereal in [true, false]
                            test_lower_triang_to_vector_rank1(n, c1, c2, adding, makereal)
                        end
                    end
                end
            end
        end
    end

    function test_strictly_lower_triang_to_vector_rank1(n, c1, c2, adding, makereal)
        vl = n * (n - 1) ÷ 2
        v = makereal ? rand(Float64, vl) : rand(ComplexF64, vl)
        vc = deepcopy(v)
        v1 = rand(ComplexF64, n)
        v2 = rand(ComplexF64, n)
        v1c = c1 ? conj.(v1) : v1
        v2c = c2 ? conj.(v2) : v2
        V = v1c * transpose(v2c)
        strictly_lower_triang_to_vector!(v, v1, v2; c1, c2, adding)
        vt = strictly_lower_triang_to_vector(V)
        if makereal
            vt = real(vt)
        end
        if adding
            vt .+= vc
        end
        @test vt ≈ v
        b_alloc = @allocated strictly_lower_triang_to_vector!(v, v1, v2; c1, c2, adding)
        @test b_alloc == 0
    end

    @testset "Test strictly_lower_triang_to_vector, Rank1" begin
        for n in [5, 10]
            for c1 in [true, false]
                for c2 in [true, false]
                    for adding in [true, false]
                        for makereal in [true, false]
                            test_strictly_lower_triang_to_vector_rank1(
                                n,
                                c1,
                                c2,
                                adding,
                                makereal,
                            )
                        end
                    end
                end
            end
        end
    end

end
