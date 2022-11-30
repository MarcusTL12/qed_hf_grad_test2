
function get_gradient(outfilename)
    g = Float64[]
    file = Iterators.Stateful(eachline("$outfilename.molecular_gradient.out"))

    # Skip first 5 lines
    for _ in 1:5
        popfirst!(file)
    end

    for l in file
        s = split(l)

        if isone(length(s))
            break
        end

        append!(g, parse(Float64, n) for n in s[3:end])
    end

    reshape(g, 3, length(g) ÷ 3)
end
