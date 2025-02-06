using Base.JuliaSyntax

function code_size(file)
    text = String(read(file))
    tokens = tokenize(text)

    byte_has_code = trues(ncodeunits(text))

    last_end = 0
    for t in tokens
        last_end = last(t.range)
        if kind(t) ∈ JuliaSyntax.KSet"Comment Whitespace NewlineWs"
            byte_has_code[t.range] .= false
        end
    end

    syntax_nodes = 0

    stack = parseall(SyntaxNode, text, ignore_warnings=true).children
    while !isempty(stack)
        x = pop!(stack)
        if kind(x) == K"doc"
            d = x.children[1]
            rng = (d.position-1) .+ (1:JuliaSyntax.span(d))
            byte_has_code[rng] .= false
            x = x.children[2]
        end
        x.children !== nothing && append!(stack, x.children)
        syntax_nodes += 1
    end

    newlines = vcat(0, findall(==('\n'), text), ncodeunits(text))
    lines = count(any(view(byte_has_code, newlines[i]+1:newlines[i+1])) for i in 1:length(newlines)-1)

    (lines=lines, bytes=sum(byte_has_code), syntax_nodes)
end
