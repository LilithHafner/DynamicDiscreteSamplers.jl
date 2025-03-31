isdefined(@__MODULE__, :Memory) || const Memory = Vector # Compat for Julia < 1.11
_get_UInt128(m::Memory, i::Integer) = UInt128(m[i]) | (UInt128(m[i+1]) << 64)
_length_from_memory(allocated_memory::Integer) = Int((allocated_memory-10523)/7)
function verify_weights(w::DynamicDiscreteSamplers.Weights)
    m = w.m
    m3 = m[3]
    for i in 5:2050
        shift = signed(i - 4 + m3)
        weight = m[i]
        shifted_significand_sum_index = 2041 + 2i
        shifted_significand_sum = _get_UInt128(m, shifted_significand_sum_index)
        expected_weight = UInt64(shifted_significand_sum<<shift)
        expected_weight += (shifted_significand_sum != 0)
        @assert weight == expected_weight
    end
end

function verify_m2(w::DynamicDiscreteSamplers.Weights)
    m = w.m
    @assert m[2] >= findlast(i -> i == 4 || m[i] != 0, 1:2050)
    if m[4] != 0
        rand(w)
        @assert m[2] == findlast(i -> m[i] != 0, 1:2050)
    end
end
function verify_m4(w::DynamicDiscreteSamplers.Weights)
    m = w.m
    m4 = zero(UInt64)
    for i in 5:2050
        m4 = Base.checked_add(m4, m[i])
    end
    @assert m[4] == m4
    # @assert m4 == 0 || UInt64(2)^32 <= m4 # This invariant is now maintained loosely and lazily
end

function verify_edit_map_points_to_correct_target(w::DynamicDiscreteSamplers.Weights)
    m = w.m
    filled_len = m[1]
    len = _length_from_memory(length(m))
    for i in 1:len
        edit_map_entry = m[i+10523]
        if i > filled_len
            @assert edit_map_entry == 0
        elseif edit_map_entry != 0
            @assert m[edit_map_entry>>11 + 1] == i
        end
    end
end

function verify(w::DynamicDiscreteSamplers.Weights)
    verify_weights(w)
    verify_m2(w)
    verify_m4(w)
    verify_edit_map_points_to_correct_target(w)
end
