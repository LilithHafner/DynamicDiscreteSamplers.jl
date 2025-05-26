isdefined(@__MODULE__, :Memory) || const Memory = Vector # Compat for Julia < 1.11
_get_UInt128(m::Memory, i::Integer) = UInt128(m[i]) | (UInt128(m[i+1]) << 64)
_length_from_memory(allocated_memory::Integer) = Int((allocated_memory-10524)/7)
function verify_weights(m::Memory)
    m3 = m[4]
    for i in 6:2051
        shift = signed(i - 5 + m3)
        weight = m[i]
        shifted_significand_sum_index = 2042 + 2i
        shifted_significand_sum = _get_UInt128(m, shifted_significand_sum_index)
        expected_weight = UInt64(shifted_significand_sum<<shift)
        expected_weight += (shifted_significand_sum != 0)
        @assert weight == expected_weight
    end
end

function verify_m2(m::Memory)
    @assert m[3] == findlast(i -> i == 4 || m[i] != 0, 1:2051)
end
function verify_m4(m::Memory)
    m4 = zero(UInt64)
    for i in 6:2051
        m4 = Base.checked_add(m4, m[i])
    end
    @assert m[5] == m4
    # @assert m4 == 0 || UInt64(2)^32 <= m4 # This invariant is now maintained loosely and lazily
end

function verify_edit_map_points_to_correct_target(m::Memory)
    filled_len = m[1]
    len = _length_from_memory(length(m))
    for i in 1:len
        edit_map_entry = m[i+10524]
        if i > filled_len
            @assert edit_map_entry == 0
        elseif edit_map_entry != 0
            @assert m[edit_map_entry>>11 + 1] == i
        end
    end
end

function verify(m::Memory)
    verify_weights(m)
    verify_m2(m)
    verify_m4(m)
    verify_edit_map_points_to_correct_target(m)
end
