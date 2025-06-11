isdefined(@__MODULE__, :Memory) || const Memory = Vector # Compat for Julia < 1.11
_get_UInt128(m::Memory, i::Integer) = UInt128(m[i]) | (UInt128(m[i+1]) << 64)
_length_from_memory(allocated_memory::Integer) = Int((allocated_memory-10524)/7)
function verify_weights(m::Memory)
    m3 = m[3]
    for i in 6:2051
        shift = signed(i - 5 + m3)
        weight = m[i]
        shifted_significand_sum_index = 2040 + 2i
        shifted_significand_sum = _get_UInt128(m, shifted_significand_sum_index)
        expected_weight = UInt64(shifted_significand_sum<<shift)
        expected_weight += (shifted_significand_sum != 0)
        @assert weight == expected_weight
    end
end

verify_m1(m::Memory) = m[1] == _length_from_memory(length(m))
function verify_m2(m::Memory)
    @assert m[2] == findlast(i -> i == 5 || m[i] != 0, 1:2051)
end
function verify_m5(m::Memory)
    m5 = zero(UInt64)
    for i in 6:2051
        m5 = Base.checked_add(m5, m[i])
    end
    @assert m[5] == m5
    # @assert m4 == 0 || UInt64(2)^32 <= m4 # This invariant is now maintained loosely and lazily
end

function verify_m4_and_edit_map_points_to_correct_target(m::Memory)
    len = _length_from_memory(length(m))
    m4 = 0
    for i in 1:len
        edit_map_entry = m[i+10524]
        if edit_map_entry != 0
            @assert m[edit_map_entry>>11 + 1] == i
        end
        m4 += edit_map_entry != 0
    end
    @assert m[4] == m4
end

function verify(m::Memory)
    verify_weights(m)
    verify_m1(m)
    verify_m2(m)
    verify_m5(m)
    verify_m4_and_edit_map_points_to_correct_target(m)
end
