export 
    Entry,
    set_value!,
    Flags,
    add_entry!,
    parse_flags!

"""
Simple class for parsing command line arguments.

Example usage:
```
flags = Flags()
add_entry!(flags, "hello", 5, Int, "a description")
parse_flags!(flags, ARGS)
println(flags)
```
"""

type Entry
    key::String
    value::Any
    datatype::DataType
    description::String
        function Entry(key::String, value::Any, datatype::DataType, 
                description::String)
            if typeof(value) != datatype
                value = parse(datatype, value)
            end
        return new(key, value, datatype, description)
    end
end

function set_value!(entry::Entry, value::Any)
    if typeof(value) != entry.datatype
        value = parse(entry.datatype, value)
    end
    entry.value = value
end

type Flags
    d::Dict{String, Entry}
    function Flags()
        return new(Dict{String, Entry}())
    end
end

function Base.getindex(flags::Flags, key::String)
    return flags.d[key].value
end

function Base.setindex!(flags::Flags, value::Any, key::String)
    set_value!(flags.d[key], value)
end

function add_entry!(flags::Flags, key::String, value::Any, 
        datatype::DataType, description::String = "")
    entry = Entry(key, value, datatype, description)
    flags.d[entry.key] = entry
end

function Base.show(io::IO, flags::Flags)
    println("Flags:")
    for k in sort(collect(keys(flags.d)))
        println("$(k) => $(flags.d[k])")
    end
end

function Base.convert(::Type{Dict}, flags::Flags)
    d = Dict()
    for (key, entry) in flags.d
        d[key] = entry.value
    end
    return d
end

"""
# Description:
    - parses args into (key, value) pairs.

# Args:
    - flags: dictionary that defines all the flags to use
        along with default values.
    - args: list of strings such that in iterating the list
        odd-index strings are keys and even-index strings are
        values. On/Off flags not implemented.

# Returns:
    - returns the dictionary containing parsed args
"""
function parse_flags!(flags, args)
    # parse command line
    for idx in 1:2:length(args)
        # remove "--"" from start of key
        key = args[idx][3:end]
        
        # validate key in flags, and warn if not
        if !in(key, keys(flags.d))
           println("$(key) not found in flags, skipping.")
           continue
        end 
   
        value = args[idx + 1]
        flags[key] = value
    end
    return flags
end