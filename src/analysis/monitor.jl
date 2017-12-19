export
    Monitor,
    Submonitor,
    ScenarioRecorderMonitor,
    monitor

abstract type Submonitor end

# wrapper around a collection of submonitors
type Monitor
    output_directory::String
    submonitors::Array{Submonitor}
    function Monitor(output_directory::String = "", 
            submonitors::Array{Submonitor} = Submonitor[])
        if !isdir(output_directory)
            mkdir(output_directory)
        end
        return new(output_directory, submonitors)
    end
end
function monitor(mon::Monitor, col::DatasetCollector, seed::Int)
    for sub in mon.submonitors
        @spawn monitor(sub, col, mon.output_directory, seed)
    end
end

# simulates a scenario, records it, and saves it to file
@with_kw type ScenarioRecorderMonitor <: Submonitor
    freq::Int = 100 # frequency to apply the monitor (in scenarios)
    zoom::Float64 = 10. # camera zoom
end
function monitor(mon::ScenarioRecorderMonitor, col::DatasetCollector,
        output_directory::String, seed::Int)
    if seed % mon.freq == 0
        rand!(col, seed)
        simulate!(col.scene, col.models, col.roadway, col.eval.rec, 
            col.eval.prime_time + col.eval.sampling_time)
        frames = Frames(MIME("image/png"), fps = 10)
        veh_id = col.scene.vehicles[1].id
        stats = [
            CarFollowingStatsOverlay(veh_id, 2), 
            NeighborsOverlay(veh_id, textparams = TextParams(x = 600, y_start=300))
        ]
        cam = CarFollowCamera(veh_id, mon.zoom)
        for idx in 1:length(col.eval.rec)
            frame = render(col.eval.rec[idx - length(col.eval.rec)], 
                col.roadway, stats, cam = cam)
            push!(frames, frame)
        end
        filename = "seed_$(seed)_veh_id_$(veh_id).gif"
        filepath = joinpath(output_directory, filename)
        write(filepath, frames)
    end
end