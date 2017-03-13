export
    Monitor,
    Submonitor,
    ScenarioRecorderMonitor,
    monitor

abstract Submonitor

# wrapper around a collection of submonitors
@with_kw type Monitor
    output_directory::String = "" # where to store outputs of submonitors
    submonitors::Array{Submonitor} = Submonitor[]
end

function monitor(mon::Any, col::DatasetCollector, seed::Int)
    for sub in mon.submonitors
        monitor(sub, col, mon.output_directory, seed)
    end
end

# simulates a scenario, records it, and saves it to file
@with_kw type ScenarioRecorderMonitor
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
        veh_id = col.scene.vehicles[1].def.id
        stats = follow_veh_id == [
            CarFollowingStatsOverlay(veh_id, 2), 
            NeighborsOverlay(veh_id, textparams = TextParams(x = 600, y_start=300))
        ]
        cam = CarFollowCamera(veh_id, mon.zoom)
        for idx in 1:length(col.eval.rec)
            frame = render(get_scene(col.eval.rec, idx - length(col.eval.rec)), 
                col.roadway, stats, cam = cam)
            push!(frames, frame)
        end
        filename = "seed_$(seed)_veh_id_$(veh_id).gif"
        filepath = joinpath(output_directory, filename)
        write(filepath, frames)
    end
end