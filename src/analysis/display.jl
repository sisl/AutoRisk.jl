
import Base: display, show

import AutoViz: SceneOverlay

export
    display,
    show

function Base.display(col::DatasetCollector, seed::Int64)
    reset!(col, seed)
    simulate!(col.scene, col.models, col.roadway, col.eval.rec, 
        col.eval.prime_time)
    veh_id_to_idx = get_veh_id_to_idx(col.scene)
    extract_features!(col.eval.rec, col.roadway, col.models, col.eval.features)
    simulate!(col.scene, col.models, col.roadway, col.eval.rec, 
        col.eval.sampling_time)
    extract_targets!(col.eval.rec, col.roadway, col.eval.targets, 
        veh_id_to_idx, col.eval.veh_idx_can_change)
    show(col)
end

function Base.show(col::DatasetCollector)
in_collision_veh_idxs = find(col.eval.targets[1,:] .== 1.)
    @manipulate for follow_veh_idx in in_collision_veh_idxs,
                zoom in collect(1.:2:20.),
                i in 0:(col.eval.rec.nscenes - 1)
        # set camera
        follow_veh_id = -1
        if follow_veh_idx == 0
            cam = FitToContentCamera()
        else
            for (veh_id, veh_idx) in col.eval.veh_id_to_idx
                if veh_idx == follow_veh_idx
                    follow_veh_id = veh_id
                    break
                end
            end
            cam = CarFollowCamera(follow_veh_id, zoom)
        end
        
        # render scene
        idx = -(col.eval.rec.nscenes - i)
        carcolors = Dict{Int,Colorant}()
        for veh in get_scene(col.eval.rec, idx)
            carcolors[veh.def.id] = veh.def.id == follow_veh_id ? colorant"green" : colorant"red"
        end
        stats = follow_veh_id == -1 ? [] : [CarFollowingStatsOverlay(follow_veh_id)]
        render(get_scene(col.eval.rec, idx), col.roadway, stats,
            cam = cam, car_colors = carcolors)
    end
end
