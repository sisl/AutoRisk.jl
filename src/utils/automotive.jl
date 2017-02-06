export
    stadium_roadway_length,
    straight_roadway_length,
    get_total_roadway_length,
    stadium_roadway_area,
    straight_roadway_area,
    get_total_roadway_area,
    inverse_ttc_to_ttc,
    push_forward_records!

"""
AutomotiveDrivingModels Core additional functionality
"""

### Roadway
function stadium_roadway_length(roadway::Roadway)
    seg = roadway.segments[1]
    straight_length = seg.lanes[1].curve[1].pos.x
    curve_length = pi * seg.lanes[1].curve[end].pos.y
    return 2 * straight_length + 2 * curve_length
end

function straight_roadway_length(roadway::Roadway)
    return roadway.segments[1].lanes[1].curve[end].pos.x
end

function get_total_roadway_length(roadway::Roadway)
    # straight
    if length(roadway.segments) == 1
        return straight_roadway_length(roadway)

    # stadium
    else
        return stadium_roadway_length(roadway)
    end
end

function stadium_roadway_area(roadway::Roadway)
    seg = roadway.segments[1]
    
    # area of straight components
    slength = seg.lanes[1].curve[1].pos.x
    width = seg.lanes[1].width
    num_lanes = length(seg.lanes)
    straight_area = 2 * slength * width * num_lanes

    # area of curves
    inner_radius = seg.lanes[1].curve[end].pos.y
    outer_radius = inner_radius + num_lanes * width
    curve_area = pi * outer_radius^2 - pi * inner_radius^2 

    return straight_area + curve_area
end

function straight_roadway_area(roadway::Roadway)
    width = roadway.segments[1].lanes[1].width
    num_lanes = length(roadway.segments[1].lanes)
    slength = roadway.segments[1].lanes[1].curve[end].pos.x
    area = num_lanes * width * slength
    return area
end

function get_total_roadway_area(roadway::Roadway)
    # straight
    if length(roadway.segments) == 1
        return straight_roadway_area(roadway)

    # stadium
    else
        return stadium_roadway_area(roadway)
    end
end

function Base.:(==)(c1::CurvePt, c2::CurvePt)
    return (c1.pos == c2.pos
        && c1.s == c2.s
        && (c1.k == c2.k || (isnan(c1.k) && isnan(c2.k)))
        && (c1.kd == c2.kd || (isnan(c1.kd) && isnan(c2.kd))))
end 

function Base.:(==)(l1::Lane, l2::Lane)
    return (l1.tag == l2.tag 
        && all(pt1 == pt2 for (pt1, pt2) in zip(l1.curve, l2.curve))
        && l1.width == l2.width
        && l1.speed_limit == l2.speed_limit)
end

function Base.:(==)(r1::RoadSegment, r2::RoadSegment)
    return (r1.id == r2.id 
        && all(l1 == l2 for (l1, l2) in zip(r1.lanes, r2.lanes)))
end

function Base.:(==)(r1::Roadway, r2::Roadway)
    return all(rs1 == rs2 for (rs1, rs2) in zip(r1.segments, r2.segments))
end

### Vehicle
function Base.:(==)(vd1::VehicleDef, vd2::VehicleDef)
    return (vd1.id == vd2.id 
            && vd1.class == vd2.class 
            && vd1.length == vd2.length
            && vd1.width == vd2.width)
end

function Base.:(==)(f1::Frenet, f2::Frenet)
    return (f1.roadind == f2.roadind 
            && f1.s == f2.s 
            && f1.t == f2.t 
            && f1.ϕ == f2.ϕ)
end

function Base.:(==)(vs1::VehicleState, vs2::VehicleState)
    return (vs1.posG == vs2.posG 
            && vs1.posF == vs2.posF 
            && vs1.v == vs2.v)
end

function Base.:(==)(v1::Vehicle, v2::Vehicle)
    return v1.state == v2.state && v1.def == v2.def
end


function Base.show(io::IO, vehicle::Vehicle)
    print(io, "Vehicle Def:\n$(vehicle.def)")
    print(io, "\nVehicle State:\n$(vehicle.state)")
end

function Base.show(io::IO, def::VehicleDef)
    print(io, "\tid: $(def.id)")
    print(io, "\tclass: $(def.class)")
    print(io, "\tlength: $(def.length)")
    print(io, "\twidth: $(def.width)")
end

function Base.show(io::IO, state::VehicleState)
    print(io, "\tglobal position: $(state.posG)\n")
    print(io, "\tlane-relative position:\n$(state.posF)\n")
    print(io, "\tvelocity: $(state.v)\n")
end

function Base.show(io::IO, f::Frenet)
    print(io, "\t\troadind: $(f.roadind)\n")
    print(io, "\t\tdistance along lane: $(f.s)\n")
    print(io, "\t\tlane offset: $(f.t)\n")
    print(io, "\t\tlane-relative heading: $(f.ϕ)")
end

### Scene
function Base.:(==)(s1::Scene, s2::Scene)
    if s1.n_vehicles != s2.n_vehicles
        return false
    end

    for (v1, v2) in zip(s1, s2)
        if v1 != v2
            return false
        end
    end

    return true
end

function Base.show(io::IO, scene::Scene)
    for (i, veh) in enumerate(scene.vehicles)
        println(io, "vehicle $(i):\n$(veh)")
    end
end

### SceneRecord
"""
Description:
    - Empty a scene record beginning at a pastframe (i.e., all subsequent frames
        removed).

Args:
    - rec: scene record to partially empty
    - pastframe: nonpositive interger indicating the frame in the past after 
        which frames should be removed (this value begins counting after the 
        first frame in the record).
"""
function push_forward_records!(rec::SceneRecord, pastframe::Int)
    # calling with pastframe of 0 does not change the record, return immediately
    if pastframe == 0
        return rec
    end
    s, e = 1 - pastframe, rec.nscenes
    for (i, past_index) in enumerate(s:e)
        copy!(rec.scenes[i], rec.scenes[past_index])
    end
    rec.nscenes = e - s + 1
    return rec
end

# get_neighbor_rear_along_lane in ADM.jl has a bug, and this is here to fix it
# the bug is that when rear neighbors are on a different segment than the ego
# vehicle, the ego will mistake its fore neighbors for its rear ones
# the fix in this function is to not consider fore neighbors by marking 
# all the vehicles that have a negative distance and not considering them twice
function AutomotiveDrivingModels.get_neighbor_rear_along_lane(
    scene::Scene,
    roadway::Roadway,
    tag_start::LaneTag,
    s_base::Float64,
    targetpoint_primary::VehicleTargetPoint, # the reference point whose distance we want to minimize
    targetpoint_valid::VehicleTargetPoint; # the reference point, which if distance to is positive, we include the vehicle
    max_distance_rear::Float64 = 250.0, # max distance to search rearward [m]
    index_to_ignore::Int=-1,
    )

    best_ind = 0
    best_dist = max_distance_rear
    tag_target = tag_start

    ignore = Set{Int}()

    dist_searched = 0.0
    while dist_searched < max_distance_rear

        lane = roadway[tag_target]

        for (i,veh) in enumerate(scene)
            if i != index_to_ignore && !in(veh.def.id, ignore)

                s_adjust = NaN

                if veh.state.posF.roadind.tag == tag_target
                    s_adjust = 0.0

                elseif is_between_segments_hi(veh.state.posF.roadind.ind, lane.curve) &&
                       is_in_entrances(roadway[tag_target], veh.state.posF.roadind.tag)

                    distance_between_lanes = abs(roadway[tag_target].curve[1].pos - roadway[veh.state.posF.roadind.tag].curve[end].pos)
                    s_adjust = -(roadway[veh.state.posF.roadind.tag].curve[end].s + distance_between_lanes)

                elseif is_between_segments_lo(veh.state.posF.roadind.ind) &&
                       is_in_exits(roadway[tag_target], veh.state.posF.roadind.tag)

                    distance_between_lanes = abs(roadway[tag_target].curve[end].pos - roadway[veh.state.posF.roadind.tag].curve[1].pos)
                    s_adjust = roadway[tag_target].curve[end].s + distance_between_lanes
                end

                if !isnan(s_adjust)
                    s_valid = veh.state.posF.s + get_targetpoint_delta(targetpoint_valid, veh) + s_adjust
                    dist_valid = s_base - s_valid + dist_searched
                    if dist_valid ≥ 0.0
                        s_primary = veh.state.posF.s + get_targetpoint_delta(targetpoint_primary, veh) + s_adjust
                        dist = s_base - s_primary + dist_searched
                        if dist < best_dist
                            best_dist = dist
                            best_ind = i
                        end
                    else
                        push!(ignore, veh.def.id)
                    end
                end
            end
        end

        if best_ind != 0
            break
        end

        if !has_prev(lane) ||
           (tag_target == tag_start && dist_searched != 0.0) # exit after visiting this lane a 2nd time
            break
        end

        dist_searched += s_base
        s_base = lane.curve[end].s + abs(lane.curve[end].pos - prev_lane_point(lane, roadway).pos) # end of prev lane plus crossover
        tag_target = prev_lane(lane, roadway).tag
    end

    NeighborLongitudinalResult(best_ind, best_dist)
end

### DriverModel
function Base.:(==)(d1::DriverModel, d2::DriverModel)
    fns = fieldnames(d1)
    return (fns == fieldnames(d2)
        && all(getfield(d1, f) == getfield(d2, f) for f in fns))
end

function Base.:(==)(a1::LatLonAccel, a2::LatLonAccel)
    return a1.a_lat == a2.a_lat && a1.a_lon == a2.a_lon
end

### Features
function inverse_ttc_to_ttc(inv_ttc::FeatureValue; censor_hi::Float64 = 30.0)
    if inv_ttc.i == FeatureState.MISSING
        # if the value is missing then leave at zero and set missing
        return FeatureValue(0.0, FeatureState.MISSING)
    elseif inv_ttc.i == FeatureState.GOOD && inv_ttc.v == 0.0
        # if the car in front is pulling away, then set to a censored hi value
        return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
    else
        # even if the value was censored hi, can still take the inverse
        return FeatureValue(1.0 / inv_ttc.v)
    end
end

function changed_lanes_recently(rec::SceneRecord, roadway::Roadway, 
        vehicle_index::Int, pastframe::Int = 0; lane_change_timesteps = 10,
        lane_change_heading_threshold = .1)
    # get final information
    scene = get_scene(rec, pastframe)
    veh = scene[vehicle_index]
    veh_id = veh.def.id
    final_lane = veh.state.posF.roadind.tag.lane

    # step backward to check for lane change
    for dt in 1:lane_change_timesteps
        if pastframe_inbounds(rec, pastframe - dt)
            veh_state = get_vehiclestate(rec, veh_id, pastframe - dt)
            if (veh_state.posF.roadind.tag.lane != final_lane
                || abs(veh_state.posF.ϕ) > lane_change_heading_threshold)
                return true
            end
        end
    end
    return false
end

function get_collision_type(rec::SceneRecord, roadway::Roadway, 
        vehicle_index::Int, pastframe::Int = 0)
    # get collision result from scene
    scene = get_scene(rec, pastframe)
    collision = get_first_collision(scene, vehicle_index)

    # label and return the collision
    in_collision = collision.is_colliding
    ego_idx, other_idx = collision.A, collision.B
    collision_type = 0
    if in_collision
        # colliding vehicles 
        ego = scene[ego_idx]
        other = scene[other_idx]

        if (changed_lanes_recently(rec, roadway, ego_idx, pastframe)
            || changed_lanes_recently(rec, roadway, other_idx, pastframe))
            collision_type = 1

        # ego vehicle further along lane then rear-end-lead
        elseif ego.state.posF.s > other.state.posF.s 
            collision_type = 2

        # other vehicle further along lane then rear-end-follow
        else
            collision_type = 3
        end

    end

    return collision_type
end

