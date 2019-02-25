export
    stadium_roadway_length,
    straight_roadway_length,
    get_total_roadway_length,
    stadium_roadway_area,
    straight_roadway_area,
    get_total_roadway_area,
    get_roadway_type,
    inverse_ttc_to_ttc,
    push_forward_records!,
    executed_hard_brake,
    srand

"""
AutomotiveDrivingModels Core additional functionality
"""
function nlanes(roadway::Roadway)
    return length(roadway.segments[1].lanes)
end

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

function get_roadway_type(roadway::Roadway)
    # straight
    if length(roadway.segments) == 1
        return "straight"
    else
        return "stadium"
    end
end

function Base.:(==)(c1::CurvePt, c2::CurvePt)
    return (c1.pos == c2.pos
        && c1.s == c2.s
        && (c1.k == c2.k || (isnan(c1.k) && isnan(c2.k)))
        && (c1.kd == c2.kd || (isnan(c1.kd) && isnan(c2.kd))))
end 

#=
Deprecated overwrite of AutomotiveDrivingModels' version
function Base.:(==)(l1::Lane, l2::Lane)
    return (l1.tag == l2.tag 
        && all(pt1 == pt2 for (pt1, pt2) in zip(l1.curve, l2.curve))
        && l1.width == l2.width
        && l1.speed_limit == l2.speed_limit)
end
=#

function Base.:(==)(r1::RoadSegment, r2::RoadSegment)
    return (r1.id == r2.id 
        && all(l1 == l2 for (l1, l2) in zip(r1.lanes, r2.lanes)))
end

function Base.:(==)(r1::Roadway, r2::Roadway)
    return all(rs1 == rs2 for (rs1, rs2) in zip(r1.segments, r2.segments))
end

### Vehicle
function Base.:(==)(vd1::VehicleDef, vd2::VehicleDef)
    return (vd1.class == vd2.class 
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


# function Base.show(io::IO, vehicle::Vehicle)
#     print(io, "Vehicle Def:\n$(vehicle.def)")
#     print(io, "\nVehicle State:\n$(vehicle.state)")
# end

# function Base.show(io::IO, def::VehicleDef)
#     print(io, "\tid: $(def.id)")
#     print(io, "\tclass: $(def.class)")
#     print(io, "\tlength: $(def.length)")
#     print(io, "\twidth: $(def.width)")
# end

# function Base.show(io::IO, state::VehicleState)
#     print(io, "\tglobal position: $(state.posG)\n")
#     print(io, "\tlane-relative position:\n$(state.posF)\n")
#     print(io, "\tvelocity: $(state.v)\n")
# end

# function Base.show(io::IO, f::Frenet)
#     print(io, "\t\troadind: $(f.roadind)\n")
#     print(io, "\t\tdistance along lane: $(f.s)\n")
#     print(io, "\t\tlane offset: $(f.t)\n")
#     print(io, "\t\tlane-relative heading: $(f.ϕ)")
# end

### Scene
function Base.:(==)(s1::Scene, s2::Scene)
    if s1.n != s2.n
        return false
    end

    for (v1, v2) in zip(s1, s2)
        if v1 != v2
            return false
        end
    end

    return true
end

#=
Deprecated overwrite of AutomotiveDrivingModels' version
function Base.show(io::IO, scene::Scene)
    for (i, veh) in enumerate(scene)
        println(io, "vehicle $(i):\n$(veh)")
    end
end
=#

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
    s, e = 1 - pastframe, rec.nframes
    for (i, past_index) in enumerate(s:e)
        copy!(rec.frames[i], rec.frames[past_index])
    end
    rec.nframes = e - s + 1
    return rec
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
        # if the value is missing then censor hi and set missing
        return FeatureValue(censor_hi, FeatureState.MISSING)
    elseif inv_ttc.i == FeatureState.GOOD && inv_ttc.v == 0.0
        # if the car in front is pulling away, then set to a censored hi value
        return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
    else
        # even if the value was censored hi, can still take the inverse
        ttc = 1.0 / inv_ttc.v
        if ttc > censor_hi
            return FeatureValue(censor_hi, FeatureState.CENSORED_HI)
        else
            return FeatureValue(ttc, FeatureState.GOOD)
        end
    end
end

function changed_lanes_recently(rec::SceneRecord, roadway::Roadway, 
        vehicle_index::Int, pastframe::Int = 0; lane_change_timesteps = 10)
    # get final information
    scene = rec[pastframe]
    veh = scene[vehicle_index]
    veh_id = veh.id
    final_lane = veh.state.posF.roadind.tag.lane

    # step backward to check for lane change
    for dt in 1:lane_change_timesteps
        if pastframe_inbounds(rec, pastframe - dt)
            veh_state = rec[pastframe - dt][vehicle_index].state
            if veh_state.posF.roadind.tag.lane != final_lane
                return true
            end
        end
    end
    return false
end

function get_collision_type(rec::SceneRecord, roadway::Roadway, 
        vehicle_index::Int, pastframe::Int = 0)
    # get collision result from scene
    scene = rec[pastframe]
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

        else 
            neigh = get_neighbor_fore_along_lane(scene, ego_idx, roadway)
            # fore neighbor is the other vehicle then the ego vehicle is the 
            # rear vehicle and should be labeled as a rear-end-rear collision
            if neigh.ind == other_idx
                collision_type = 3
            else
                # otherwise label as a rear-end-fore collision
                collision_type = 2
            end
        end
    end

    return collision_type
end

"""
Description:
    - Returns whether a given vehicle index executed a hard brake, as defined 
        as decelerating at a given rate for a number of frames. The function 
        assumes that the vehicle index does not change.

Args:
    - rec: scene record with scenes to evaluate
    - roadway: roadway on which scenes take place
    - vehicle_index: of the vehicle in the scenes (assumed constant)
    - pastframe: frame in the past at which to begin eval
    - hard_brake_threshold: decel defining a hard brake
    - n_past_frames: number of past frames over which decel must have occurred

Returns:
    - boolean indicating if hard brake occurred
"""
function executed_hard_brake(rec::SceneRecord, roadway::Roadway, 
        vehicle_index::Int, pastframe::Int = 0; hard_brake_threshold = -4.,
        n_past_frames = 3)
    # check whether decelerating at suffcient rate for n_past_frames frames
    hard_brake = true
    for dt in 0:(n_past_frames - 1)
        if pastframe_inbounds(rec, pastframe - dt)
            frame_accelfs = convert(Float64, get(
                ACCFS, rec, roadway, vehicle_index, pastframe - dt))
            frame_accel = convert(Float64, get(
                ACC, rec, roadway, vehicle_index, pastframe - dt))
            if isnan(frame_accelfs)
                frame_accel = frame_accel
            elseif isnan(frame_accel)
                frame_accel = frame_accelfs
            else
                frame_accel = min(frame_accel, frame_accelfs)
            end
            if frame_accel > hard_brake_threshold
                hard_brake = false
                break
            end
        end
    end
    return hard_brake
end

"""
Overriding IDM track_longitudinal! in order to clamp accel in negative velocity
situations. 
"""
function AutomotiveDrivingModels.track_longitudinal!(
        model::IntelligentDriverModel, scene::Scene, roadway::Roadway, 
        ego_index::Int, target_index::Int)
    veh_ego = scene[ego_index]
    v = veh_ego.state.v

    if target_index > 0
        veh_target = scene[target_index]

        s_gap = get_frenet_relative_position(get_rear_center(veh_target),
                                             veh_ego.state.posF.roadind, roadway).Δs

        if s_gap > 0.0
            Δv = veh_target.state.v - v
            s_des = model.s_min + v*model.T - v*Δv / (2*sqrt(model.a_max*model.d_cmf))
            v_ratio = model.v_des > 0.0 ? (v/model.v_des) : 1.0
            model.a = model.a_max * (1.0 - v_ratio^model.δ - (s_des/s_gap)^2)
        elseif s_gap > -veh_ego.def.length
            model.a = -model.d_max
        else
            Δv = model.v_des - v
            model.a = Δv*model.k_spd
        end

        if isnan(model.a)

            warn("IDM acceleration was NaN!")
            if s_gap > 0.0
                Δv = veh_target.state.v - v
                s_des = model.s_min + v*model.T - v*Δv / (2*sqrt(model.a_max*model.d_cmf))
                println("\tΔv: ", Δv)
                println("\ts_des: ", s_des)
                println("\tv_des: ", model.v_des)
                println("\tδ: ", model.δ)
                println("\ts_gap: ", s_gap)
            elseif s_gap > -veh_ego.def.length
                println("\td_max: ", model.d_max)
            end

            model.a = 0.0
        end
    else
        # no lead vehicle, just drive to match desired speed
        Δv = model.v_des - v
        model.a = Δv*model.k_spd # predicted accel to match target speed
    end

    low = v < 0. ? 0. : -model.d_max
    model.a = clamp(model.a, low, model.a_max)
    model
end

### Behavior
# some driver models will need to have a random seed set for reproducibility
# so add a base method that does nothing
Random.srand(model::DriverModel, seed::Int) = model

# adding σ to static longitudinal 
mutable struct StaticLongitudinalDriver <: LaneFollowingDriver
    a::Float64
    σ::Float64
    StaticLongitudinalDriver(a::Float64=0.0, σ::Float64=0.0) = new(a, σ)
end
get_name(::StaticLongitudinalDriver) = "ProportionalSpeedTracker"
function Base.rand(model::StaticLongitudinalDriver)
    if isnan(model.σ) || model.σ ≤ 0.0
        return model.a
    else
        return  rand(Normal(model.a, model.σ))
    end
end

# get_driver
get_driver(model::DriverModel) = model




