export 
    DelayedDriver,
    get_name,
    action_context
    set_desired_speed!,
    observe!,
    rand,
    pdf,
    logpdf

import AutomotiveDrivingModels: 
    LongitudinalDriverModel, 
    get_name,
    action_context
    set_desired_speed!,
    observe!

type DelayedDriver <: DriverModel{LatLonAccel, ActionContext}
    driver::DriverModel
    rec::SceneRecord
    reaction_time::Float64 # reaction time (time delay in responding) [s]
    pastframe::Int # index into the record of the scene to observe

    function DelayedDriver(driver::DriverModel; reaction_time::Float64 = 0.5)
        Δt = get_timestep(driver.action_context)
        n_scenes = Int(ceil(reaction_time / Δt)) + 1
        rec = SceneRecord(n_scenes, Δt)
        new(driver, rec, reaction_time, -(n_scenes - 1))
    end
end

get_name(::DelayedDriver) = "DelayedDriver"
action_context(driver::DelayedDriver) = action_context(driver.driver)
set_desired_speed!(driver::DelayedDriver, v_des::Float64) = set_desired_speed!(
    driver.driver, v_des)
get_driver(driver::DelayedDriver) = get_driver(driver.driver)
function observe!(driver::DelayedDriver, scene::Scene, roadway::Roadway, egoid::Int)
    update!(driver.rec, scene)
    # internal driver begins observing after reaction time has passed
    if driver.rec.nscenes == length(driver.rec.scenes)
        pastscene = get_scene(driver.rec, driver.pastframe)
        observe!(driver.driver, pastscene, roadway, egoid)
    end
end
function Base.rand(driver::DelayedDriver)
    # return 0 acceleration prior to observing anything
    if driver.rec.nscenes == length(driver.rec.scenes)
        return rand(driver.driver)
    else
        return LatLonAccel(0.,0.)
    end
end
Distributions.pdf(driver::DelayedDriver, a::LatLonAccel) = pdf(driver.driver, a)
Distributions.logpdf(driver::DelayedDriver, a::LatLonAccel) = logpdf(driver.driver, a)