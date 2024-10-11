local file = io.open("fps_log.csv", "w")
file:write("timestamp,fps\n")

function get_system_time()
    return os.date("!%Y-%m-%d %H:%M:%S UTC")
end

function log_fps()
    local timestamp = get_system_time() 
    local fps = mp.get_property_native("estimated-vf-fps")
    if fps then
        file:write(string.format("%s,%f\n", timestamp, fps))
    end
end

mp.register_event("shutdown", function() file:close() end)
mp.add_periodic_timer(1, log_fps) 