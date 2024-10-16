using Sockets
include("skill_model.jl")

struct SkillProgressServer
    host::String
    port::Int
    server::Sockets.TCPServer

    function SkillProgressServer(host="0.0.0.0", port=65432)
        ip = occursin(r"^\d+\.\d+\.\d+\.\d+$", host) ? IPv4(host) : getaddrinfo(host)[1].address
        server = listen(ip, port)
        new(host, port, server)
    end
end

function handle_client(conn)
    # println("Connected by ", getpeername(conn))
    try
        while true
            data = readavailable(conn)
            if isempty(data)
                break
            end
            try
                user_data = JSON3.read(String(data), Dict)
                information = JSON3.read("prior_skill.json", Dict)
                priors = Dict(
                    :skill => NormalMeanVariance(information["posterior_stats"]["skill"]...),
                    :learning_rate => Beta(information["posterior_stats"]["learning_rate"]...),
                    :difficulty => NormalMeanVariance(information["posterior_stats"]["difficulty"]...)
                )
                results = analyze_progress(user_data["performance"], priors[:skill], priors[:learning_rate], priors[:difficulty])

                write(conn, JSON3.write(results))
            catch e
                if e isa JSON3.Error
                    println("Invalid JSON received from ", getpeername(conn))
                    error_response = JSON3.write(Dict("error" => "Invalid JSON input"))
                    write(conn, error_response)
                else
                    println("Unexpected error processing data from ", getpeername(conn), ": ", e)
                    error_response = JSON3.write(Dict("error" => "Internal server error"))
                    write(conn, error_response)
                end
            end
        end
    finally
        close(conn)
    end
end

function run(server::SkillProgressServer)
    println("RxInfer is running in the background...")
    try
        while true
            conn = accept(server.server)
            @async handle_client(conn)
        end
    catch e
        if e isa InterruptException
            println("Server shutting down.")
        else
            rethrow(e)
        end
    finally
        close(server.server)
    end
end

# Usage
server = SkillProgressServer()
run(server)