
set_project("Butane")
set_version("1.0.0")

add_requires("eigen")

option("examples")
    set_default(false)
    set_showmenu(true)
    set_description("Build the examples")

target("butane")
    set_kind("headeronly")
    set_languages("c++17")
    add_packages("eigen")

    add_includedirs("/opt/libtorch/include/torch/csrc/api/include/", {public = true})
    add_includedirs("/opt/libtorch/include/", {public = true})

    add_linkdirs("/usr//lib64", {public = true})
    add_linkdirs("/opt/libtorch/lib", {public = true})

    add_links("torch", "torch_cpu", "c10", {public = true})

    add_includedirs("butane/cpp/", {public = true})

if has_config("examples") then
    for _, filepath in ipairs(os.files("examples/cpp/*.cpp")) do
        local name = path.basename(filepath)
        target(name)
            set_kind("binary")
            add_files(filepath)
            add_deps("butane") -- This pulls in all those {public = true} paths/links
            set_languages("c++17")
    end
end
