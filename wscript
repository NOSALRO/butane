#!/usr/bin/env python
# encoding: utf-8


import sys
import os
import fnmatch
import glob

sys.path.insert(0, sys.path[0] + "/waf_tools")

VERSION = "1.0.0"
APPNAME = "butane"

srcdir = os.path.abspath("butane/")
blddir = os.path.abspath("build/")

from waflib.Build import BuildContext
from waflib import Logs
from waflib.Tools import waf_unit_test


def options(opt):
    opt.load("compiler_cxx")
    opt.load("compiler_c")
    opt.load("boost")
    opt.load("torch")
    opt.load("eigen")
    opt.add_option("--examples", default=False, dest='build_examples', action='store_true')


def configure(conf):

    conf.load("compiler_cxx")
    conf.load("compiler_c")
    conf.load("boost")
    conf.load("torch")
    conf.load("eigen")

    # we need pthread for video saving
    conf.check_boost(required=True)
    conf.check_torch(required=True)
    conf.check_eigen(required=True)
    conf.recurse("./submodules/kmeans-torch-cpp/")

    # We require C++17
    if conf.env.CXX_NAME in ["icc", "icpc"]:
        common_flags = "-Wall -std=c++20"
        opt_flags = " -O3 -xHost -unroll -g "
    elif conf.env.CXX_NAME in ["clang"]:
        common_flags = "-Wall -std=c++20"
        # no-stack-check required for Catalina
        opt_flags = " -O3 -g -faligned-new -fno-stack-check -Wno-narrowing"
    else:
        gcc_version = int(conf.env["CC_VERSION"][0] + conf.env["CC_VERSION"][1])
        if gcc_version < 50:
            conf.fatal("We need C++20 features. Your compiler does not support them!")
        else:
            common_flags = "-Wall -std=c++20"
            opt_flags = " -O3 -g "
            if gcc_version >= 71:
                opt_flags = opt_flags + " -faligned-new"

    all_flags = common_flags + opt_flags + ""
    conf.env["CXXFLAGS"] = conf.env["CXXFLAGS"] + all_flags.split()


def build(bld):
    # compilation of experiment
    bld.recurse("./submodules/kmeans-torch-cpp/")
    libs = "TORCH EIGEN BOOST"
    path_prefix = bld.path.abspath() + '/'

    examples = []
    export_includes = ["./butane"]

    for root, dirs, files in os.walk(bld.path.abspath() + "/examples/"):
        for f in files:
            if f.endswith("cpp"):
                examples.append(f"{root}{f}".replace(path_prefix, ''))

    if bld.options.build_examples:
        for example in examples:
            bld.program(
                features="cxx",
                install_path=None,
                source=[example],
                includes=export_includes,
                uselib=libs,
                use="torch_kmeans",
                target=example.split("/")[-1].split(".")[0],
            )

    install_files = []
    for root, dirs, files in os.walk(APPNAME):
        for f in files:
            if f.endswith(".hpp") or f.endswith(".h"):
                install_files.append(f"{root}/{f}")

    for f in install_files:
        dir_name = "/".join(f.split("/")[:-1])
        bld.install_files("${PREFIX}/include/" + dir_name, f)
