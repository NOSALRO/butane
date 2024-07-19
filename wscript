#!/usr/bin/env python
# encoding: utf-8


import sys
import os
import fnmatch
import glob

sys.path.insert(0, sys.path[0] + "/waf_tools")

VERSION = "1.0.0"
APPNAME = "model_factory"

srcdir = "."
blddir = "build"

from waflib.Build import BuildContext
from waflib import Logs
from waflib.Tools import waf_unit_test


def options(opt):
    opt.load("compiler_cxx")
    opt.load("compiler_c")
    opt.load('boost')
    opt.load("torch")
    opt.load("algevo")
    opt.load("eigen")
    opt.load("opencv")
    opt.load("tbb")
    opt.load("misc")
    opt.add_option("--tests", action="store_true", help="Compile Tests", dest="tests")


def configure(conf):

    conf.get_env()["BUILD_GRAPHIC"] = False

    conf.load("compiler_cxx")
    conf.load("compiler_c")
    conf.load('boost')
    conf.load("torch")
    conf.load("eigen")
    conf.load("opencv")
    conf.load("tbb")

    # we need pthread for video saving
    # conf.check(features='cxx cxxprogram', lib=['pthread'], uselib_store='PTHREAD')
    conf.check_boost(required=True)
    conf.check_torch(required=True)
    conf.check_eigen(required=True)
    conf.check_opencv(required=False)
    conf.check_tbb(required=False)
    conf.check_misc(required=False)

    conf.check(features="cxx cxxprogram", lib=["pthread"], uselib_store="PTHREAD")

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
    print(conf.env["CXXFLAGS"])


def build(bld):
    # compilation of experiment
    libs = "TORCH EIGEN FASTSIM OPENCV TBB BOOST"

    experiments, tests, srcs = [], [], []

    for root, dirs, files in os.walk("./examples/"):
        for f in files:
            if f.endswith("cpp"):
                experiments.append(f"{root}/{f}")

    # for root, dirs, files in os.walk("./tests/"):
    #     for f in files:
    #         if f.endswith("cpp"):
    #             tests.append(f"{root}{f}")

    for root, dirs, files in os.walk("./src/"):
        for f in files:
            if f.endswith("cpp"):
                srcs.append(f"{root}/{f}")

    # First create lib
    # bld.program(
    #     features="cxxstlib",
    #     source=srcs,
    #     includes="./src/",
    #     uselib=libs,
    #     defines=[],
    #     target="modelfactory",
    # )

    # Then compile rest (experiments)
    for experiment in experiments:
        bld.program(
            features="cxx",
            install_path=None,
            source=[experiment],
            includes="./src/model_factory",
            uselib=libs,
            # use="modelfactory",
            # target="/".join(experiment.split("/")[2:]).split(".")[0],
            target=experiment.split("/")[-1].split('.')[0]
        )
    # if bld.options.tests:
    #     # (tests)
    #     for test in tests:
    #         bld.program(
    #             features="cxx",
    #             install_path=None,
    #             source=[test],
    #             includes="./",
    #             uselib=libs,
    #             use="modelfactory",
    #             target=test.split("/")[-1].split('.')[0]
    #         )
