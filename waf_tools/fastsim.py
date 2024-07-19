#! /usr/bin/env python
# encoding: utf-8

"""
Quick n dirty libtorch detection
"""

import os
from copy import deepcopy
from waflib.Configure import conf


def options(opt):
    opt.add_option('--fastsim', type='string', help='path to libfastsim', dest='fastsim')

@conf
def check_fastsim(conf, *k, **kw):
    def fail(msg, required):
        if required:
            conf.fatal(msg)
        conf.end_msg(msg, 'RED')
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    # OSX/Mac uses .dylib and GNU/Linux .so
    suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.fastsim:
        includes_check = [conf.options.fastsim + 'include/']
        libs_check = [conf.options.fastsim + 'lib/']
    else:
        includes_check = ['/usr/local/include', '/usr/include', '/usr/include/torch/csrc/api/include', '/usr/local/include/torch/csrc/api/include', '/opt/NOSALRO/include', '/opt/include']
        libs_check = ['/usr/local/lib', '/usr/lib/', '/opt/NOSALRO/lib', '/opt/lib']

    fastsim_include = []
    fastsim_lib = []
    fastsim_lib_path = []
    fastsim_found = False
    fastsim_include_path = []
    try:
        fastsim_include_path.append(get_directory('libfastsim', includes_check))
        fastsim_lib_path.append(get_directory('libfastsim.a', libs_check))
        fastsim_lib = ['fastsim']

        fastsim_found = True
    except:
        fastsim_found = False

    conf.env.INCLUDES_FASTSIM = None
    conf.env.LIBPATH_FASTSIM = None
    conf.start_msg('fastsim: Checking for FASTSIM')
    if fastsim_found:
        conf.end_msg(fastsim_include_path)
        conf.env.INCLUDES_FASTSIM = fastsim_include_path
        conf.env.LIBPATH_FASTSIM = fastsim_lib_path
        conf.env.LIB_FASTSIM += fastsim_lib
    else:
        conf.end_msg('FASTSIM not found.', 'RED')
