#!/usr/bin/env python
# encoding: utf-8


import os
from waflib import Utils, Logs
from waflib.Configure import conf


@conf
def check_misc(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    msg = ''
    if not required:
        msg = ' [optional]'

    includes_check = ['/usr/include', '/usr/local/include', '/usr/local/lib/python3.10/dist-packages/cmeel.prefix/include/']
    libs_check = ['/usr/lib', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib/x86_64-linux-gnu/', '/usr/lib64', '/usr/local/lib/python3.10/dist-packages/cmeel.prefix/lib/']

    try:
        conf.start_msg('Checking for Misc includes' + msg)
        dirs = []
        dirs.append(get_directory('torch_kmeans/kmeans.hpp', includes_check))
        conf.end_msg(dirs)
        dirs = list(set(dirs))
        conf.env.INCLUDES_MISC = dirs

        conf.start_msg('Checking for Misc libraries' + msg)
        lib_files = []

        lib_dirs = []
        libraries = ['torch_kmeans']
        for lib in libraries:
            lib_dir = get_directory('lib' + lib + '.a', libs_check)
            lib_dirs.append(lib_dir)
        lib_dirs = list(set(lib_dirs))
        conf.end_msg(lib_dirs)
        conf.env.LIBPATH_MISC = lib_dirs

        conf.env.LIB_TORCH_KMEANS = ['torch_kmeans']
    except:
        if required:
            conf.fatal('Not found')
        conf.end_msg('Not found', 'RED')
        return
    return 1
