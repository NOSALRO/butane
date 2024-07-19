#!/usr/bin/env python
# encoding: utf-8


import os
from waflib import Utils, Logs
from waflib.Configure import conf


def options(opt):
  opt.add_option('--pinocchio', type='string', help='path to pinocchio', dest='pinocchio')


@conf
def check_pinocchio(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return res[:-len(filename)-1]

    required = kw.get('required', False)

    msg = ''
    if not required:
        msg = ' [optional]'

    includes_check = ['/usr/include', '/usr/local/include', '/usr/local/lib/python3.10/dist-packages/cmeel.prefix/include/']
    libs_check = ['/usr/lib', '/usr/local/lib', '/opt/local/lib', '/sw/lib', '/lib', '/usr/lib/x86_64-linux-gnu/', '/usr/lib64', '/usr/local/lib/python3.10/dist-packages/cmeel.prefix/lib/']

    # OSX/Mac uses .dylib and GNU/Linux .so
    lib_suffix = 'dylib' if conf.env['DEST_OS'] == 'darwin' else 'so'

    if conf.options.pinocchio:
        includes_check = [conf.options.pinocchio + '/include/pinocchio/']
        libs_check = [conf.options.pinocchio + '/lib']

    try:
        conf.start_msg('Checking for Pinocchio includes' + msg)
        dirs = []
        dirs.append(get_directory('pinocchio/macros.hpp', includes_check))
        conf.end_msg(dirs)
        dirs = list(set(dirs))
        conf.env.INCLUDES_PINOCCHIO = dirs

        conf.start_msg('Checking for Pinocchio libraries' + msg)
        lib_files = []

        lib_dirs = []
        libraries = ['pinocchio']
        for lib in libraries:
            lib_dir = get_directory('lib' + lib + '.' + lib_suffix, libs_check)
            lib_dirs.append(lib_dir)
        lib_dirs = list(set(lib_dirs))
        conf.end_msg(lib_dirs)
        conf.env.LIBPATH_PINOCCHIO = lib_dirs

        conf.env.LIB_PINOCCHIO = ['pinocchio']
    except:
        if required:
            conf.fatal('Not found')
        conf.end_msg('Not found', 'RED')
        return
    return 1
