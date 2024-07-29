#!/usr/bin/env python
# encoding: utf-8


import os
from waflib import Utils, Logs
from waflib.Configure import conf

def options(opt):
    opt.add_option('--torch_kmeans', type='string', help='path to torch_kmeans', dest='torch_kmeans')

@conf
def check_torch_kmeans(conf, *k, **kw):
    def get_directory(filename, dirs):
        res = conf.find_file(filename, dirs)
        return './' + res[:-len(filename)-1]

    required = kw.get('required', False)

    msg = ''
    if not required:
        msg = ' [optional]'

    if conf.options.torch_kmeans:
        includes_check = [conf.options.torch_kmeans + '/include', conf.options.butane]
        lib_check = [conf.options.torch_kmeans + '/lib']
    else:
        includes_check = ['/usr/include', 'submodules/kmeans-torch-cpp/src/', '/usr/local/include/torch/csrc/api/include', '/opt/libtorch/include', '/opt/libtorch/include/torch/csrc/api/include']
        libs_check = ['/usr/local/lib', '/usr/lib/', '/opt/NOSALRO/lib', '/opt/lib', 'build/submodules/kmeans-torch-cpp/']

    try:
        conf.start_msg('Checking for TORCH_KMEANS includes' + msg)
        dirs = []
        dirs.append(get_directory('kmeans/kmeans.hpp', includes_check))
        conf.end_msg(dirs)
        dirs = list(set(dirs))
        conf.env.INCLUDES_TORCH_KMEANS = dirs

        # lib_path = []
        # lib_path.append(get_directory('libtorch_kmeans.a', libs_check))
        # conf.env.LIBPATH_TORCH_KMEANS = lib_path
        # conf.env.LIB_TORCH_KMEANS = ['torch_kmeans']
    except:
        if required:
            conf.fatal('Not found')
        conf.end_msg('Not found', 'RED')
        return
    return 1
