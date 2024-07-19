#! /usr/bin/env python
# encoding: utf-8
# Konstantinos Chatzilygeroudis - 2015

"""
Quick n dirty libtorch detection
"""

import os
from copy import deepcopy
from waflib.Configure import conf


def options(opt):
    opt.add_option('--torch', type='string', help='path to libtorch', dest='torch')

@conf
def check_torch(conf, *k, **kw):
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

    if conf.options.torch:
        includes_check = [conf.options.torch + '/include', conf.options.torch + '/include/torch/csrc/api/include']
        libs_check = [conf.options.torch + '/lib']
    else:
        includes_check = ['/usr/local/include', '/usr/include', '/usr/include/torch/csrc/api/include', '/usr/local/include/torch/csrc/api/include', '/opt/libtorch/include', '/opt/libtorch/include/torch/csrc/api/include']
        libs_check = ['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/', '/opt/libtorch/lib/']

    # libtorch requires Caffe2
    # Caffe2 requires phtread, gflags, glog, protobuf, mkl, and mkldnn (for CPU version)

    # caffe2_major = -1
    # caffe2_minor = -1
    # caffe2_patch = -1
    caffe2_include = []
    caffe2_libpath = []
    caffe2_check = list(set(includes_check + ['/usr/local/include', '/usr/include', '/opt/intel/mkl/include']))
    caffe2_lib_checks = list(set(libs_check + ['/usr/local/lib', '/opt/intel/mkl/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']))
    caffe2_found = False
    caffe2_libs = []
    try:
        # caffe2_include = [get_directory('caffe2/core/macros.h', caffe2_check)]
        caffe2_include.append(get_directory('gflags/gflags.h', caffe2_check))
        caffe2_include.append(get_directory('glog/logging.h', caffe2_check))
        caffe2_include.append(get_directory('google/protobuf/service.h', caffe2_check))
        # caffe2_include.append(get_directory('mkldnn.h', caffe2_check))
        caffe2_libpath = [get_directory('libgflags.' + suffix, caffe2_lib_checks)]
        caffe2_libpath.append(get_directory('libglog.' + suffix, caffe2_lib_checks))
        caffe2_libpath.append(get_directory('libprotobuf.' + suffix, caffe2_lib_checks))
        caffe2_libpath.append(get_directory('libdnnl.' + suffix, caffe2_lib_checks))
        caffe2_libpath.append('/usr/include/openapi/dnnl/')

        caffe2_libs = ['gflags', 'glog', 'protobuf', 'dnnl']

        # MKL is a beast by itself - assuming found outside of this file

        # NOT SURE IF WE NEED THE FOLLOWING LIBS
        # caffe2_libpath = [get_directory('lib?' + suffix, caffe2_lib_checks)]
        # libcaffe2_detectron_ops.so
        # libcaffe2_module_test_dynamic.so
        # libcaffe2_observers.so

        caffe2_include = list(set(caffe2_include))
        caffe2_libpath = list(set(caffe2_libpath))
        caffe2_found = True
    except:
        caffe2_found = False

    # libtorch requires C10 library (CPU version)
    c10_include = []
    c10_libpath = []
    c10_check = includes_check
    c10_lib_checks = libs_check #['/usr/local/lib', '/usr/local/lib64', '/usr/lib', '/usr/lib64', '/usr/lib/x86_64-linux-gnu/']
    c10_found = False
    try:
        c10_include = [get_directory('c10/core/Device.h', c10_check)]
        c10_libpath = [get_directory('libc10.' + suffix, c10_lib_checks)]

        c10_include = list(set(c10_include))
        c10_libpath = list(set(c10_libpath))
        c10_found = True
    except:
        c10_found = False

    try:
        conf.start_msg('Checking for libtorch includes')

        torch_include = []
        torch_include.append(get_directory('torch/torch.h', includes_check))
        torch_include = list(set(torch_include))
        conf.end_msg('Found in ' + torch_include[0]) # TO-DO: Add version?

        more_includes = []
        if caffe2_found:
            more_includes += caffe2_include
        if c10_found:
            more_includes += c10_include

        conf.start_msg('Checking for libtorch libs')
        torch_lib = []
        torch_lib.append(get_directory('libtorch.' + suffix, libs_check))
        torch_lib.append(get_directory('libtorch_cpu.' + suffix, libs_check)) # CPU version
        torch_lib = list(set(torch_lib))
        conf.env.INCLUDES_TORCH = torch_include + more_includes
        conf.env.LIBPATH_TORCH = torch_lib
        conf.env.LIB_TORCH = ['torch', 'torch_cpu']
        conf.end_msg(conf.env.LIB_TORCH)
        conf.start_msg('libtorch: Checking for Caffe2')
        if caffe2_found:
            conf.end_msg(caffe2_include)
            conf.env.LIBPATH_TORCH = conf.env.LIBPATH_TORCH + caffe2_libpath
            conf.env.LIB_TORCH += caffe2_libs
        else:
            conf.end_msg('Not found - Your programs may not compile', 'RED')

        conf.start_msg('libtorch: Checking for C10')
        if c10_found:
            conf.end_msg(c10_include)
            conf.env.LIBPATH_TORCH = conf.env.LIBPATH_TORCH + c10_libpath
            conf.env.LIB_TORCH.append('c10')
        else:
            conf.end_msg('Not found - Your programs may not compile', 'RED')

        # remove duplicates
        conf.env.INCLUDES_TORCH = list(set(conf.env.INCLUDES_TORCH))
        conf.env.LIBPATH_TORCH = list(set(conf.env.LIBPATH_TORCH))
    except:
        fail('Not found', required)
    return 1
