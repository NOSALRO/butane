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
    opt.add_option('--algevo', type='string', help='path to algevo', dest='algevo')

@conf
def check_algevo(conf, *k, **kw):
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

    if conf.options.algevo:
        includes_check = [conf.options.algevo]
    else:
        includes_check = ['/usr/local/include', '/usr/include', '/opt']

    algevo_include = []
    algevo_found = False
    algevo_include_path = None
    try:
        for dp in includes_check:
            for root, dirs, files in os.walk(dp):
                if 'algevo' in dirs:
                    algevo_include_path = f'{root}'

        for root, dirs, _ in os.walk(algevo_include_path):
            for d in dirs:
                for _, _, files in os.walk(f'{root}{d}'):
                    for f in files:
                        algevo_include.append(f'{root}{d}{f}')
        algevo_found = True
    except:
        algevo_found = False

    conf.env.INCLUDES_ALGEVO = None
    conf.start_msg('algevo: Checking for ALGEVO')
    if algevo_found:
        conf.end_msg(algevo_include_path)
        conf.env.INCLUDES_ALGEVO = algevo_include_path
    else:
        conf.end_msg('ALGEVO not found.', 'RED')