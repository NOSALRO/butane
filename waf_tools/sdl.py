"""
Quick n dirty sdl detection
"""

import os, glob, types
from waflib.Configure import conf


def options(opt):
    pass

@conf
def check_sdl(conf, *k, **kw):
    def fail(msg, required):
        if required:
            conf.fatal(msg)
        conf.end_msg(msg, 'RED')

    conf.start_msg('Checking for SDL (1.2 - sdl-config)')

    try:
        conf.check_cfg(path='sdl-config', args='--cflags --libs', package='', uselib_store='SDL')
    except:
        fail('sdl2-config not found in the PATH', required)
        return
    conf.end_msg('OK')
    conf.env.DEFINES_SDL += ['USE_SDL']
    return
