"""Polyfill for the `imp` module removed in Python 3.12.

Required by the Ubuntu 24.04 apt package of MRtrix3, whose scripts still
call imp.find_module / imp.load_module / imp.load_source.
"""
import importlib.util
import os
import sys
import types

PY_SOURCE = 1
PY_COMPILED = 2
C_EXTENSION = 3
PKG_DIRECTORY = 5
C_BUILTIN = 6
PY_FROZEN = 7


def find_module(name, path=None):
    search = path if path is not None else sys.path
    for p in search:
        full = os.path.join(p, name)
        if os.path.isfile(full + '.py'):
            return (None, full + '.py', ('.py', 'r', PY_SOURCE))
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, '__init__.py')):
            return (None, full, ('', '', PKG_DIRECTORY))
    raise ImportError('No module named ' + repr(name))


def load_module(name, file, filename, description):
    if file is not None:
        file.close()
    suffix, mode, type_ = description
    if type_ == PKG_DIRECTORY:
        filename = os.path.join(filename, '__init__.py')
    spec = importlib.util.spec_from_file_location(name, filename)
    if spec is None:
        raise ImportError('Cannot find spec for ' + repr(name) + ' at ' + repr(filename))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_source(name, pathname, file=None):
    return load_module(name, None, pathname, ('.py', 'r', PY_SOURCE))


def new_module(name):
    return types.ModuleType(name)
