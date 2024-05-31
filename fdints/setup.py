#!/usr/bin/env python

"""
setup.py file for SWIG
"""

from distutils.core import setup, Extension

include_gsl_dir = "/usr/include/"
lib_gsl_dir = "/usr/lib/x86_64-linux-gnu/"


extension_mod = Extension("_fdints", ["fdints_wrap.cxx", "fdints.cpp"],
	include_dirs=[include_gsl_dir],library_dirs=[lib_gsl_dir],libraries=["gsl"])
#ext = Extension("sl", sources = ["gsl_test.pyx"],include_dirs=[numpy.get_include(),include_gsl_dir],library_dirs=[lib_gsl_dir],libraries=["gsl"])

setup(name = "fdints", ext_modules=[extension_mod])