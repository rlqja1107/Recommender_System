
from distutils.core import setup
from distutils.core import Extension, setup
from Cython.Build import cythonize
import numpy

"""
setup
"""
ext=".pyx"
extension=[]
extension.append(Extension(name="CorMvg",sources=["CorMvg/CorMvg"+ext],include_dirs=[numpy.get_include()]))
extension.append(Extension(name="SVD",sources=["SVD/SVD"+ext],include_dirs=[numpy.get_include()]))
extension.append(Extension(name="SVD_pp",sources=["SVD_pp/SVD_pp"+ext],include_dirs=[numpy.get_include()]))
extension.append(Extension(name="Integrated_Model",sources=["Integrated_Model"+ext],include_dirs=[numpy.get_include()]))
extension.append(Extension(name="BPR_MF",sources=["BPR/BPR_MF"+ext],include_dirs=[numpy.get_include()]))
extension.append(Extension(name="algo_common_func",sources=["algo_common_func"+ext],include_dirs=[numpy.get_include()]))
setup(ext_modules=cythonize(extension))





