from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='baal',
    version='0.1dev',
    packages=['baal',],
    license='MIT',
    ext_modules = cythonize("baal/structures/*.pyx"),
    long_description=open('README.md').read(),
)
