from __future__ import absolute_import, division, print_function
# import os
# import os.path as op
from setuptools import setup, find_packages

packname = 'gmacc'
version = '2022.04.28'


with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name=packname,
    version=version,
    author='luklehma',
    author_email='luklehma@uni-potsdam.de',
    description='Support Package for the Publication "xxxx".',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="xxx",
    project_urls={
        "Ground Motion Accelerator": "xxxx",
    },
    classifiers=[
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Operating System :: POSIX',
        'Topic :: Scientific/Engineering'],
    package_dir={'': 'src'},
    packages=find_packages(where="src"),
    python_requires='>=3.6, <4',


    
    license='GPLv3',
    keywords=[
        'geophysics, xxxx'],

)


##############
# package_data={
#     packname: ['data/*']}
# cmdclass={
#     'install': CustomInstallCommand},
# 
# from setuptools.command.install import install
# class CustomInstallCommand(install):
#     def run(self):
#         install.run(self)
