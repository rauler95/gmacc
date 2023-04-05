from __future__ import absolute_import, division, print_function
# import os
# import os.path as op
from setuptools import setup, find_packages
from setuptools.command.install import install

packname = 'gmacc'
version = '2023.4.4'

packages = ['%s' % packname]
packs = find_packages(where="src")
for p in packs:
    packages.append('%s.%s' % (packname, p))


# with open("README.md", "r", encoding="utf-8") as fh:
#     long_description = fh.read()


class CustomInstallCommand(install):
    def run(self):
        install.run(self)


setup(
    name=packname,
    version=version,
    author='luklehma',
    author_email='luklehma@uni-potsdam.de',
    description='Support Package for the Publication "xxxx".',
    # long_description=long_description,
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
    # packages=find_packages(where="src"),
    packages=packages,
    package_dir={'gmacc': 'src'},
    python_requires='>=3.6, <3.8',
    license='GPLv3',
    keywords=[
        'geophysics, xxxx'],
    cmdclass={
        'install': CustomInstallCommand},

)


##############
# package_data={
#     packname: ['data/*']}
