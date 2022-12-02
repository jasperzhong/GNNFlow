import os
import subprocess
import sys

import torch.utils
from setuptools import Extension, find_packages, setup
from setuptools.command.build_ext import build_ext

gnnflow_lib = Extension(
    "libgnnflow", sources=[]
)

curdir = os.path.dirname(os.path.abspath(__file__))


def get_cmake_bin():
    cmake_bin = "cmake"
    try:
        subprocess.check_output([cmake_bin, "--version"])
    except OSError:
        raise RuntimeError(
            "Cannot find CMake executable. "
            "Please install CMake and try again."
        )
    return cmake_bin


class CustomBuildExt(build_ext):
    def build_extensions(self):
        cmake_bin = get_cmake_bin()

        debug = os.environ.get("DEBUG", "0")
        config = 'Debug' if debug == "1" else 'Release'
        print("Building with CMake config: {}".format(config))

        ext_name = self.extensions[0].name
        build_dir = self.get_ext_fullpath(ext_name).replace(
            self.get_ext_filename(ext_name), '')
        build_dir = os.path.abspath(build_dir)

        cmake_args = [
            "-DCMAKE_BUILD_TYPE={}".format(config),
            "-DCMAKE_EXPORT_COMPILE_COMMANDS=ON",
            "-DPYTHON_EXECUTABLE:FILEPATH={}".format(sys.executable),
            "-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={}".format(build_dir),
            "-DCMAKE_PREFIX_PATH={}".format(torch.utils.cmake_prefix_path),
        ]

        cmake_build_args = ['--config', config, '--', '-j']

        if not os.path.exists(self.build_lib):
            os.makedirs(self.build_lib)

        os.chdir("build")

        try:
            subprocess.check_call([cmake_bin, "..", *cmake_args])
            subprocess.check_call(
                [cmake_bin, "--build", ".", *cmake_build_args])
        except subprocess.CalledProcessError as e:
            raise RuntimeError("CMake build failed") from e

        os.chdir(curdir)


require_list = ["torch", "numpy"]

test_require_list = ["unittest", "parameterized"]

setup(
    name="gnnflow",
    version="0.0.1",
    description="A comprehensive framework for dynamic graph neural networks",
    license="Apache 2.0",
    url="https://github.com/jasperzhong/GNNFlow",
    packages=find_packages(exclude=("tests")),
    ext_modules=[gnnflow_lib],
    cmdclass={"build_ext": CustomBuildExt},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent"
        "Topic :: Machine Learning Package"
    ],
    python_requires='>=3.6',
    install_requires=require_list,
    tests_require=test_require_list,
)
