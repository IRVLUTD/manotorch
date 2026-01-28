from setuptools import setup
import warnings

DEPENDENCY_PACKAGE_NAMES = [
    "matplotlib",
    "torch",
    "tqdm",
    "numpy",
    "opencv-python",
    "scipy",
    "trimesh[recommend]",
    "pyvista[all]",
    "deprecation",
    "open3d",
    # "chumpy @ git+https://github.com/JWRoboticsVision/chumpy.git",
]


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn(
            "Missing dependencies: {}. We recommend you follow "
            "the installation instructions at "
            "https://github.com/IRVLUTD/manotorch#installation".format(missing_dependencies)
        )


# with open("README.md", "r") as fh:
#     long_description = fh.read()

check_dependencies()

setup(
    name="manotorch",
    version="0.0.3",
    author="Jikai Wang",
    author_email="jikai.wang@utdallas.edu",
    packages=["manotorch", "mano"],
    python_requires=">=3.10.0",
    description="MANO pyTORCH",
    # long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/IRVLUTD/manotorch",
    install_requires=DEPENDENCY_PACKAGE_NAMES,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU GENERAL PUBLIC LICENSE",
        "Operating System :: OS Independent",
    ],
)
