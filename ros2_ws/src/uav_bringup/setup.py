from glob import glob

from setuptools import setup

PACKAGE = "uav_bringup"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
        (f"share/{PACKAGE}/launch", glob("launch/*.launch.py")),
        (f"share/{PACKAGE}/config", glob("config/*")),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Do Minh Hien",
    maintainer_email="thuyxn2006@gmail.com",
    description="Launch files and parameters for the real drone",
    license="Apache-2.0",
)
