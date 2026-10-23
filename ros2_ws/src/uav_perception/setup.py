from setuptools import setup

PACKAGE = "uav_perception"

setup(
    name=PACKAGE,
    version="0.1.0",
    packages=[PACKAGE],
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{PACKAGE}"]),
        (f"share/{PACKAGE}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    extras_require={"test": ["pytest"]},
    zip_safe=True,
    maintainer="Do Minh Hien",
    maintainer_email="thuyxn2006@gmail.com",
    description="perception_node: camera frames -> segmentation backend -> uav_core Detections",
    license="Apache-2.0",
    entry_points={"console_scripts": [f"perception_node = {PACKAGE}.perception_node:main"]},
)
