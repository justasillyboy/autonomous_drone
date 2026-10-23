from setuptools import setup

PACKAGE = "uav_mission"

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
    description="mission_node: a thin ROS shell around uav_core.mission",
    license="Apache-2.0",
    entry_points={"console_scripts": [f"mission_node = {PACKAGE}.mission_node:main"]},
)
