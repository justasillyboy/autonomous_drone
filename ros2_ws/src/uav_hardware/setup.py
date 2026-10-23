from setuptools import setup

PACKAGE = "uav_hardware"

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
    description="action_node (FlightCommand -> MAVROS, drop servo) and bno080_node (IMU)",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            f"action_node = {PACKAGE}.action_node:main",
            f"bno080_node = {PACKAGE}.bno080_node:main",
        ]
    },
)
