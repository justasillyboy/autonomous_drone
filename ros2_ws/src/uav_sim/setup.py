from setuptools import setup

PACKAGE = "uav_sim"

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
    description="Competition field, independent referee, ball dropper and moving target",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            f"make_world = {PACKAGE}.world:main",
            f"sim_referee = {PACKAGE}.referee_node:main",
            f"ball_dropper = {PACKAGE}.ball_dropper_node:main",
            f"target_mover = {PACKAGE}.target_mover_node:main",
            f"sim_operator = {PACKAGE}.operator_node:main",
            f"direction_check = {PACKAGE}.direction_check_node:main",
        ]
    },
)
