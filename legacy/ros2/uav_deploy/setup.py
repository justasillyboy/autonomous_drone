from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'uav_deploy'

setup(
    name=package_name,
    version='0.0.0',
    # find_packages sẽ tự động tìm thư mục UAV_node và main nếu có __init__.py
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Khai báo đường dẫn cho launch files
        (os.path.join('share', package_name, 'launch'),
            glob('launch/*.launch.py')),
        # Khai báo đường dẫn cho config files
        (os.path.join('share', package_name, 'config'),
            glob('config/*.yaml')),
    ],
    install_requires=[
        'setuptools',
        'ultralytics',
        'adafruit-circuitpython-bno08x'
    ],
    zip_safe=True,
    maintainer='vandong',
    maintainer_email='vandong@todo.todo',
    description='UAV ROS2 Python package',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [
            'camera_node = UAV_node.camera_node:main',
            'perception_node = UAV_node.perception_node:main',
            'mission_node = UAV_node.mission_control_node:main',
            'action_node = UAV_node.action_mavros_node:main',
            'sensor_node = UAV_node.sensor_bridge_node:main',
        ],
    },
)