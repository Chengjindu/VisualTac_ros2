from setuptools import setup
import os
from glob import glob

package_name = 'gelwedge_demo_ros2'

setup(
    name=package_name,
    version='0.0.1',
    packages=[package_name],
    data_files=[
        # Required ROS2 ament index resources
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        # Include launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),

        # Include non-Python resources (models, calibration files, compiled libs, etc.)
        (os.path.join('share', package_name), [
            os.path.join(package_name, 'transformation_matrix.npy'),
            os.path.join(package_name, 'find_marker.so'),
        ]),
    ],
    install_requires=['setuptools', 'numpy', 'opencv-python', 'rclpy'],
    zip_safe=True,
    maintainer='chengjin',
    maintainer_email='duchengjin@cvte.com',
    description='ROS2 GelWedge visual-tactile demo package',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # main tactile processing node
            'gelwedge_demo_node = gelwedge_demo_ros2.gelwedge_demo_node:main',

            # optional calibration script to run via ros2 run
            'calibrate_matrix = gelwedge_demo_ros2.transformation_matrix_calculation:main',
        ],
    },
)
