import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'autonoms_car'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'torch', 'numpy'],
    zip_safe=True,
    maintainer='dhana',
    maintainer_email='dhana@todo.todo',
    description='ROS 2 and Isaac Sim AI Obstacle Avoidance for Differential-Drive Toy Car',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'rule_based_teacher = autonoms_car.rule_based_teacher:main',
            'dataset_collector = autonoms_car.dataset_collector:main',
            'ai_controller = autonoms_car.ai_controller_node:main',
            'mock_lidar = autonoms_car.mock_lidar_node:main',
            'train_mlp = autonoms_car.train:main',
        ],
    },
)
