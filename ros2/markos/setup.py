from setuptools import find_packages, setup

package_name = 'markos'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    package_data={'': ['py.typed']},
    install_requires=['setuptools', 'pyserial'],
    zip_safe=True,
    maintainer='Konstantinos Speranski',
    maintainer_email='konstantinossperanski@gmail.com',
    description='Bridge between ROS 2 and the Thor arm firmware, with the reach planner and a firmware simulator',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'thor_bridge = markos.bridge_node:main',
        ],
    },
)
