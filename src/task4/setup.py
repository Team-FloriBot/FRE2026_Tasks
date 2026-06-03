from setuptools import setup
from glob import glob
import os

package_name = "task4"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FRE2026 Team",
    maintainer_email="[EMAIL]",
    description="Task 4: Coverage Path Planning for Autonomous Lawn Mower",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "coverage_planner = task4.coverage_planner:main",
        ],
    },
)