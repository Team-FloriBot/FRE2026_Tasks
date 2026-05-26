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
    description="Task 4: Marker detection, global marker mapping, coverage planner, and Nav2 configuration for autonomous agrar-robot.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "marker_detector = task4.marker_detector:main",
            "global_marker_map = task4.global_marker_map:main",
            "coverage_planner = task4.coverage_planner:main",
        ],
    },
)