from glob import glob
import os

from setuptools import setup


package_name = "floribot_scan_mux"

setup(
    name=package_name,
    version="0.0.1",
    packages=[package_name],
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob("config/*.yaml")),
        (os.path.join("share", package_name, "launch"), glob("launch/*.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="FRE2026 Team",
    maintainer_email="fre2026@todo.todo",
    description="Fixed-profile LaserScan mux for the FloriBot front scan input.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "front_scan_mux = floribot_scan_mux.front_scan_mux:main",
        ],
    },
)
