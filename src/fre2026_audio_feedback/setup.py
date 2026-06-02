from glob import glob
import os

from setuptools import setup

package_name = "fre2026_audio_feedback"

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
    maintainer_email="tzimmerman@stud.hs-heilbronn.de",
    description="Audio feedback node for FRE2026 task 2 and task 3 classifications.",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "audio_feedback_node = fre2026_audio_feedback.audio_feedback_node:main",
        ],
    },
)
