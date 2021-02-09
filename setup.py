from setuptools import find_packages, setup

setup(
    name="il-representations",
    version="0.0.1",
    description="Representation learning for imitation learning",
    # >=3.7.0 because that's what `imitation` requires
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    # FIXME(sam): move from requirements.txt to setup.py once merge is done
    install_requires=[],
    # FIXME(sam): keeping this as reminder to add all experiment scripts as
    # console_scripts
    # entry_points={
    #     "console_scripts": [],
    # },
)
