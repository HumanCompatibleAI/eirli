from setuptools import find_packages, setup

setup(
    name="il-representations",
    version="0.0.1",
    description="Representation learning for imitation learning (NO DEPENDENCIES!)",
    # >=3.7.0 because that's what `imitation` requires
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    # Dependencies are in requirements.txt, not setup.py; install
    # requirements.txt separately. This setup.py is *JUST* for EIRLI code.
    install_requires=[],
)
