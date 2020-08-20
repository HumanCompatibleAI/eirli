from setuptools import find_packages, setup

setup(
    name="il-representations",
    version="0.0.1",
    description="Representation learning for imitation learning",
    # >=3.7.0 because that's what `imitation` requires
    python_requires=">=3.7.0",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy~=1.19.0",
        "gym[atari]==0.17.*",
        "sacred~=0.8.1",
        "torch==1.6.*",
        "torchvision==0.7.*",
        "opencv-python~=4.3.0.36",
        "pyyaml~=5.3.1",
        "sacred~=0.8.1",
        "tensorboard~=2.2.0",

        # testing/dev utils
        "pytest~=5.4.3",
        "isort~=5.0",
        "yapf~=0.30.0",
        "flake8~=3.8.3",
        "autoflake~=1.3.1",
        "pytest-flake8~=1.0.6",
        "pytest-isort~=1.1.0",

        # imitation needs special branch as of 2020-08-20
        ("imitation @ git+git://github.com/HumanCompatibleAI/imitation"
         "@image-env-changes#egg=imitation"),
        ("stable_baselines3 @ git+https://github.com/HumanCompatibleAI/stable-baselines3.git"
         "@imitation#egg=stable-baselines3"),

        # environments
        "magical @ git+https://github.com/qxcv/magical@master",
        "dm_control~=0.0.319497192",
        ("dmc2gym @ git+git://github.com/denisyarats/dmc2gym"
         "@6e34d8acf18e92f0ea0a38ecee9564bdf2549076"),
    ],
    entry_points={
        "console_scripts": [
            "run_rep_learner=il_representations.scripts.run_rep_learner:main",
            "il_train=il_representations.scripts.il_train:main",
            "il_test=il_representations.scripts.il_test:main",
        ],
    },
)
