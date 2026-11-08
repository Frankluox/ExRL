"""Install ExRL and the bundled research dependencies in one editable tree."""

from setuptools import find_namespace_packages, find_packages, setup

roots = {
    "exrl": "exrl",
    "robomimic": "vendor/robomimic/robomimic",
    "lerobot": "vendor/lerobot/src/lerobot",
    "stable_baselines3": "vendor/dsrl/stable-baselines3/stable_baselines3",
    "agent": "vendor/dsrl/dppo/agent",
    "env": "vendor/dsrl/dppo/env",
    "model": "vendor/dsrl/dppo/model",
    "util": "vendor/dsrl/dppo/util",
}
packages = []
for name, directory in roots.items():
    packages.append(name)
    packages.extend(f"{name}.{sub}" for sub in find_namespace_packages(directory))

setup(
    name="exrl",
    version="0.1.0",
    description="Reinforcement learning over action-chunk execution lengths",
    python_requires=">=3.10",
    packages=packages,
    package_dir=roots,
    package_data={"stable_baselines3": ["version.txt", "py.typed"]},
    install_requires=["PyYAML>=6,<7"],
)
