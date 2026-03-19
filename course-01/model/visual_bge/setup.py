from setuptools import setup, find_packages

print("DEBUG: find_packages() ->", find_packages())

setup(
    name="visual_bge",
    version="0.1.0",
    description="visual_bge",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "visual_bge": [
            "eva_clip/*",
            "eva_clip/**/*",
        ],
    },
    install_requires=[
            "torchvision",
            "timm",
            "einops",
            "ftfy",
    ],
    python_requires=">=3.6",
)
