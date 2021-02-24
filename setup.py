import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="imax",
    version="0.0.1-beta1",
    author="Raphael Pisoni",
    author_email="raphael.pisoni@gmail.com",
    description="Image augmentation library for Jax",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/4rtemi5/imax",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
    ],
    packages=setuptools.find_packages(),
    python_requires='>=3.6',
    install_requires=[
        'jax',
        'jaxlib',
    ],
)
