import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="simple-mapreduce",
    version="0.0.3",
    author="Aleksandr Zuev",
    author_email="zuev08@gmail.com",
    description="Simple map-reduce implementation with no dependencies",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/File5/simple-mapreduce",
    license="MIT",
    packages=['mapreduce'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.5',
)
