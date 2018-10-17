import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EmoPy",
    version="0.0.4",
    author="ThoughtWorks Arts",
    author_email="info@thoughtworksarts.io",
    description="A deep neural net toolkit for emotion analysis via Facial Expression Recognition (FER)",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/thoughtworksarts/EmoPy",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Operating System :: MacOS :: MacOS X"
    ],
    python_requires='>=3.6.3,<3.7',
    install_requires=[
       'keras>=2.2.0',
       'lasagne',
       'pytest',
       'matplotlib>2.1.0',
       'numpy<=1.14.5,>=1.13.3',
       'scikit-image>=0.13.1',
       'scikit-learn>=0.19.1',
       'scikit-neuralnetwork>=0.7',
       'scipy>=0.19.1',
       'tensorflow>=1.10.1',
       'opencv-python',
       'h5py',
       'pydot',
       'graphviz',
    ]
)
