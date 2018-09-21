import setuptools

with open("README.md","r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="EmoPy",
    version="0.0.2",
    author="ThoughtWorks Arts",
    author_email="andy@thoughtworks.io",
    description="A deep neural net toolkit for emotion analysis via Facial Expression Recognition (FER)",
    long_description=long_description,
    long_description_content_type = "text/markdown",
    url="https://github.com/thoughtworksarts/EmoPy",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
    python_requires='>=3.6.3,<3.7',
    install_requires=[
       'keras>=2.2.0',
       'lasagne',
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
