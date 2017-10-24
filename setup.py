from setuptools import setup

setup(name='Riot-python',
      version='0.1',
      description='RIOT Facial Recognition Neural Net',
      url='https://git.thoughtworks.net/arts-residency/riot-python.git',
      author='',
      author_email='',
      license='',
      packages=[],
      install_requires=[
          'lasagne',
        'matplotlib>=2.1.0',
        'numpy>=1.13.3',
          'scikit-image>=0.13.1',
          'scikit-learn>=0.19.1',
          'scikit-neuralnetwork>=0.7',
          'scipy>=0.19.1'
      ],
      dependency_links=[
          'https://github.com/Lasagne/Lasagne/archive/master.zip#egg=lasagne-0.2_dev'
      ],
      zip_safe=False)