from setuptools import setup
from codecs import open
from os import path


from lib_main import __version__


ext_modules = []

here = path.abspath(path.dirname(__file__))
requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))


setup(name='lib_main',
      version=__version__,
      description='Repository for you work',
      author='Your Name',
      author_email='Your email',
      packages=['lib_main'],
      install_requires=requires_list,
      )
