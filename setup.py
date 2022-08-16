from setuptools import setup


with open('requirements.txt', 'r', encoding='utf-8') as r:
    requirements = [line.strip() for line in r]

setup(name='rewrite_text',
      packages=['generation',
                'utils',
                'with_fairseq'],
      install_requires=requirements)

