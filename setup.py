from setuptools import setup

setup(name='cheetah_gym',
      version='0.0.1',
      install_requires=[
        'gym',
        'pybullet',
        'torch',
        'numpy',
        'opencv-python',
        'scipy',
        'stable-baselines3==1.8.0'
      ]
)
