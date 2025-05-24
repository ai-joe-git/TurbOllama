from setuptools import setup, find_packages

setup(
    name="turbollama",
    use_scm_version=True,
    setup_requires=['setuptools_scm'],
    packages=find_packages(),
    include_package_data=True,
    entry_points={
        'console_scripts': [
            'turbollama=turbollama.main:main',
        ],
    },
)
