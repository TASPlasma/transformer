from setuptools import setup, find_packages

setup(
    name='transformer',
    version='0.1.0',
    # Add a brief description of your package
    description='A description of your package',
    author='Your Name',  # Replace with your name
    author_email='your.email@example.com',  # Replace with your email
    # Replace with the URL of your repository
    url='https://github.com/TASPlasma/transformer',
    packages=find_packages(),  # Automatically find and include all packages in the project
    install_requires=[
        'absl-py==2.1.0',
        'chex==0.1.86',
        'equinox==0.11.4',
        'jax==0.4.30',
        'jaxlib==0.4.30',
        'jaxtyping==0.2.30',
        'ml-dtypes==0.4.0',
        'numpy==1.24.1',
        'opt-einsum==3.3.0',
        'optax==0.2.2',
        'scipy==1.13.1',
        'toolz==0.12.1',
        'typeguard==2.13.3',
        'typing_extensions==4.12.2'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',  # Adjust according to your project's requirements
)
