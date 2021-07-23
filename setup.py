from setuptools import setup


with open("README.md") as readme_file:
    readme = readme_file.read()

with open("requirements.txt") as requirements_file:
    install_requires = [line.strip() for line in requirements_file]

setup_args = dict(
      name="positional-bias",
      version='0.0.2',
      description="Lib with postional bias module implemented in pytorch and jax",
      url="https://github.com/maximzubkov/positional-bias",
      author="Zubkov Maksim",
      author_email="zubkov.md@phystech.edu",
      long_description_content_type="text/markdown",
      long_description=readme,
      install_requires=install_requires,
      license="MIT",
      packages=["positional_bias"],
      keywords=["jax", "pytorch", "transformer", "linear-transformer", "positional-bias"],
      zip_safe=False
)

if __name__ == "__main__":
    setup(**setup_args)
