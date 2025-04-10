"""
================================================================
Setup Script for Qurry
================================================================

"""

from setuptools import setup
from setuptools_rust import Binding, RustExtension


setup(
    rust_extensions=[
        RustExtension(
            "qurry.boorust",
            "crates/boorust/Cargo.toml",
            binding=Binding.PyO3,
            optional=True,
        )
    ],
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
