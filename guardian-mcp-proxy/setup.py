from setuptools import setup, find_packages

setup(
    name="guardian-mcp-proxy",
    version="0.1.0",
    description="GUARDIAN MCP Security Proxy — drop-in transport layer that intercepts and audits every MCP tool call",
    author="NEXUS Team",
    python_requires=">=3.10",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "fastapi>=0.111.0",
        "uvicorn[standard]>=0.29.0",
        "httpx>=0.27.0",
        "pyyaml>=6.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0", "httpx>=0.27.0"],
    },
    entry_points={
        "console_scripts": [
            "guardian-mcp=guardian_mcp_proxy.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Security",
    ],
)
