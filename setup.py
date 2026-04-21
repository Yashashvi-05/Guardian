from setuptools import setup, find_packages

setup(
    name="guardian-env",
    version="0.2.0",
    description="GUARDIAN Fleet — AI Security Oversight RL Environment for OpenEnv",
    author="GUARDIAN Team",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "gymnasium>=0.29.0",
        "numpy>=1.24.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "datasets>=2.14.0",
        "trl>=0.8.0",
        "peft>=0.7.0",
        "torch>=2.0.0",
    ],
    extras_require={
        "train": ["unsloth", "bitsandbytes"],
        "dashboard": ["gradio>=4.0.0"],
        "faiss": ["faiss-cpu"],
        "all": ["unsloth", "bitsandbytes", "gradio>=4.0.0", "faiss-cpu", "umap-learn", "scikit-learn"],
    },
    entry_points={
        "console_scripts": [
            "guardian-train=guardian.training.train_grpo:main",
            "guardian-eval=guardian.training.evaluation:main",
            "guardian-dashboard=guardian.dashboard.app:main",
        ]
    },
)