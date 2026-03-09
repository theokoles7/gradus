[gradus](https://github.com/theokoles7/gradus/blob/main/README.md) / [documentation](./README.md) / setup

# Setup & Installation

## 1. Clone the Repository

```bash
git clone git@github.com:theokoles7/gradus.git && cd gradus
```

## 2. Create a Virtual/Conda Environment

```bash
# Using venv
python3 -m venv gradus-env && source gradus-env/bin/activate

# Using conda
conda create -n gradus python=3.13 -y && conda activate gradus
```

## 3. Install Dependencies

```bash
pip install -e .
```

## 4. Verify Installation

```bash
gradus version
```