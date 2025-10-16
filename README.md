# Text-to-SVG Generator

A Python script that generates SVG code from text prompts using machine learning models.

## Prerequisites

- macOS (tested on macOS 14+)
- Homebrew package manager
- Python 3.12 (will be installed automatically)

## Installation

### 1. Clone or Download the Project

```bash
cd /path/to/your/projects
# If you have the project files, navigate to the directory
cd text-to-svg
```

### 2. Install Python 3.12

The script requires Python 3.12. Install it using Homebrew:

```bash
brew install python@3.12
```

### 3. Create Virtual Environment

Create a virtual environment to isolate dependencies:

```bash
python3.12 -m venv .venv
```

### 4. Activate Virtual Environment

**Option A: Manual activation (Recommended)**
```bash
source .venv/bin/activate
```

**Option B: Using the unified activation script**
```bash
./activate.sh                    # Show status and instructions
./activate.sh --activate         # Try automatic activation
./activate.sh --help             # Show help
```

### 5. Install Dependencies

Install the required Python packages:

```bash
pip install torch transformers safetensors
pip install "numpy<2"  # Fix compatibility issue
pip install peft  # For LoRA adapters
```

### 6. Download Models and Setup

The setup script will check your environment and guide you on authenticating with Hugging Face if needed.

```bash
python setup.py
```

**Option B: Auto setup (downloads GPT-2 automatically)**
```bash
python setup.py --auto
```

**Option C: Public models only (no authentication)**
```bash
python setup.py --public-only
```

**Option D: Authentication only**
```bash
python setup.py --auth-only
```

**All functionality is now in the unified setup.py script above.**

## Usage

### Running the Script

**Method 1: Default prompt (Recommended)**
```bash
.venv/bin/python text-to-svg.py
```

**Method 2: Custom prompt**
```bash
.venv/bin/python text-to-svg.py "draw a red circle with blue border"
```

**Method 3: Using the activation script**
```bash
./activate.sh --activate
python text-to-svg.py
```

**Method 4: Get help**
```bash
.venv/bin/python text-to-svg.py --help
```

### Customizing the Script

The unified `text-to-svg.py` script automatically detects available models and uses the best one. You can:

- **Use default prompt**: Run without arguments
- **Use custom prompt**: Pass your prompt as an argument
- **Modify default prompt**: Edit the `default_prompt` variable in the script

Example:
```bash
.venv/bin/python text-to-svg.py "draw a red circle with a blue border"
```

## Project Structure

```
text-to-svg/
├── .venv/                         # Virtual environment (Python 3.12)
├── models/                        # Downloaded models directory
│   ├── base_model/               # Base Llama model (if authenticated)
│   ├── public_model/             # Public model (GPT-2, etc.)
│   └── adapter_config.json       # LoRA adapter files
├── setup.py                      # 🔥 Unified setup script
├── text-to-svg.py                # 🎨 Main SVG generator script
├── activate.sh                   # Unified activation script
├── quick-start.sh                # Quick reference guide
└── README.md                     # This file
```

## Dependencies

- **torch**: PyTorch deep learning framework
- **transformers**: Hugging Face transformers library
- **safetensors**: Safe tensor serialization
- **numpy**: Numerical computing (version < 2 for compatibility)
- **peft**: Parameter-Efficient Fine-Tuning library for LoRA adapters
- **huggingface_hub**: Hugging Face model hub interface

## Troubleshooting

### Common Issues

1. **"command not found: python" or activation not working**
   - Use: `source .venv/bin/activate`
   - Or try: `./activate.sh --activate`
   - Verify with: `which python` and `python --version`
   - For direct execution: `.venv/bin/python script_name.py`

2. **Authentication required error (401/403)**
   - Llama models require Hugging Face authentication
   - Run: `python setup.py --auth-only`
   - Or use public models: `python setup.py --public-only`

3. **Model not found error**
   - Use the unified setup: `python setup.py`
   - It automatically handles authentication and provides alternatives

4. **MPS device errors on Apple Silicon**
   - The scripts automatically use CPU to avoid MPS issues
   - This is normal and doesn't affect functionality

5. **NumPy compatibility warning**
   - This is normal and doesn't affect functionality
   - The script automatically uses NumPy < 2 for compatibility

### Verification

Test that everything is working:

```bash
.venv/bin/python -c "import torch; import transformers; print('✅ All packages working!')"
```

## Environment Management

### Activating the Environment
```bash
./activate_env.sh
```

### Deactivating the Environment
```bash
deactivate
```

### Checking Python Version
```bash
.venv/bin/python --version
# Should output: Python 3.12.12
```

## Model Information

The script currently uses a custom text-to-SVG model. You may need to:

1. Verify the model exists on Hugging Face
2. Use an alternative model compatible with transformers
3. Modify the model loading code for your specific use case

## License

This project is for educational and experimental purposes.

## Support

If you encounter issues:

1. Check that Python 3.12 is installed correctly
2. Verify all dependencies are installed in the virtual environment
3. Ensure you're using the correct execution command
4. Check the model availability and compatibility
