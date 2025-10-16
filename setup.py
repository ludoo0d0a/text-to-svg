#!/usr/bin/env python3.12

"""
Comprehensive setup script for text-to-SVG project
Handles authentication, model downloads, and environment setup
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download, whoami
from huggingface_hub.utils import HfHubHTTPError

class TextToSVGSetup:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.models_dir = self.project_dir / "models"
        self.venv_dir = self.project_dir / ".venv"
        
    def print_header(self, title):
        """Print a formatted header"""
        print(f"\n{'='*60}")
        print(f"🤖 {title}")
        print(f"{'='*60}")
    
    def print_step(self, step, description):
        """Print a step with emoji"""
        print(f"\n{step} {description}")
    
    def check_authentication(self):
        """Check if user is authenticated with Hugging Face"""
        try:
            user_info = whoami()
            print(f"✅ Authenticated as: {user_info['name']}")
            return True
        except Exception:
            print("❌ Not authenticated with Hugging Face")
            return False
    
    def setup_authentication(self):
        """Setup Hugging Face authentication"""
        self.print_step("🔐", "Setting up Hugging Face authentication")
        
        print("📋 To authenticate:")
        print("1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token (read access)")
        print("3. Copy the token")
        
        print(f"\n🔗 Direct link: https://huggingface.co/settings/tokens")
        
        response = input("\n❓ Do you want to run the login command now? (y/N): ").strip().lower()
        if response in ['y', 'yes']:
            try:
                print("🚀 Running: huggingface-cli login")
                result = subprocess.run([
                    sys.executable, "-m", "huggingface_hub.commands.login"
                ], capture_output=False, text=True)
                
                if result.returncode == 0:
                    print("✅ Authentication successful!")
                    return True
                else:
                    print("❌ Authentication failed")
                    return False
            except Exception as e:
                print(f"❌ Error running login: {e}")
                return False
        else:
            print("⏭️  Skipping authentication setup")
            return False
    
    def download_llama_model(self):
        """Download the Llama base model"""
        try:
            self.print_step("🦙", "Downloading Llama-3.2-3B-Instruct")
            
            base_model_dir = self.models_dir / "base_model"
            base_model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = snapshot_download(
                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                local_dir=base_model_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Llama model downloaded to: {model_path}")
            return True
            
        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                print("🔐 Authentication required for Llama model")
                print("💡 You need to request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
                return False
            else:
                print(f"❌ Error downloading Llama model: {e}")
                return False
        except Exception as e:
            print(f"❌ Unexpected error: {e}")
            return False
    
    def download_public_model(self, model_choice=None):
        """Download a public model"""
        public_models = [
            ("microsoft/DialoGPT-medium", "DialoGPT Medium - Good for conversations"),
            ("gpt2", "GPT-2 - Classic text generation"),
            ("facebook/blenderbot-400M-distill", "BlenderBot - Conversational AI"),
            ("microsoft/DialoGPT-small", "DialoGPT Small - Lightweight")
        ]
        
        self.print_step("📥", "Downloading public model")
        
        print("📋 Available public models:")
        for i, (model, description) in enumerate(public_models, 1):
            print(f"   {i}. {model} - {description}")
        
        if model_choice is None:
            while True:
                choice = input("\n❓ Choose a model (1-4) or 'q' to skip: ").strip()
                
                if choice.lower() == 'q':
                    return None
                    
                try:
                    choice_idx = int(choice) - 1
                    if 0 <= choice_idx < len(public_models):
                        selected_model = public_models[choice_idx][0]
                        break
                    else:
                        print("❌ Invalid choice. Please enter 1-4 or 'q'")
                except ValueError:
                    print("❌ Invalid input. Please enter a number or 'q'")
        else:
            selected_model = public_models[model_choice - 1][0]
        
        try:
            print(f"\n📥 Downloading: {selected_model}")
            
            public_model_dir = self.models_dir / "public_model"
            public_model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = snapshot_download(
                repo_id=selected_model,
                local_dir=public_model_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Public model downloaded to: {model_path}")
            return selected_model
            
        except Exception as e:
            print(f"❌ Error downloading public model: {e}")
            return None
    
    def download_adapter(self):
        """Download the LoRA adapter"""
        try:
            self.print_step("🔧", "Downloading LoRA adapter")
            
            adapter_path = snapshot_download(
                repo_id="mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg",
                local_dir=self.models_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Adapter downloaded to: {adapter_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading adapter: {e}")
            return False
    
    def create_usage_scripts(self, has_llama=False, public_model=None):
        """Create usage scripts based on available models"""
        self.print_step("📝", "Creating usage scripts")
        
        if has_llama:
            # Create Llama + LoRA script
            llama_script = '''#!/usr/bin/env python3.12

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

def generate_svg_with_llama(prompt: str):
    """Generate SVG using Llama model with LoRA adapter"""
    
    print("🔄 Loading Llama model with LoRA adapter...")
    
    # Load base model
    tokenizer = AutoTokenizer.from_pretrained("./models/base_model")
    base_model = AutoModelForCausalLM.from_pretrained(
        "./models/base_model",
        dtype=torch.float16,
        device_map="auto"
    )
    
    # Load LoRA adapter
    model = PeftModel.from_pretrained(base_model, "./models")
    
    # Generate
    inputs = tokenizer(prompt, return_tensors="pt")
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=True,
            top_p=0.95,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    prompt = input("Enter your text prompt: ") or "draw a simple star in SVG"
    result = generate_svg_with_llama(prompt)
    print("Generated SVG:")
    print("-" * 50)
    print(result)
    print("-" * 50)
'''
            with open("text-to-svg-llama.py", "w") as f:
                f.write(llama_script)
            os.chmod("text-to-svg-llama.py", 0o755)
            print("✅ Created text-to-svg-llama.py")
        
        if public_model:
            # Create public model script
            public_script = f'''#!/usr/bin/env python3.12

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_text_with_public_model(prompt: str):
    """Generate text using {public_model}"""
    
    print(f"🔄 Loading {public_model}...")
    
    tokenizer = AutoTokenizer.from_pretrained("./models/public_model")
    model = AutoModelForCausalLM.from_pretrained(
        "./models/public_model",
        dtype=torch.float32,
        device_map="cpu"
    )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    inputs = {{k: v.to(model.device) for k, v in inputs.items()}}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
    
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return result

if __name__ == "__main__":
    prompt = input("Enter your text prompt: ") or "Create an SVG for a simple star:"
    result = generate_text_with_public_model(prompt)
    print("Generated text:")
    print("-" * 50)
    print(result)
    print("-" * 50)
'''
            with open("text-generator-public.py", "w") as f:
                f.write(public_script)
            os.chmod("text-generator-public.py", 0o755)
            print("✅ Created text-generator-public.py")
    
    def check_environment(self):
        """Check if virtual environment exists and has required packages"""
        self.print_step("🔍", "Checking environment")
        
        if not self.venv_dir.exists():
            print("❌ Virtual environment not found")
            return False
        
        # Check if packages are installed
        try:
            result = subprocess.run([
                self.venv_dir / "bin" / "python", "-c", 
                "import torch, transformers, safetensors, peft; print('✅ All packages installed')"
            ], capture_output=True, text=True, cwd=self.project_dir)
            
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print("❌ Some packages are missing")
                print("Install with: pip install torch transformers safetensors peft 'numpy<2'")
                return False
        except Exception as e:
            print(f"❌ Error checking environment: {e}")
            return False
    
    def run_setup(self, auto_mode=False):
        """Run the complete setup process"""
        self.print_header("Text-to-SVG Project Setup")
        
        # Check environment
        env_ok = self.check_environment()
        if not env_ok:
            print("\n💡 Please run the installation steps first:")
            print("1. python3.12 -m venv .venv")
            print("2. source .venv/bin/activate")
            print("3. pip install torch transformers safetensors peft 'numpy<2'")
            return False
        
        # Check authentication
        is_authenticated = self.check_authentication()
        
        # Try to download Llama model if authenticated
        llama_success = False
        if is_authenticated:
            llama_success = self.download_llama_model()
        
        # Download adapter (public)
        adapter_success = self.download_adapter()
        
        # Download public model as fallback
        public_model = None
        if not llama_success or auto_mode:
            if auto_mode:
                public_model = self.download_public_model(model_choice=2)  # Default to gpt2
            else:
                public_model = self.download_public_model()
        
        # Create usage scripts
        self.create_usage_scripts(has_llama=llama_success, public_model=public_model)
        
        # Final summary
        self.print_header("Setup Complete!")
        
        if llama_success and adapter_success:
            print("🎉 Llama model + LoRA adapter setup complete!")
            print("📝 Run: .venv/bin/python text-to-svg-llama.py")
            print("💡 This provides the best SVG generation quality")
        
        if public_model:
            print(f"🎉 Public model ({public_model}) setup complete!")
            print("📝 Run: .venv/bin/python text-generator-public.py")
            print("💡 This works without authentication")
        
        if not is_authenticated and not llama_success:
            print("\n🔐 To get the best results (Llama model):")
            print("1. Get token from: https://huggingface.co/settings/tokens")
            print("2. Run: python setup.py")
            print("3. Choose to authenticate when prompted")
        
        print(f"\n📁 Models directory: {self.models_dir}")
        print("📚 Check README.md for detailed usage instructions")
        
        return True

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup text-to-SVG project")
    parser.add_argument("--auto", action="store_true", 
                       help="Run in auto mode (download gpt2, skip prompts)")
    parser.add_argument("--auth-only", action="store_true",
                       help="Only setup authentication")
    parser.add_argument("--public-only", action="store_true",
                       help="Only download public models")
    
    args = parser.parse_args()
    
    setup = TextToSVGSetup()
    
    if args.auth_only:
        setup.print_header("Authentication Setup")
        setup.setup_authentication()
    elif args.public_only:
        setup.print_header("Public Model Download")
        setup.check_environment()
        public_model = setup.download_public_model(model_choice=2)  # Default to gpt2
        setup.create_usage_scripts(has_llama=False, public_model=public_model or "gpt2")
    else:
        setup.run_setup(auto_mode=args.auto)

if __name__ == "__main__":
    main()
