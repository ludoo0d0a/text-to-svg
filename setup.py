#!/usr/bin/env python3.12

"""
Comprehensive setup script for text-to-SVG project
Handles authentication, model downloads, and environment setup
"""

import os
import sys
import subprocess
from pathlib import Path
from huggingface_hub import whoami
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
                "import torch, transformers, safetensors, peft, llama_cpp; print('✅ All packages installed')"
            ], capture_output=True, text=True, cwd=self.project_dir)
            
            if result.returncode == 0:
                print(result.stdout.strip())
                return True
            else:
                print("❌ Some packages are missing")
                print("Install with: pip install torch transformers safetensors peft 'numpy<2' llama-cpp-python")
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
            print("3. pip install torch transformers safetensors peft 'numpy<2' llama-cpp-python")
            return False
        
        # Check authentication
        is_authenticated = self.check_authentication()
        
        # Final summary
        self.print_header("Setup Complete!")
        
        print("✅ Environment check passed.")
        print("🚀 You can now run the main script:")
        print("   python text-to-svg.py")
        print("\n💡 The script will offer to download any missing models automatically.")
        
        if not is_authenticated:
            print("\n🔐 For models requiring authentication (like 'llama_lora'):")
            print("   You will need to log in to Hugging Face.")
            print("   Run: huggingface-cli login")
        
        print("\n📚 Check README.md for detailed usage instructions")
        
        return True

def main():
    """Main function with command line options"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup text-to-SVG project")
    parser.add_argument("--auto", action="store_true", 
                       help="Run in auto mode (download gpt2, skip prompts)")
    
    args = parser.parse_args()
    
    setup = TextToSVGSetup()
    
    # The main purpose of setup.py is now to check the environment and guide the user.
    setup.run_setup(auto_mode=args.auto)

if __name__ == "__main__":
    main()
