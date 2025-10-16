#!/usr/bin/env python3.12

"""
Unified Text-to-SVG Generator
Automatically detects available models and generates SVG from text prompts
"""

import os
import sys
import torch
from pathlib import Path
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download, hf_hub_download, whoami
from huggingface_hub.utils import HfHubHTTPError

# --- Model Configuration ---
# List of supported direct-access models from Hugging Face Hub, in order of preference.
SUPPORTED_DIRECT_MODELS = [
    "mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg",
    "vinoku89/svg-code-generator"
]
DEFAULT_MODEL = "vinoku89/svg-code-generator"
# List of supported GGUF models.
SUPPORTED_GGUF_MODELS = {
    "mradermacher/svg-code-generator-GGUF": "svg-code-generator-L3-8B-Instruct-Q4_K_M.gguf"
}


class TextToSVGGenerator:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.models_dir = self.project_dir / "models"
        self.default_prompt = "draw a simple star in SVG"
        
    def detect_available_models(self):
        """Detect which models are available"""
        available_models = {}
        
        # Check for Llama + LoRA setup
        base_model_path = self.models_dir / "base_model"
        adapter_files = list(self.models_dir.glob("adapter_*.json"))
        
        if base_model_path.exists() and adapter_files:
            available_models["llama_lora"] = {
                "type": "llama_lora",
                "base_path": base_model_path,
                "adapter_path": self.models_dir,
            }
        
        # Check for public model
        public_model_path = self.models_dir / "public_model"
        if public_model_path.exists():
            available_models["public"] = {
                "type": "public",
                "path": public_model_path
            }
        
        # Check for direct model access (mohannad-tazi)
        from huggingface_hub import repo_info
        for model_name in SUPPORTED_DIRECT_MODELS:
            try:
                repo_info(model_name)
                available_models[model_name] = {"type": "local", "name": model_name}
            except Exception:
                # Model not accessible, skip it
                pass

        # Check for local GGUF models
        for model_repo, filename in SUPPORTED_GGUF_MODELS.items():
            gguf_path = self.models_dir / filename
            if gguf_path.exists():
                available_models[model_repo] = {
                    "type": "gguf",
                    "path": gguf_path
                }
        return available_models
    
    def generate_with_llama_lora(self, prompt):
        """Generate using Llama model with LoRA adapter"""
        try:
            from peft import PeftModel
            
            print("🔄 Loading Llama model with LoRA adapter...")
            
            # Load base model
            base_model_path = self.models_dir / "base_model"
            tokenizer = AutoTokenizer.from_pretrained(base_model_path)
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path,
                # Use bfloat16 for better memory efficiency and numerical stability
                dtype=torch.bfloat16,
                device_map="cpu" # Explicitly load on CPU to avoid device memory issues
            )
            
            # Load LoRA adapter
            model = PeftModel.from_pretrained(base_model, self.models_dir)
            
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
            
        except Exception as e:
            print(f"❌ Error with Llama+LoRA: {e}")
            return None
    
    def generate_with_public_model(self, prompt):
        """Generate using public model (GPT-2, etc.)"""
        try:
            public_model_path = self.models_dir / "public_model"
            
            print("🔄 Loading public model...")
            
            tokenizer = AutoTokenizer.from_pretrained(public_model_path)
            model = AutoModelForCausalLM.from_pretrained(
                public_model_path,
                dtype=torch.float32,
                device_map="cpu"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
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
            
        except Exception as e:
            print(f"❌ Error with public model: {e}")
            return None
    
    def generate_with_direct_model(self, prompt, model_name):
        """Generate using a direct-access model from Hugging Face Hub"""
        try:
            print(f"🔄 Loading direct-access model: {model_name}...")
            print("💡 This is a large model and may take several minutes to load.")
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                # Use bfloat16 for better memory efficiency on modern hardware
                dtype=torch.bfloat16,
                device_map="cpu"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                with tqdm(total=200, desc="Generating SVG") as pbar:
                    outputs = model.generate(
                        **inputs,
                        max_length=200,
                        do_sample=True,
                        top_p=0.95,
                        temperature=0.7,
                        pad_token_id=tokenizer.eos_token_id
                        # The progress bar update is a simple way to show activity - REMOVED
                    )
                    pbar.update(200) # Mark as complete
            
            result = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return result
            
        except Exception as e:
            print(f"❌ Error with direct model: {e}")
            return None

    def generate_with_gguf_model(self, prompt, model_path):
        """Generate using a GGUF model with llama-cpp-python"""
        try:
            from llama_cpp import Llama

            print(f"🔄 Loading GGUF model: {model_path.name}...")
            
            # Load the GGUF model
            llm = Llama(
                model_path=str(model_path),
                n_ctx=512,  # Context window
                n_gpu_layers=-1, # Offload all layers to GPU if available
                verbose=False
            )

            # Create the prompt for the model
            full_prompt = f"USER: {prompt}\nASSISTANT:"

            print("🤖 Generating SVG with GGUF model...")
            output = llm(
                full_prompt,
                max_tokens=256,
                stop=["USER:", "\n"],
                temperature=0.7,
                echo=False # Don't echo the prompt in the output
            )
            
            result = output['choices'][0]['text'].strip()
            return result

        except ImportError:
            print("❌ GGUF model requires 'llama-cpp-python'. Please install it.")
            return None
        except Exception as e:
            print(f"❌ Error with GGUF model: {e}")
            return None

    def download_model(self, model_key):
        """Downloads the specified model if it's known."""
        print(f"💡 Model '{model_key}' not found locally.")
        response = input(f"❓ Would you like to download it now? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("⏭️  Skipping download.")
            return False

        if model_key == "llama_lora":
            # Requires both base model and adapter
            print("\n--- Llama + LoRA Setup ---")
            # Check auth first
            try:
                whoami()
                print("✅ Authenticated with Hugging Face.")
                base_success = self._download_llama_base_model()
                adapter_success = self._download_lora_adapter()
                return base_success and adapter_success
            except Exception:
                print("❌ Authentication with Hugging Face is required for the Llama base model.")
                print("💡 Please run 'huggingface-cli login' and try again.")
                return False
        elif model_key in SUPPORTED_GGUF_MODELS:
            return self._download_gguf_model(model_key)
        elif model_key == "public":
            # For simplicity, we'll download a default public model (gpt2)
            return self._download_public_model()
        else:
            print(f"❌ Download logic for model '{model_key}' is not implemented.")
            return False

    def _download_llama_base_model(self):
        """Download the Llama base model"""
        try:
            print("\n🦙 Downloading Llama-3.2-3B-Instruct base model...")
            base_model_dir = self.models_dir / "base_model"
            base_model_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id="meta-llama/Llama-3.2-3B-Instruct",
                local_dir=base_model_dir,
                local_dir_use_symlinks=False
            )
            print("✅ Llama base model downloaded successfully.")
            return True
        except HfHubHTTPError as e:
            if "401" in str(e) or "403" in str(e):
                print("🔐 Authentication required for Llama model.")
                print("💡 You may need to request access at: https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct")
            else:
                print(f"❌ HTTP Error downloading Llama model: {e}")
            return False
        except Exception as e:
            print(f"❌ Unexpected error during Llama download: {e}")
            return False

    def _download_lora_adapter(self):
        """Download the LoRA adapter"""
        try:
            print("\n🔧 Downloading LoRA adapter...")
            snapshot_download(
                repo_id="mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg",
                local_dir=self.models_dir,
                local_dir_use_symlinks=False
            )
            print("✅ LoRA adapter downloaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Error downloading adapter: {e}")
            return False

    def _download_gguf_model(self, model_key):
        """Download a GGUF model"""
        try:
            print(f"\n🧠 Downloading GGUF model: {model_key}...")
            repo_id = model_key
            filename = SUPPORTED_GGUF_MODELS[model_key]
            
            self.models_dir.mkdir(parents=True, exist_ok=True)
            
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                local_dir=self.models_dir,
            )
            print(f"✅ GGUF model downloaded successfully.")
            return True
        except Exception as e:
            print(f"❌ Error downloading GGUF model: {e}")
            return False

    def _download_public_model(self):
        """Download a default public model (gpt2)"""
        try:
            model_repo = "gpt2"
            print(f"\n📥 Downloading public model: {model_repo}...")
            public_model_dir = self.models_dir / "public_model"
            public_model_dir.mkdir(parents=True, exist_ok=True)
            
            snapshot_download(
                repo_id=model_repo,
                local_dir=public_model_dir,
                local_dir_use_symlinks=False
            )
            
            print(f"✅ Public model '{model_repo}' downloaded successfully.")
            return True
            
        except Exception as e:
            print(f"❌ Error downloading public model: {e}")
            return False
    
    def generate_svg(self, prompt=None, model_name=None):
        """Generate SVG from text prompt using the best available model"""
        
        if prompt is None:
            prompt = self.default_prompt
        
        print(f"🎯 Generating SVG for: '{prompt}'")
        print("=" * 60)
        
        # Detect available models
        available_models = self.detect_available_models()
        
        if not available_models:
            print("❌ No models found!")
            print("💡 Run: python setup.py --auto")
            return None
        
        print(f"📋 Available models: {list(available_models.keys())}")
        
        # Determine which model to use
        target_model_key = model_name
        if target_model_key is None:
            # Default to vinoku89 if no model is specified
            target_model_key = DEFAULT_MODEL
            print(f"💡 No model specified, using default: {target_model_key}")

        if target_model_key not in available_models:
            print(f"❌ Specified model '{target_model_key}' is not available.")
            print(f"💡 Please choose from: {list(available_models.keys())}")
            return None

        # Use the selected model
        model_info = available_models[target_model_key]
        print(f"\n🔄 Using model: {target_model_key}")
        
        result = None
        if model_info["type"] == "llama_lora":
            result = self.generate_with_llama_lora(prompt)
        elif model_info["type"] == "public":
            result = self.generate_with_public_model(prompt)
        elif model_info["type"] == "local":
            result = self.generate_with_direct_model(prompt, model_name=model_info["name"])
        elif model_info["type"] == "gguf":
            result = self.generate_with_gguf_model(prompt, model_path=model_info["path"])
        
        if result:
            print(f"✅ Generated using {target_model_key} model")
            return result
        else:
            print(f"❌ Model {target_model_key} failed to generate output.")
            return None
    
def print_help(self):
        print("🎨 Text-to-SVG Generator")
        print("=" * 30)
        print()
        print("Usage:")
        print("  python text-to-svg.py [options] ['your prompt']")
        print()
        print("Options:")
        print("  --model <model_name>   Specify which model to use (e.g., 'llama_lora').")
        print("  --help, -h             Show this help message.")
        print()
        print("Examples:")
        print("  python text-to-svg.py                                  # Use default prompt and model")
        print("  python text-to-svg.py 'a blue square'                  # Use custom prompt and default model")
        print("  python text-to-svg.py --model llama_lora 'a red star'  # Use a specific model")
        print()
        print("Default prompt:", self.default_prompt)
        print("Default model:", "vinoku89/svg-code-generator")
        print()
        print("Available models will be detected automatically.")
        print("Run 'python setup.py' to download models if none are found.")

def main():
    """Main function with argument parsing"""
    import argparse
    parser = argparse.ArgumentParser(
        description="Unified Text-to-SVG Generator",
        add_help=False # We use a custom help message
    )
    parser.add_argument(
        'prompt', 
        nargs='*', 
        help="The text prompt for SVG generation."
    )
    parser.add_argument(
        '--model', 
        type=str, 
        default=None, 
        help="Specify the model to use."
    )
    parser.add_argument(
        '--help', '-h', 
        action='store_true', 
        help="Show help message."
    )
    
    args = parser.parse_args()
    generator = TextToSVGGenerator()
    
    if args.help:
        generator.print_help()
        return

    # Join prompt arguments into a single string
    prompt_text = " ".join(args.prompt) if args.prompt else None
    
    # Generate SVG
    result = generator.generate_svg(prompt=prompt_text, model_name=args.model)
    
    if result:
        print("\n" + "=" * 60)
        print("🎨 Generated SVG:")
        print("=" * 60)
        print(result)
        print("=" * 60)
        
        # Try to extract just the SVG part
        if "<svg" in result.lower():
            svg_start = result.lower().find("<svg")
            svg_end = result.lower().rfind("</svg>") + 6
            if svg_end > svg_start:
                svg_content = result[svg_start:svg_end]
                print("\n📄 Extracted SVG:")
                print("-" * 30)
                print(svg_content)
                print("-" * 30)

                # Save the extracted SVG to a file
                output_path = generator.project_dir / "outputs" / "output.svg"
                try:
                    output_path.write_text(svg_content, encoding="utf-8")
                    print(f"✅ SVG saved to: {output_path}")
                except Exception as e:
                    print(f"❌ Could not save SVG file: {e}")
    else:
        print("\n❌ Generation failed. Check the error messages above.")
        print("💡 Try running: python setup.py")

if __name__ == "__main__":
    main()
