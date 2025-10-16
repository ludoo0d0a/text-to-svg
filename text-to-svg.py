#!/usr/bin/env python3.12

"""
Unified Text-to-SVG Generator
Automatically detects available models and generates SVG from text prompts
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Model Configuration ---
# List of supported direct-access models from Hugging Face Hub, in order of preference.
SUPPORTED_DIRECT_MODELS = [
    "mohannad-tazi/Llama-3.1-8B-Instruct-text-to-svg",
    "vinoku89/svg-code-generator",
]

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
                dtype=torch.float16,
                device_map="auto"
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
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.float32,
                device_map="cpu"
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            inputs = tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
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
            print(f"❌ Error with direct model: {e}")
            return None
    
    def generate_svg(self, prompt=None):
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
        
        # Try models in order of preference
        # Local models first, then direct-access models from the constant list
        model_order = ["llama_lora", "public"] + SUPPORTED_DIRECT_MODELS
        
        for model_key in model_order:
            if model_key in available_models:
                model_info = available_models[model_key]
                print(f"\n🔄 Trying {model_key} model...")
                
                result = None
                if model_info["type"] == "llama_lora":
                    result = self.generate_with_llama_lora(prompt)
                elif model_info["type"] == "public":
                    result = self.generate_with_public_model(prompt)
                elif model_info["type"] == "local":
                    result = self.generate_with_direct_model(prompt, model_name=model_info["name"])
                else:
                    continue
                
                if result:
                    print(f"✅ Generated using {model_key} model")
                    return result
        
        print("❌ All models failed to generate output")
        return None
    
    def print_help(self):
        """Print usage information"""
        print("🎨 Text-to-SVG Generator")
        print("=" * 30)
        print()
        print("Usage:")
        print("  python text-to-svg.py                    # Use default prompt")
        print("  python text-to-svg.py 'your prompt'      # Use custom prompt")
        print("  python text-to-svg.py --help             # Show this help")
        print()
        print("Default prompt:", self.default_prompt)
        print()
        print("Available models will be detected automatically.")
        print("Run 'python setup.py --auto' to download models if none are found.")

def main():
    """Main function"""
    generator = TextToSVGGenerator()
    
    # Handle command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] in ["--help", "-h", "help"]:
            generator.print_help()
            return
        
        # Use provided prompt
        prompt = " ".join(sys.argv[1:])
    else:
        # Use default prompt
        prompt = None
    
    # Generate SVG
    result = generator.generate_svg(prompt)
    
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
    else:
        print("\n❌ Generation failed. Check the error messages above.")
        print("💡 Try running: python setup.py --auto")

if __name__ == "__main__":
    main()
