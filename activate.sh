#!/bin/bash
# Unified Python Virtual Environment Activation Script
# Handles both direct activation and provides instructions

# Colors for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() { echo -e "${BLUE}ℹ️  $1${NC}"; }
print_success() { echo -e "${GREEN}✅ $1${NC}"; }
print_warning() { echo -e "${YELLOW}⚠️  $1${NC}"; }
print_error() { echo -e "${RED}❌ $1${NC}"; }

# Function to show help
show_help() {
    echo "🐍 Python Virtual Environment Helper"
    echo "====================================="
    echo ""
    echo "Usage:"
    echo "  ./activate.sh              # Show instructions and status"
    echo "  ./activate.sh --activate   # Try to activate directly"
    echo "  ./activate.sh --help       # Show this help"
    echo ""
    echo "📋 Manual activation commands:"
    echo "  source .venv/bin/activate  # Recommended method"
    echo "  . .venv/bin/activate       # Alternative method"
    echo ""
    echo "📋 Verification commands:"
    echo "  which python               # Check Python path"
    echo "  python --version           # Check Python version"
    echo ""
    echo "💡 To deactivate later:"
    echo "  deactivate"
    echo ""
}

# Function to check environment status
check_environment() {
    print_info "Checking virtual environment..."
    
    # Check if .venv exists
    if [ -d ".venv" ]; then
        print_success "Virtual environment found at: $(pwd)/.venv"
        
        # Check Python executable
        if [ -f ".venv/bin/python" ]; then
            python_path=$(ls -la .venv/bin/python* | head -1 | awk '{print $NF}')
            print_success "Python executable: $python_path"
            
            # Get Python version
            if [ -x ".venv/bin/python" ]; then
                python_version=$(.venv/bin/python --version 2>&1)
                print_success "Python version: $python_version"
            fi
        else
            print_warning "Python executable not found in .venv/bin/"
        fi
        
        # Check if environment is already active
        if [ -n "$VIRTUAL_ENV" ]; then
            print_success "Environment is already active: $VIRTUAL_ENV"
        else
            print_info "Environment is not currently active"
        fi
        
        return 0
    else
        print_error "Virtual environment not found!"
        echo ""
        print_info "Please run the installation steps first:"
        echo "  1. python3.12 -m venv .venv"
        echo "  2. source .venv/bin/activate"
        echo "  3. pip install torch transformers safetensors peft 'numpy<2'"
        echo "  4. python setup.py --auto"
        return 1
    fi
}

# Function to attempt activation
try_activate() {
    print_info "Attempting to activate virtual environment..."
    
    # Check if .venv directory exists
    if [ ! -d ".venv" ]; then
        print_error ".venv directory not found!"
        print_info "Please run the installation steps first."
        return 1
    fi
    
    # Try to source the activation script
    if [ -f ".venv/bin/activate" ]; then
        source .venv/bin/activate
        
        # Verify activation worked
        if [ -n "$VIRTUAL_ENV" ]; then
            print_success "Environment activated successfully!"
            print_success "Virtual environment: $VIRTUAL_ENV"
            
            if command -v python >/dev/null 2>&1; then
                print_success "Python path: $(which python)"
            fi
            
            echo ""
            print_info "You can now run your Python scripts:"
            echo "  python text-to-svg.py"
            echo "  python setup.py --auto"
            echo ""
            print_info "To deactivate later, run: deactivate"
            
            return 0
        else
            print_error "Virtual environment activation failed!"
            print_warning "Try running manually: source .venv/bin/activate"
            return 1
        fi
    else
        print_error "Activation script not found in .venv/bin/"
        return 1
    fi
}

# Main script logic
main() {
    # Change to script directory
    cd "$(dirname "$0")"
    
    # Handle command line arguments
    case "${1:-}" in
        "--activate"|"-a")
            try_activate
            ;;
        "--help"|"-h"|"help")
            show_help
            ;;
        "")
            # Default behavior: show status and instructions
            echo "🐍 Python Virtual Environment Helper"
            echo "====================================="
            echo ""
            
            if check_environment; then
                echo ""
                print_info "To activate the environment:"
                echo "  source .venv/bin/activate"
                echo ""
                print_info "Or try automatic activation:"
                echo "  ./activate.sh --activate"
            fi
            ;;
        *)
            print_error "Unknown option: $1"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

# Run main function
main "$@"