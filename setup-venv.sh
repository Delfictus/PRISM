#!/bin/bash

# Quick Virtual Environment Setup
# For any Python project on Ubuntu 24.04+

echo "╔════════════════════════════════════════════════════════════╗"
echo "║  Python Virtual Environment Setup                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

PROJECT_DIR=$(pwd)
echo "Setting up virtual environment in: $PROJECT_DIR"
echo ""

# Check if python3-venv is installed
if ! dpkg -l | grep -q python3-venv; then
    echo "Installing python3-venv..."
    sudo apt update
    sudo apt install -y python3-venv python3-full
    echo ""
fi

# Create virtual environment
if [ -d "venv" ]; then
    echo "⚠️  Virtual environment already exists."
    read -p "Recreate it? (yes/no): " RECREATE
    if [ "$RECREATE" = "yes" ]; then
        rm -rf venv
        echo "Creating fresh virtual environment..."
        python3 -m venv venv
    fi
else
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo ""
echo "Activating virtual environment..."
source venv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo ""
echo "Installing biopython..."
pip install biopython

# Check for requirements.txt
if [ -f "requirements.txt" ]; then
    echo ""
    echo "Found requirements.txt, installing dependencies..."
    pip install -r requirements.txt
fi

echo ""
echo "✅ Setup complete!"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "HOW TO USE"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Activate the virtual environment:"
echo "   source venv/bin/activate"
echo ""
echo "2. Run your Python scripts:"
echo "   python your_script.py"
echo ""
echo "3. Install more packages (if needed):"
echo "   pip install package-name"
echo ""
echo "4. Deactivate when done:"
echo "   deactivate"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Virtual environment is currently active!"
echo "Your prompt shows: (venv)"
echo ""
