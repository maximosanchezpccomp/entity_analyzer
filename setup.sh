#!/bin/bash

# Make directories
mkdir -p ~/.streamlit/
mkdir -p cache/

# Create Streamlit config
echo "\
[general]
email = \"user@domain.com\"
" > ~/.streamlit/credentials.toml

echo "\
[theme]
primaryColor = \"#1E3A8A\"
backgroundColor = \"#FFFFFF\"
secondaryBackgroundColor = \"#F8F9FA\"
textColor = \"#262730\"
font = \"sans serif\"
[server]
enableCORS = false
enableXsrfProtection = false
" > ~/.streamlit/config.toml

# Install requirements
pip install -r requirements.txt
