#!/usr/bin/env bash
# 1) Create an EMPTY repo named "rag-chatbot" at https://github.com/new
# 2) Run: bash push_to_github.sh <your-github-username>
set -e
git init
git add .
git commit -m "Initial commit: LLM-powered RAG chatbot"
git branch -M main
git remote add origin "https://github.com/$1/rag-chatbot.git"
git push -u origin main
