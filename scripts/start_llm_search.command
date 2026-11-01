#!/bin/bash
# Double-click to start (or re-open) your local LLM search.
# LM Studio and Ollama should already be running — they're apps that stay open.
cd "/Users/jackpoormanmini4/gpt-local-gemma" || exit 1

if pgrep -f "photo_index.gradio_app" >/dev/null 2>&1; then
  echo "✅ Search is already running — opening it now."
else
  echo "Starting your local LLM search… (first launch takes ~15–30 seconds)"
  nohup ./start_search.sh > data/gradio.log 2>&1 &
  for i in $(seq 1 25); do
    sleep 3
    if curl -s -o /dev/null --max-time 4 http://127.0.0.1:7860/ ; then
      echo "✅ Up and running."
      break
    fi
    printf "."
  done
fi

open "http://127.0.0.1:7860"
echo ""
echo "Opened http://127.0.0.1:7860 in your browser."
echo "You can close this Terminal window — the search keeps running."
