# Interactive Loop (Terminal-style chat)

import ollama
import requests

try:
    r = requests.get('http://localhost:11434')
    print('Ollama server is RUNNING !')
except Exception as e:
    print('Ollama server is not running!')

models = ollama.list()


try:
    # New version (0.4+) — attribute access
    names = [m.model for m in models.models]
except AttributeError:
    # old version — dict access
    names = [m['name'] for m in models['models']]

print('Available models:', names)

if any('qwen3:4b' in n for n in names):
    print('qwen3:4b model ready!')
else:
    print(' qwen3:4b model is not find!')
    print(' Fix: Rub in VS code terminal:')
    print(' ollama pull qwen3:4b')

     
history = [{'role': 'system', 'content': 'You are a helpful assistant.'}]

print(' Chat started! Type "quit" to exit.\n')

while True:
    user_input = input('👤 You: ')
    if user_input.lower() in ['quit', 'exit', 'bye']:
        print(' Chat band!')
        break
    
    history.append({'role': 'user', 'content': user_input})
    resp = ollama.chat(model='qwen3:4b', messages=history)
    reply = resp['message']['content']
    history.append({'role': 'assistant', 'content': reply})
    
    print(f' Bot: {reply}\n')