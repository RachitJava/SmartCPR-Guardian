import sys
import re

file_path = '/Users/rachit/.gemini/antigravity/scratch/smartcpr-guardian/README.md'

with open(file_path, 'r') as f:
    content = f.read()

# Fix the specific GitHub rendering issue: No extra newlines after the mermaid tag
# Correct format: 
# ```mermaid
# graph TD
content = re.sub(r'```mermaid\s+', '```mermaid\n', content)

with open(file_path, 'w') as f:
    f.write(content)
