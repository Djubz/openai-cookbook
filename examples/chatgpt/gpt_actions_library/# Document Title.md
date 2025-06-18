# Document Title
```markdown
{
  "configurations": [
    {
      "type": "debugpy",
      "request": "launch",
      "name": "Launch Python File",
      "program": "${workspaceFolder}/${input:pythonFileToDebug}"
    }
  ],
  "inputs": [
    {
      "type": "promptString",
      "id": "pythonFileToDebug",
      "description": "Enter the relative path to the Python file you want to debug (e.g., examples/api_request_parallel_processor.py)"
    }
  ]
}
```
