import subprocess
from typing import Tuple, Dict, Any

from ...object_oriented_agents.utils.logger import get_logger
from ...object_oriented_agents.core_classes.tool_interface import ToolInterface

class PythonExecTool(ToolInterface):
    """
    A Tool that executes Python code securely in a container.
    """

    def get_definition(self) -> Dict[str, Any]:
        """
        Return the JSON/dict definition of the tool's function
        in the format expected by the OpenAI function calling API.
        """
        return {
            "function": {
                "name": "execute_python_code",
                "description": "Executes Python code securely in a container. Python version 3.10 is installed in the container. pandas, numpy, matplotlib, seaborn, and scikit-learn are installed in the container.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "python_code": {
                            "type": "string",
                            "description": "The Python code to execute"
                        }
                    },
                    "required": ["python_code"]
                }
            }
        }

    def run(self, arguments: Dict[str, Any]) -> str:
        """
        Execute the Python code in a Docker container and return the output.
        """
        python_code = arguments["python_code"]
        python_code_stripped = self._strip_code_fence(python_code)

        output, errors = self._run_code_in_container(python_code_stripped)
        if errors:
            return f"[Error]\n{errors}"

        return output

    @staticmethod
    def _strip_code_fence(code: str) -> str:
        """Remove optional Markdown code fences and language hints."""
        code = code.strip()
        if code.startswith("```"):
            code = code[3:]
            if code.startswith("python"):
                code = code[len("python"):]
            code = code.lstrip()
        if code.endswith("```"):
            code = code[:-3]
        return code

    @staticmethod
    def _run_code_in_container(code: str, container_name: str = "sandbox") -> Tuple[str, str]:
        """
        Helper function that actually runs Python code inside a Docker container named `sandbox` (by default).
        """
        cmd = [
            "docker", "exec", "-i",
            container_name,
            "python", "-c", "import sys; exec(sys.stdin.read())"
        ]

        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        out, err = process.communicate(code)
        return out, err