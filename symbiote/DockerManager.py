#!/usr/bin/env python3
#
# DockerManager.py

import docker
import pexpect
import io
import shutil
import subprocess
import time
import threading
from queue import Queue, Empty
import tempfile
from pathlib import Path

class DockerManager:
    def __init__(self, image_name="symbiote_playground"):
        self.client = docker.from_env()
        self.image_name = image_name 
        self.python_container = None
        self.output_queue = Queue()
        self.output_thread = None
        self.keep_running = False
        self.docker_path = shutil.which("docker")
        if self.docker_path is None:
            log(f"Docker executable not found.")
            return

    def ensure_image_exists(self, verbose=False):
        """Ensures that the Docker image exists, building it if necessary."""
        try:
            self.client.images.get(self.image_name)
        except docker.errors.ImageNotFound:
            log(f"Image '{self.image_name}' not found. Building the image...")
            dockerfile = (
                "FROM ubuntu:latest\n"
                "ENV DEBIAN_FRONTEND=noninteractive\n"
                "RUN apt-get update && apt-get install -y --no-install-recommends "
                "perl python3 python3-pip curl wget net-tools vim jq git iputils-ping dnsutils && "
                "apt-get clean && rm -rf /var/lib/apt/lists/*\n"
                "CMD [\"/bin/bash\"]\n"
            )
            with tempfile.TemporaryDirectory() as temp_dir:
                dockerfile_path = Path(temp_dir) / "Dockerfile"
                with open(dockerfile_path, "w") as dockerfile_file:
                    dockerfile_file.write(dockerfile)
                self.client.images.build(path=temp_dir, tag=self.image_name, rm=True)

    def run_interactive_python(self, code: str):
        docker_command = f"{self.docker_path} run -it --rm {self.image_name} python3 -i"
        child = pexpect.spawn(docker_command, encoding='utf-8')

        # Wait for the initial prompt
        child.expect('>>> ')

        # Initialize a string to store the interaction log
        interaction_log = ""

        # Split the code into lines
        lines = code.strip().split('\n')

        # Iterate through each line of code and send it to the subprocess
        for i, line in enumerate(lines):
            try:
                # Send the line to the Python interpreter
                child.sendline(line)

                # Log the input line
                interaction_log += f">>> {line}\n"

                # Expect and capture all outputs until the next primary prompt
                while True:
                    index = child.expect([r'>>> ', r'\.\.\. ', pexpect.EOF])
                    interaction_log += child.before.strip() + "\n"
                    if index == 0:  # '>>> ' prompt indicates primary prompt
                        log(child.before.strip())
                        break
                    elif index == 1:  # '... ' prompt indicates continuation
                        log(child.before.strip())
                        # If this is the last line, send an empty line to execute the block
                        if i == len(lines) - 1:
                            child.sendline('')
                    elif index == 2:  # EOF indicates the end of the output
                        log(child.before.strip())
                        return interaction_log.strip()

            except pexpect.EOF:
                log(child.before.strip())
                interaction_log += child.before.strip() + "\n"
                break
            except pexpect.ExceptionPexpect as e:
                log(f"Error executing line '{line}': {e}")
                interaction_log += f"Error executing line '{line}': {e}\n"
                break
       
        # Close the child process
        child.sendline("quit()")
        child.close()

        return interaction_log.strip()

    def execute_python_file(self, code: str):
        exec_file_path = "/tmp/exec_python.py"
        code = "; ".join(line.strip() for line in code_block.splitlines() if line.strip())
        env_vars = {"CODE": code}
        command = f"echo $CODE | python3"
        response = self.run_command(command, env_vars=env_vars)
        return response

    def run_command(self, command: str, timeout: int = 10, env_vars=None):
        """Runs an isolated command in a new Docker container."""
        self.ensure_image_exists()

        try:
            # Start the container without automatic removal
            container = self.client.containers.run(
                image=self.image_name,
                command=f"bash -c \"{command}\"",
                detach=True,
                stdin_open=True,
                tty=True,
                remove=False,            # Do not automatically remove the container
                mem_limit="256m",
                network_mode="none",
                environment=env_vars
            )

            # Wait for the container to finish or timeout
            exit_code = container.wait(timeout=timeout)
            output = container.logs().decode("utf-8")

            if exit_code['StatusCode'] != 0:
                log(f"Command '{command}' failed with status {exit_code['StatusCode']}.")

            return output

        except docker.errors.ContainerError as e:
            log("Error executing command in container:", e)
        except docker.errors.ImageNotFound as e:
            log("Image not found:", e)
        except docker.errors.APIError as e:
            log("Docker API error:", e)
        except docker.errors.Timeout as e:
            log("Execution timed out.")
            container.kill()  # Ensure container is killed if it times out
        finally:
            # Explicitly remove the container after execution
            try:
                container.remove()
            except docker.errors.APIError as e:
                log("Docker API error during container deletion:", e)

# Example usage for testing
if __name__ == "__main__":
    docker_manager = DockerManager()

    # Ensure the image exists with verbose output
    docker_manager.ensure_image_exists(verbose=True)

    # Run a command and get the output
    command = "python3 --version; df -h; uname -a"
    response = docker_manager.run_command(command)
    print(response)

    # Execute Python code interactively
    code_block = """
x = 5
log("x is:", x)
y = x + 10
log("y is:", y)
"""

    docker_manager.run_interactive_python(code_block)

    response = docker_manager.execute_python_file(code_block)
    print(response)

    # Optionally delete the image
    #docker_manager.delete_image()
