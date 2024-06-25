import asyncio
import logging
import os
import subprocess
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List

import docker
import openai
from celery import Celery
from celery.result import AsyncResult
from flask import Flask, jsonify, redirect, render_template, request, url_for

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure OpenAI API
openai.api_key = 'sk-proj-OmLte5EKZi6oncGCdwbET3BlbkFJL953HWsgMCmptp7k3jvJ'

# Configure Celery and Flask
template_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), 'templates'))
static_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'static'))
app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

# Use environment variables for Redis configuration
redis_host = os.environ.get('REDIS_HOST', 'localhost')
redis_port = os.environ.get('REDIS_PORT', '6379')
redis_url = f'redis://{redis_host}:{redis_port}/0'

app.config['CELERY_BROKER_URL'] = redis_url
app.config['CELERY_RESULT_BACKEND'] = redis_url

celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'])
celery.conf.update(app.config)


class Agent(ABC):

    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{self.name}")

    @abstractmethod
    async def process(self, message: Dict) -> Dict:
        pass

    async def _get_gpt4_response(self, prompt: str) -> str:
        try:
            response = await openai.ChatCompletion.acreate(
                model="gpt-4o",
                messages=[{
                    "role":
                    "system",
                    "content":
                    "You are an AI assistant helping with web development."
                }, {
                    "role": "user",
                    "content": prompt
                }])
            return response.choices[0].message.content
        except Exception as e:
            self.logger.error(f"Error in GPT-4 API call: {str(e)}")
            raise


class ProjectManagerAgent(Agent):

    def __init__(self, name: str, agents: List[Agent]):
        super().__init__(name)
        self.agents = agents

    async def process(self, message: Dict) -> Dict:
        try:
            if message["type"] == "new_project":
                self.logger.info(
                    f"Starting new project: {message['app_name']}")
                requirements = await self.agents[0].process({
                    "type":
                    "gather_requirements",
                    "description":
                    message["description"]
                })
                architecture = await self.agents[1].process({
                    "type":
                    "design_architecture",
                    "requirements":
                    requirements["requirements"]
                })
                code = await self.agents[2].process({
                    "type":
                    "write_code",
                    "architecture":
                    architecture["architecture"]
                })
                test_results = await self.agents[3].process({
                    "type":
                    "run_tests",
                    "code":
                    code["code"]
                })
                if test_results["status"] == "failed":
                    debug_results = await self.agents[4].process({
                        "type":
                        "debug",
                        "code":
                        code["code"],
                        "errors":
                        test_results["errors"]
                    })
                    return {
                        "status": "completed",
                        "result": debug_results["fixed_code"]
                    }
                return {"status": "completed", "result": code["code"]}
            else:
                raise ValueError(f"Unknown message type: {message['type']}")
        except Exception as e:
            self.logger.error(f"Error in ProjectManagerAgent: {str(e)}")
            return {"status": "error", "message": str(e)}


class RequirementsAnalystAgent(Agent):

    async def process(self, message: Dict) -> Dict:
        try:
            self.logger.info("Analyzing requirements")
            prompt = f"Analyze the following app description and provide a structured list of requirements:\n\n{message['description']}"
            requirements = await self._get_gpt4_response(prompt)
            return {"requirements": requirements}
        except Exception as e:
            self.logger.error(f"Error in RequirementsAnalystAgent: {str(e)}")
            raise


class ArchitectAgent(Agent):

    async def process(self, message: Dict) -> Dict:
        try:
            self.logger.info("Designing architecture")
            prompt = f"Design a high-level architecture for a web application with these requirements:\n\n{message['requirements']}"
            architecture = await self._get_gpt4_response(prompt)
            return {"architecture": architecture}
        except Exception as e:
            self.logger.error(f"Error in ArchitectAgent: {str(e)}")
            raise


class DeveloperAgent(Agent):

    async def process(self, message: Dict) -> Dict:
        try:
            self.logger.info("Writing code")
            prompt = f"Write Python Flask code for a web application with the following architecture:\n\n{message['architecture']}"
            code = await self._get_gpt4_response(prompt)
            return {"code": code}
        except Exception as e:
            self.logger.error(f"Error in DeveloperAgent: {str(e)}")
            raise


class TestingAgent(Agent):

    async def process(self, message: Dict) -> Dict:
        try:
            self.logger.info("Running tests")
            test_results = await self._run_tests(message['code'])
            return {
                "status":
                "passed" if test_results["errors"] == [] else "failed",
                "errors": test_results["errors"]
            }
        except Exception as e:
            self.logger.error(f"Error in TestingAgent: {str(e)}")
            raise

    async def _run_tests(self, code: str) -> Dict:
        try:
            # Generate test cases using GPT-4
            prompt = f"Generate pytest test cases for the following Flask application:\n\n{code}"
            test_code = await self._get_gpt4_response(prompt)

            # Create a temporary directory for the test
            test_dir = f"/tmp/test_{uuid.uuid4()}"
            os.makedirs(test_dir)

            # Save the application code and test code
            with open(f"{test_dir}/app.py", 'w') as f:
                f.write(code)
            with open(f"{test_dir}/test_app.py", 'w') as f:
                f.write(test_code)

            # Run tests in a Docker container
            client = docker.from_env()
            container = client.containers.run(
                "python:3.9",
                command=f"pip install pytest flask && pytest /app/test_app.py",
                volumes={test_dir: {
                    'bind': '/app',
                    'mode': 'rw'
                }},
                detach=True)
            result = container.wait()
            logs = container.logs().decode('utf-8')

            # Clean up
            container.remove()
            os.system(f"rm -rf {test_dir}")

            if result['StatusCode'] == 0:
                return {"errors": []}
            else:
                return {"errors": [logs]}
        except Exception as e:
            self.logger.error(f"Error in running tests: {str(e)}")
            return {"errors": [str(e)]}


class DebuggingAgent(Agent):

    async def process(self, message: Dict) -> Dict:
        try:
            self.logger.info("Debugging code")
            prompt = f"Debug the following code and fix these errors:\n\nCode:\n{message['code']}\n\nErrors:\n{message['errors']}"
            fixed_code = await self._get_gpt4_response(prompt)
            return {"fixed_code": fixed_code}
        except Exception as e:
            self.logger.error(f"Error in DebuggingAgent: {str(e)}")
            raise


class VirtualEnvironment:

    def __init__(self):
        self.file_system = VirtualFileSystem()
        self.version_control = VersionControl()
        self.web_server = WebServer()


class VirtualFileSystem:

    def __init__(self):
        self.root = "/virtual_root"
        os.makedirs(self.root, exist_ok=True)

    def create_file(self, path: str, content: str):
        try:
            full_path = os.path.join(self.root, path)
            os.makedirs(os.path.dirname(full_path), exist_ok=True)
            with open(full_path, 'w') as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Error creating file: {str(e)}")
            raise

    def read_file(self, path: str) -> str:
        try:
            full_path = os.path.join(self.root, path)
            with open(full_path, 'r') as f:
                return f.read()
        except Exception as e:
            logging.error(f"Error reading file: {str(e)}")
            raise

    def write_file(self, path: str, content: str):
        try:
            full_path = os.path.join(self.root, path)
            with open(full_path, 'w') as f:
                f.write(content)
        except Exception as e:
            logging.error(f"Error writing file: {str(e)}")
            raise


class VersionControl:

    def __init__(self):
        self.repo_path = "/virtual_root"
        try:
            subprocess.run(["git", "init", self.repo_path], check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error initializing git repository: {str(e)}")
            raise

    def commit(self, message: str):
        try:
            subprocess.run(["git", "-C", self.repo_path, "add", "."],
                           check=True)
            subprocess.run(
                ["git", "-C", self.repo_path, "commit", "-m", message],
                check=True)
        except subprocess.CalledProcessError as e:
            logging.error(f"Error committing changes: {str(e)}")
            raise


class WebServer:

    def __init__(self):
        self.client = docker.from_env()

    def start(self, app_code: str):
        try:
            # Create a temporary directory for the app
            app_dir = f"/tmp/app_{uuid.uuid4()}"
            os.makedirs(app_dir)

            # Save the app code
            with open(f"{app_dir}/app.py", 'w') as f:
                f.write(app_code)

            # Create a Dockerfile
            with open(f"{app_dir}/Dockerfile", 'w') as f:
                f.write('''
                FROM python:3.9
                WORKDIR /app
                COPY . /app
                RUN pip install flask
                CMD ["python", "app.py"]
                ''')

            # Build and run the Docker container
            image, _ = self.client.images.build(path=app_dir,
                                                tag=f"app_{uuid.uuid4()}")
            container = self.client.containers.run(image.id,
                                                   detach=True,
                                                   ports={'5000/tcp': 5000})

            logging.info(f"Web server started in container {container.id}")
            return container.id
        except Exception as e:
            logging.error(f"Error starting web server: {str(e)}")
            raise

    def stop(self, container_id: str):
        try:
            container = self.client.containers.get(container_id)
            container.stop()
            container.remove()
            logging.info(
                f"Web server stopped and removed container {container_id}")
        except Exception as e:
            logging.error(f"Error stopping web server: {str(e)}")
            raise


class AgentFramework:

    def __init__(self):
        self.environment = VirtualEnvironment()
        self.agents = [
            ProjectManagerAgent("Project Manager", []),
            RequirementsAnalystAgent("Requirements Analyst"),
            ArchitectAgent("Architect"),
            DeveloperAgent("Developer"),
            TestingAgent("Tester"),
            DebuggingAgent("Debugger")
        ]
        self.agents[0].agents = self.agents[1:]

    async def build_app(self, app_name: str, description: str):
        try:
            message = {
                "type": "new_project",
                "app_name": app_name,
                "description": description
            }
            result = await self.agents[0].process(message)
            if result["status"] == "completed":
                self.environment.file_system.write_file(
                    f"{app_name}.py", result["result"])
                self.environment.version_control.commit(
                    f"Initial commit for {app_name}")
                container_id = self.environment.web_server.start(
                    result["result"])
                result["container_id"] = container_id
            return result
        except Exception as e:
            logging.error(f"Error in build_app: {str(e)}")
            return {"status": "error", "message": str(e)}


# Celery task
@celery.task(bind=True)
def build_app_task(self, app_name: str, description: str):
    try:
        framework = AgentFramework()
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            framework.build_app(app_name, description))
        return result
    except Exception as e:
        logger.error(f"Error in build_app_task: {str(e)}")
        return {'status': 'error', 'message': str(e)}


# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/build', methods=['GET', 'POST'])
def build():
    if request.method == 'POST':
        app_name = request.form['app_name']
        description = request.form['description']
        try:
            task = build_app_task.delay(app_name, description)
            return redirect(url_for('task_status', task_id=task.id))
        except Exception as e:
            logger.error(f"Error in build route: {str(e)}")
            return render_template('error.html', error=str(e))
    return render_template('build.html')


@app.route('/status/<task_id>')
def task_status(task_id):
    task = AsyncResult(task_id, app=celery)
    try:
        if task.state == 'PENDING':
            response = {'state': task.state, 'status': 'Pending...'}
        elif task.state != 'FAILURE':
            response = {
                'state': task.state,
                'status': task.info.get('status', '')
            }
            if 'result' in task.info:
                response['result'] = task.info['result']
        else:
            response = {'state': task.state, 'status': str(task.info)}
        return render_template('status.html', response=response)
    except Exception as e:
        logger.error(f"Error in task_status route: {str(e)}")
        return render_template('error.html', error=str(e))


@app.route('/projects')
def projects():
    # Here you would typically fetch projects from a database
    # For now, we'll just return a dummy list
    projects = [{
        'name': 'Project 1',
        'description': 'A sample project'
    }, {
        'name': 'Project 2',
        'description': 'Another sample project'
    }]
    return render_template('projects.html', projects=projects)


if __name__ == "__main__":
    app.run(debug=False)
