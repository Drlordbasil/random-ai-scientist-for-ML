import os
import subprocess
import requests
import logging
import ast
from typing import List, Tuple, Dict

# Import necessary libraries for embeddings and language model interactions
from groq import Groq
import chromadb
import ollama

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

client = Groq()
MODEL = 'llama-3.1-70b-versatile'

# Initialize the knowledge base using ChromaDB
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="ml_knowledge")

# Memory for storing past analyses, codes, decisions, and topics
memory = {
    'analyses': [],
    'codes': [],
    'decisions': [],
    'topics': []
}

def create_workspace() -> str:
    """
    Create a workspace directory for the AI scientist if it doesn't exist.

    Returns:
        str: Path to the workspace directory.
    """
    workspace_dir = 'AI_scientist_workspace'
    if not os.path.exists(workspace_dir):
        os.makedirs(workspace_dir)
    return workspace_dir

def get_embedding(text: str) -> List[float]:
    """
    Get the embedding of a given text using the embedding model.

    Args:
        text (str): The text to embed.

    Returns:
        List[float]: The embedding vector.
    """
    try:
        response = ollama.embeddings(model="mxbai-embed-large", prompt=text)
        embedding = response["embedding"]
        return embedding
    except Exception as e:
        logging.error(f"Error getting embedding: {e}")
        return []

def execute_code_file(file_path: str) -> Dict[str, str]:
    """
    Execute a Python code file and capture its output.

    Args:
        file_path (str): Path to the Python file.

    Returns:
        Dict[str, str]: A dictionary containing the output or error.
    """
    try:
        process = subprocess.Popen(['python', file_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        stdout, stderr = process.communicate(timeout=300)
        output = stdout + stderr
        return {"output": output}
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate()
        output = stdout + stderr
        return {"error": "Execution timed out", "output": output}
    except Exception as e:
        return {"error": str(e)}

def web_search(query: str) -> Dict[str, List[Dict[str, str]]]:
    """
    Perform a web search for academic papers related to the query.

    Args:
        query (str): The search query.

    Returns:
        Dict[str, List[Dict[str, str]]]: A dictionary containing search results.
    """
    try:
        response = requests.get(
            'https://api.semanticscholar.org/graph/v1/paper/search',
            params={'query': query, 'limit': 5, 'fields': 'title,abstract'}
        )
        response.raise_for_status()
        data = response.json()
        papers = data.get('data', [])
        results = []
        for paper in papers:
            title = paper.get('title', '')
            abstract = paper.get('abstract', '')
            results.append({'title': title, 'abstract': abstract})
            doc_id = paper.get('paperId', '') or title
            content = title + "\n" + abstract
            embedding = get_embedding(content)
            collection.add(
                ids=[doc_id],
                embeddings=[embedding],
                documents=[content]
            )
        return {"results": results}
    except Exception as e:
        logging.error(f"Web search error: {e}")
        return {"results": []}

def retrieve_relevant_info(query: str) -> List[str]:
    """
    Retrieve relevant information from the knowledge base for a given query.

    Args:
        query (str): The query to search for.

    Returns:
        List[str]: A list of relevant documents.
    """
    embedding = get_embedding(query)
    if not embedding:
        return []
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )
    documents = results.get('documents', [[]])[0]
    return documents

# Define agent classes for different tasks
class ResearcherAgent:
    """
    Agent responsible for generating and selecting research topics.
    """
    def generate_research_topics(self) -> str:
        """
        Generate multiple novel research topics.

        Returns:
            str: A string containing a list of research topics.
        """
        system_prompt = (
            "You are a creative and forward-thinking AI researcher specializing in machine learning. "
            "Your task is to generate innovative and impactful research topics."
        )
        prompt = (
            "Generate 5 novel and interesting research topics in the field of machine learning. "
            "Each topic should be unique and include a brief description."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        topics = response.choices[0].message.content.strip()
        memory['topics'].append(topics)
        return topics

    def select_promising_topic(self, topics: str) -> str:
        """
        Select the most promising research topic from a list.

        Args:
            topics (str): A string containing a list of research topics.

        Returns:
            str: The selected topic with justification.
        """
        system_prompt = (
            "You are an experienced AI researcher with expertise in evaluating research ideas."
        )
        prompt = (
            f"Given the following research topics:\n{topics}\n\n"
            "Evaluate each topic for novelty, feasibility, and potential impact. "
            "Select the most promising topic for further research and provide a detailed justification for your choice."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        selection = response.choices[0].message.content.strip()
        return selection

class CodeDeveloperAgent:
    """
    Agent responsible for developing experiment plans and generating code.
    """
    def generate_experiment_plan(self, topic: str, retrieved_info: str) -> str:
        """
        Generate a detailed experiment plan.

        Args:
            topic (str): The selected research topic.
            retrieved_info (str): Relevant information from the knowledge base.

        Returns:
            str: The experiment plan.
        """
        system_prompt = (
            "You are a methodical AI developer skilled in designing comprehensive experiment plans for machine learning research."
        )
        prompt = (
            f"Based on the selected topic:\n{topic}\n\n"
            f"Using the following retrieved information:\n{retrieved_info}\n\n"
            "Develop a detailed experiment plan. Include objectives, hypotheses, methodologies, datasets, and evaluation metrics."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        plan = response.choices[0].message.content.strip()
        return plan

    def generate_experiment_code(self, plan: str, retrieved_info: str, previous_codes: str) -> str:
        """
        Generate Python code to implement the experiment.

        Args:
            plan (str): The experiment plan.
            retrieved_info (str): Relevant information from the knowledge base.
            previous_codes (str): Previously developed codes.

        Returns:
            str: The generated Python code.
        """
        system_prompt = (
            "You are a proficient AI programmer with expertise in implementing machine learning experiments."
        )
        prompt = (
            f"Based on the following experiment plan:\n{plan}\n\n"
            f"Utilize the following information:\n{retrieved_info}\n\n"
            f"Refer to previous codes if necessary:\n{previous_codes}\n\n"
            "Write efficient and well-documented Python code to implement the experiment. "
            "Ensure code correctness, readability, and adherence to best practices."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        code = response.choices[0].message.content.strip()
        return code

class CodeTesterAgent:
    """
    Agent responsible for testing and fixing the experiment code.
    """
    def __init__(self, code: str, file_name: str, workspace_dir: str):
        self.code = code
        self.file_name = file_name
        self.workspace_dir = workspace_dir
        self.file_path = os.path.join(workspace_dir, file_name)

    def save_code(self):
        """
        Save the code to a file.
        """
        with open(self.file_path, 'w') as f:
            f.write(self.code)

    def validate_python_code(self) -> bool:
        """
        Validate the syntax of the Python code.

        Returns:
            bool: True if code is syntactically correct, False otherwise.
        """
        try:
            ast.parse(self.code)
            return True
        except SyntaxError as e:
            logging.error(f"Syntax error: {e}")
            return False

    def test_code(self) -> Tuple[bool, str]:
        """
        Test the code by executing it.

        Returns:
            Tuple[bool, str]: (True, output) if code runs successfully, (False, error message) otherwise.
        """
        if not self.validate_python_code():
            return False, "Syntax Error in code."
        self.save_code()
        result = execute_code_file(self.file_path)
        output = result.get("output", "")
        if "Traceback" in output:
            return False, output
        return True, output

    def fix_code(self, error_output: str) -> str:
        """
        Fix the code using the error output.

        Args:
            error_output (str): The error message from code execution.

        Returns:
            str: The fixed code.
        """
        system_prompt = (
            "You are a skilled AI programmer adept at debugging and fixing code errors."
        )
        prompt = (
            f"The following Python code has errors:\n\n{self.code}\n\n"
            f"The errors are:\n\n{error_output}\n\n"
            "Please fix the code to eliminate the errors and improve its performance."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        fixed_code = response.choices[0].message.content.strip()
        self.code = fixed_code
        return fixed_code

class AnalystAgent:
    """
    Agent responsible for analyzing experiment results.
    """
    def analyze_results(self, output: str, previous_analyses: str) -> str:
        """
        Analyze the output of the experiment.

        Args:
            output (str): The output from code execution.
            previous_analyses (str): Previous analyses for context.

        Returns:
            str: The analysis of the results.
        """
        system_prompt = (
            "You are an analytical AI scientist specializing in interpreting experimental results."
        )
        prompt = (
            f"Given the following output from an experiment:\n{output}\n\n"
            f"Considering previous analyses:\n{previous_analyses}\n\n"
            "Provide a comprehensive analysis of the results, including insights, potential improvements, and conclusions."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        analysis = response.choices[0].message.content.strip()
        memory['analyses'].append(analysis)
        return analysis

class PeerReviewerAgent:
    """
    Agent responsible for providing peer review feedback.
    """
    def peer_review_analysis(self, analysis: str) -> str:
        """
        Peer review the analysis provided.

        Args:
            analysis (str): The analysis to review.

        Returns:
            str: Peer review feedback.
        """
        system_prompt = (
            "You are an objective and critical AI peer reviewer evaluating scientific analyses."
        )
        prompt = (
            f"Critically evaluate the following analysis:\n{analysis}\n\n"
            "Provide feedback on strengths, weaknesses, and suggest actionable improvements."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        review = response.choices[0].message.content.strip()
        return review

class DecisionMakerAgent:
    """
    Agent responsible for deciding the next steps in the research process.
    """
    def decide_next_steps(self, analysis: str, peer_review: str) -> str:
        """
        Decide on the next steps based on analysis and peer review.

        Args:
            analysis (str): The analysis of the results.
            peer_review (str): The peer review feedback.

        Returns:
            str: The decision on next steps.
        """
        system_prompt = (
            "You are a strategic AI decision-maker overseeing research directions."
        )
        prompt = (
            f"Based on the analysis:\n{analysis}\n\n"
            f"And peer review feedback:\n{peer_review}\n\n"
            "Determine the most appropriate next steps: refine the experiment, explore alternative approaches, or proceed to publish. "
            "Justify your decision with clear reasoning."
        )
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
        )
        decision = response.choices[0].message.content.strip()
        memory['decisions'].append(decision)
        return decision

class KnowledgeBaseAgent:
    """
    Agent responsible for updating the knowledge base.
    """
    def update_knowledge_base(self, content: str, doc_id: str):
        """
        Update the knowledge base with new information.

        Args:
            content (str): The content to store.
            doc_id (str): The document ID.
        """
        embedding = get_embedding(content)
        collection.add(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[content]
        )

class ManagerAgent:
    """
    Agent responsible for managing the overall research process using a Tree of Thought approach.
    """
    def __init__(self):
        self.researcher = ResearcherAgent()
        self.developer = CodeDeveloperAgent()
        self.tester = None  # Will be initialized later
        self.analyst = AnalystAgent()
        self.peer_reviewer = PeerReviewerAgent()
        self.decision_maker = DecisionMakerAgent()
        self.kb_agent = KnowledgeBaseAgent()
        self.workspace_dir = create_workspace()
        self.iteration = 1
        self.max_iterations = 10
        self.previous_codes = ""
        self.thought_tree = []

    def plan_tasks(self):
        """
        Plan tasks using a Tree of Thought approach and assign them to agents.
        """
        logging.info(f"Manager is planning tasks for iteration {self.iteration}")

        # Generate research topics
        topics = self.researcher.generate_research_topics()
        logging.info(f"Generated topics:\n{topics}")
        self.thought_tree.append(('generate_topics', topics))

        # Select promising topic
        selection = self.researcher.select_promising_topic(topics)
        logging.info(f"Selected topic:\n{selection}")
        selected_topic = selection.split('\n')[0].strip()
        self.thought_tree.append(('select_topic', selected_topic))

        # Perform web search and retrieve info
        web_search(selected_topic)
        retrieved_info_list = retrieve_relevant_info(selected_topic)
        retrieved_info = "\n".join(retrieved_info_list)
        logging.info(f"Retrieved information:\n{retrieved_info}")
        self.thought_tree.append(('retrieve_info', retrieved_info))

        # Generate experiment plan
        plan = self.developer.generate_experiment_plan(selected_topic, retrieved_info)
        logging.info(f"Experiment plan:\n{plan}")
        self.thought_tree.append(('experiment_plan', plan))

        # Generate experiment code
        experiment_code = self.developer.generate_experiment_code(plan, retrieved_info, self.previous_codes)
        logging.info("Generated experiment code.")
        self.thought_tree.append(('experiment_code', experiment_code))

        # Initialize code tester agent
        self.tester = CodeTesterAgent(experiment_code, f"experiment_{self.iteration}.py", self.workspace_dir)

    def execute_tasks(self):
        """
        Execute the planned tasks, testing and refining code as necessary.
        """
        # Code testing and fixing loop
        code_is_valid, test_output = self.tester.test_code()
        while not code_is_valid:
            logging.info("Code has errors, attempting to fix...")
            experiment_code = self.tester.fix_code(test_output)
            self.thought_tree.append(('fix_code', experiment_code))
            code_is_valid, test_output = self.tester.test_code()
        logging.info("Code is valid and executable.")
        self.previous_codes += self.tester.code + "\n"

        # Analyze results
        analysis = self.analyst.analyze_results(test_output, "\n".join(memory['analyses']))
        logging.info(f"Analysis:\n{analysis}")
        self.thought_tree.append(('analysis', analysis))

        # Peer review
        peer_review = self.peer_reviewer.peer_review_analysis(analysis)
        logging.info(f"Peer review feedback:\n{peer_review}")
        self.thought_tree.append(('peer_review', peer_review))

        # Decision making
        decision = self.decision_maker.decide_next_steps(analysis, peer_review)
        logging.info(f"Decision on next steps:\n{decision}")
        self.thought_tree.append(('decision', decision))

        # Update knowledge base
        content = f"Experiment on topic: {self.tester.file_name}\nAnalysis: {analysis}\nPeer Review: {peer_review}\nDecision: {decision}"
        self.kb_agent.update_knowledge_base(content, f"iteration_{self.iteration}")

        # Decide next action
        if "proceed to publish" in decision.lower():
            logging.info("Decided to publish findings. Ending research.")
            return False  # End the loop
        elif "refine the experiment" in decision.lower():
            logging.info("Refining the current experiment.")
            # Adjust the plan and re-execute
            self.developer.generate_experiment_plan(self.tester.code, "\n".join(memory['analyses']))
            return True  # Continue the loop without incrementing iteration
        elif "explore alternative approaches" in decision.lower():
            logging.info("Exploring alternative approaches.")
            self.iteration += 1
            return True  # Proceed to next iteration
        else:
            logging.info("No clear decision, proceeding to next iteration.")
            self.iteration += 1
            return True  # Proceed to next iteration

    def run(self):
        """
        Run the manager agent to coordinate the research process.
        """
        while self.iteration <= self.max_iterations:
            self.plan_tasks()
            continue_research = self.execute_tasks()
            if not continue_research:
                break

def run_ml_scientist():
    """
    Main function to run the AI scientist agents in a collaborative workflow.
    """
    manager = ManagerAgent()
    manager.run()
    logging.info("AI scientist has completed its research cycle.")

if __name__ == "__main__":
    run_ml_scientist()
