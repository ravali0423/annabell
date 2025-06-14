from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from annabell.tools.stable_video_tool import StableVideoDiffusionTool
from typing import List


@CrewBase
class Annabell():
    """Annabell crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    @agent
    def story_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['story_writer'],  # maps to agents.yaml key
            verbose=True
        )
    
    @agent
    def heygen_video_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['heygen_video_agent'],
            tools=[StableVideoDiffusionTool()],
            verbose=True
        )


    @task
    def write_horror_script(self) -> Task:
        return Task(
            config=self.tasks_config['write_horror_script'],  # maps to tasks.yaml key
            output_file='annabell_script.txt'
        )
    
    @task
    def generate_video_task(self) -> Task:
        return Task(
            config=self.tasks_config['generate_video_task'],
            output_file='annabell_video.mp4'
        )


    @crew
    def crew(self) -> Crew:
        """Creates the Annabell crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )
