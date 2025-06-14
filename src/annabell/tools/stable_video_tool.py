from crewai.tools import BaseTool
from typing import Type
from pydantic import BaseModel, Field
import requests

# I had a different docker project that was running a stable diffusion API
# and I wanted to use it with the crewai framework. This is a tool that
# allows you to do that. It sends a request to the API and returns the
# path to the generated video. The API is expected to be running on
# localhost:7860. The API is expected to return a JSON response with
# a key "video_path" that contains the path to the generated video.
# This is not a complete implementation, but it should give you an idea of
# how I planned implement a tool that interacts with an external API.

class StableVideoInput(BaseModel):
    """Input schema for StableVideoDiffusionTool."""
    prompt: str = Field(..., description="The scene description to generate a video from.")


class StableVideoDiffusionTool(BaseTool):
    name: str = "Stable Video Diffusion Generator"
    description: str = (
        "Generates a short cinematic video from a prompt using a locally running Stable Video Diffusion API."
    )
    args_schema: Type[BaseModel] = StableVideoInput

    def _run(self, prompt: str) -> str:
        try:
            url = "http://localhost:7860/generate"

            print(f"Sending request to Stable Video Diffusion API with prompt: {prompt}")

            payload = {
                "prompt": prompt,
                "num_frames": 24,
                "fps": 12
            }

            headers = {
                "Content-Type": "application/json"
            }

            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()

            result = response.json()
            video_path = result.get("video_path")

            if video_path:
                return f"🎞️ Stable Video Diffusion video created at: {video_path}"
            else:
                return f"⚠️ Video generation succeeded but no video path was returned."

        except requests.RequestException as e:
            return f"❌ Stable Video Diffusion API request failed: {str(e)}"
        except Exception as e:
            return f"❌ Unexpected error: {str(e)}"
