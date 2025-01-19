import asyncio
from typing import Annotated
import re
import os
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta
from livekit import agents, rtc, api
from livekit.agents import JobContext, WorkerOptions, cli, tokenize, tts
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.voice_assistant import VoiceAssistant
from livekit.plugins import deepgram, openai, silero

load_dotenv(dotenv_path=".env.local")

class InterviewAssistantFunctions(agents.llm.FunctionContext):

    @agents.llm.ai_callable(
        description="Called to save candidate's email and name in the system."
    )
    async def save_candidate_details(
        self,
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The candidate's email address"
            ),
        ],
        name: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The candidate's full name"
            ),
        ],
    ):
        api_token = os.getenv('API_TOKEN')
        api_url = f"{os.getenv('CRM_CANDIDATE_ENDPOINT')}"

        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        data = {
            'email': email,
            'name': name,
            'tags': ['new_candidate']
        }

        try:
            response = requests.post(api_url, json=data, headers=headers)
            response.raise_for_status()
            candidate_id = response.json()['candidate']['id']
            return f"Candidate details saved with ID: {candidate_id}."
        except requests.RequestException as e:
            print(f"Error saving candidate details: {e}")
            return "Failed to save candidate details. Please try again."

    @agents.llm.ai_callable(
        description="Ask basic HR questions like experience and skills."
    )
    async def ask_hr_questions(
        self,
        experience: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Candidate's years of experience"
            ),
        ],
        skills: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Candidate's key skills"
            ),
        ],
        previous_companies: Annotated[
            str,
            agents.llm.TypeInfo(
                description="Names of previous companies the candidate has worked for"
            ),
        ],
    ):
        print(f"Experience: {experience}, Skills: {skills}, Companies: {previous_companies}")
        return "HR questions answered successfully."

    @agents.llm.ai_callable(
        description="Schedule an interview slot for the candidate."
    )
    async def schedule_interview(
        self,
        email: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The candidate's email address"
            ),
        ],
        slot: Annotated[
            str,
            agents.llm.TypeInfo(
                description="The interview slot in ISO 8601 format (e.g., 2025-01-18T10:30:00+00:00)"
            ),
        ],
    ):
        api_token = os.getenv('API_TOKEN')
        api_url = f"{os.getenv('INTERVIEW_SLOTS_ENDPOINT')}"

        headers = {
            'Authorization': f'Bearer {api_token}',
            'Content-Type': 'application/json'
        }

        data = {
            'email': email,
            'slot': slot
        }

        try:
            response = requests.post(api_url, json=data, headers=headers)
            response.raise_for_status()
            return f"Interview scheduled successfully for {slot}."
        except requests.RequestException as e:
            print(f"Error scheduling interview: {e}")
            return "Failed to schedule the interview. Please try again."

async def entrypoint(ctx: JobContext):
    await ctx.connect()
    print(f"Connected to room: {ctx.room.name}")

    chat_context = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content=(
                    "You are Ivy, an AI-powered interview assistant for TechCorp. Your job is to assist candidates in scheduling interviews,"
                    "answering HR questions, and ensuring candidate information is saved securely."
                    "Start every conversation by asking for the candidate's name and email address to ensure accurate records."
                    "Ask HR-related questions such as experience, skills, and past companies, then proceed to schedule an interview slot."
                    "If any issues arise, ensure the candidate feels supported and offer to reschedule if necessary."
                ),
            )
        ]
    )

    assistant = VoiceAssistant(
        vad=silero.VAD.load(),
        stt=deepgram.STT(),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=deepgram.TTS(),
        fnc_ctx=InterviewAssistantFunctions(),
        chat_ctx=chat_context,
    )

    assistant.start(ctx.room)

    await assistant.say(
        "Hello! I'm Ivy, your interview assistant. Can I start by getting your name and email address?", 
        allow_interruptions=True
    )

    while ctx.room.connection_state == rtc.ConnectionState.CONN_CONNECTED:
        await asyncio.sleep(1)

if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))
