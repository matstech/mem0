import json
import logging
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from tqdm import tqdm

from mem0 import MemoryClient
from src.memzero.progress import Mem0BenchmarkProgress

load_dotenv()

logger = logging.getLogger(__name__)


# Update custom instructions
custom_instructions = """
Generate personal memories that follow these guidelines:

1. Each memory should be self-contained with complete context, including:
   - The person's name, do not use "user" while creating memories
   - Personal details (career aspirations, hobbies, life circumstances)
   - Emotional states and reactions
   - Ongoing journeys or future plans
   - Specific dates when events occurred

2. Include meaningful personal narratives focusing on:
   - Identity and self-acceptance journeys
   - Family planning and parenting
   - Creative outlets and hobbies
   - Mental health and self-care activities
   - Career aspirations and education goals
   - Important life events and milestones

3. Make each memory rich with specific details rather than general statements
   - Include timeframes (exact dates when possible)
   - Name specific activities (e.g., "charity race for mental health" rather than just "exercise")
   - Include emotional context and personal growth elements

4. Extract memories only from user messages, not incorporating assistant responses

5. Format each memory as a paragraph with a clear narrative structure that captures the person's experience, challenges, and aspirations
"""


class MemoryADD:
    def __init__(self, data_path=None, batch_size=2, is_graph=False, log_progress=False, progress_mode="detailed"):
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            org_id=os.getenv("MEM0_ORGANIZATION_ID"),
            project_id=os.getenv("MEM0_PROJECT_ID"),
        )

        self.mem0_client.update_project(custom_instructions=custom_instructions)
        self.batch_size = batch_size
        self.data_path = data_path
        self.data = None
        self.is_graph = is_graph
        self.progress = Mem0BenchmarkProgress("mem0:add", enabled=log_progress, mode=progress_mode)
        if data_path:
            self.load_data()

    def load_data(self):
        with open(self.data_path, "r") as f:
            self.data = json.load(f)
        return self.data

    def add_memory(self, user_id, message, metadata, retries=3):
        for attempt in range(retries):
            try:
                _ = self.mem0_client.add(
                    message, user_id=user_id, version="v2", metadata=metadata, enable_graph=self.is_graph
                )
                return
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)  # Wait before retrying
                    continue
                else:
                    raise e

    def add_memories_for_speaker(self, speaker, messages, timestamp, desc):
        for i in tqdm(
            range(0, len(messages), self.batch_size),
            desc=desc,
            disable=self.progress.tqdm_disabled,
        ):
            batch_messages = messages[i : i + self.batch_size]
            self.add_memory(speaker, batch_messages, metadata={"timestamp": timestamp})

    def _conversation_sessions(self, conversation):
        session_keys = []
        for key in conversation.keys():
            if key in ["speaker_a", "speaker_b"] or "date" in key or "timestamp" in key:
                continue
            session_keys.append(key)
        return session_keys

    def process_conversation(self, item, idx):
        conversation = item["conversation"]
        speaker_a = conversation["speaker_a"]
        speaker_b = conversation["speaker_b"]
        session_keys = self._conversation_sessions(conversation)

        speaker_a_user_id = f"{speaker_a}_{idx}"
        speaker_b_user_id = f"{speaker_b}_{idx}"

        if self.progress.mode == "detailed":
            self.progress.start_conversation(
                idx,
                conversation_id=str(idx),
                total_units=len(session_keys),
                unit_label="session",
                note="starting conversation ingest",
            )

        # delete all memories for the two users
        if self.progress.mode == "detailed":
            self.progress.log_step(
                phase="setup",
                step="reset_remote_memories",
                current_unit=1 if session_keys else None,
                note="deleting existing remote memories for both speakers",
            )
        self.mem0_client.delete_all(user_id=speaker_a_user_id)
        self.mem0_client.delete_all(user_id=speaker_b_user_id)

        for session_idx, key in enumerate(session_keys, start=1):
            date_time_key = key + "_date_time"
            timestamp = conversation[date_time_key]
            chats = conversation[key]

            if self.progress.mode == "detailed":
                self.progress.log_step(
                    phase="ingest",
                    step="session_start",
                    current_unit=session_idx,
                    note=f"preparing session={key}",
                )

            messages = []
            messages_reverse = []
            for chat in chats:
                if chat["speaker"] == speaker_a:
                    messages.append({"role": "user", "content": f"{speaker_a}: {chat['text']}"})
                    messages_reverse.append({"role": "assistant", "content": f"{speaker_a}: {chat['text']}"})
                elif chat["speaker"] == speaker_b:
                    messages.append({"role": "assistant", "content": f"{speaker_b}: {chat['text']}"})
                    messages_reverse.append({"role": "user", "content": f"{speaker_b}: {chat['text']}"})
                else:
                    raise ValueError(f"Unknown speaker: {chat['speaker']}")

            if self.progress.mode == "detailed":
                self.progress.log_step(
                    phase="ingest",
                    step="adding_speaker_a",
                    current_unit=session_idx,
                    note=f"speaker={speaker_a_user_id}",
                )
                self.add_memories_for_speaker(
                    speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"
                )
                self.progress.log_step(
                    phase="ingest",
                    step="adding_speaker_b",
                    current_unit=session_idx,
                    note=f"speaker={speaker_b_user_id}",
                )
                self.add_memories_for_speaker(
                    speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"
                )
                self.progress.mark_unit_complete(
                    phase="ingest",
                    step="session_complete",
                    current_unit=session_idx,
                    note=f"completed session={key}",
                )
            else:
                thread_a = threading.Thread(
                    target=self.add_memories_for_speaker,
                    args=(speaker_a_user_id, messages, timestamp, "Adding Memories for Speaker A"),
                )
                thread_b = threading.Thread(
                    target=self.add_memories_for_speaker,
                    args=(speaker_b_user_id, messages_reverse, timestamp, "Adding Memories for Speaker B"),
                )

                thread_a.start()
                thread_b.start()
                thread_a.join()
                thread_b.join()

        if self.progress.mode == "detailed":
            self.progress.finish_conversation(note="remote memories added successfully")
        else:
            logger.info("Messages added successfully for conversation %s", idx)

    def process_all_conversations(self, max_workers=10):
        if not self.data:
            raise ValueError("No data loaded. Please set data_path and call load_data() first.")
        total_conversations = len(self.data)
        total_sessions = sum(len(self._conversation_sessions(item["conversation"])) for item in self.data)
        self.progress.start_run(
            total_conversations=total_conversations,
            total_units=total_sessions if self.progress.mode == "detailed" else None,
            unit_label="session",
            note="starting mem0 hosted add benchmark",
        )

        if self.progress.mode == "detailed":
            for idx, item in enumerate(self.data):
                self.process_conversation(item, idx)
        else:
            self.progress.log_step(
                phase="ingest",
                step="submitting_parallel_conversations",
                note=f"max_workers={max_workers}",
            )
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(self.process_conversation, item, idx): idx for idx, item in enumerate(self.data)}
                for future in as_completed(future_map):
                    idx = future_map[future]
                    future.result()
                    self.progress.mark_conversation_complete(
                        conversation_index=idx,
                        conversation_id=str(idx),
                        note="parallel conversation finished",
                    )

        self.progress.finish_run(note="mem0 hosted add benchmark completed")
