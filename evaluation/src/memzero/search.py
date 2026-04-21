import json
import logging
import os
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from dotenv import load_dotenv
from jinja2 import Template
from openai import OpenAI
from prompts import ANSWER_PROMPT, ANSWER_PROMPT_GRAPH
from tqdm import tqdm

from mem0 import MemoryClient
from src.memzero.progress import Mem0BenchmarkProgress

load_dotenv()

logger = logging.getLogger(__name__)


class MemorySearch:
    def __init__(
        self,
        output_path="results.json",
        top_k=10,
        filter_memories=False,
        is_graph=False,
        log_progress=False,
        progress_mode="detailed",
    ):
        self.mem0_client = MemoryClient(
            api_key=os.getenv("MEM0_API_KEY"),
            org_id=os.getenv("MEM0_ORGANIZATION_ID"),
            project_id=os.getenv("MEM0_PROJECT_ID"),
        )
        self.top_k = top_k
        self.openai_client = OpenAI()
        self.results = defaultdict(list)
        self.output_path = output_path
        self.filter_memories = filter_memories
        self.is_graph = is_graph
        benchmark_name = "mem0:search_graph" if is_graph else "mem0:search"
        self.progress = Mem0BenchmarkProgress(benchmark_name, enabled=log_progress, mode=progress_mode)

        if self.is_graph:
            self.ANSWER_PROMPT = ANSWER_PROMPT_GRAPH
        else:
            self.ANSWER_PROMPT = ANSWER_PROMPT

    def search_memory(self, user_id, query, max_retries=3, retry_delay=1):
        start_time = time.time()
        retries = 0
        while retries < max_retries:
            try:
                if self.is_graph:
                    logger.info("Searching with graph")
                    memories = self.mem0_client.search(
                        query,
                        user_id=user_id,
                        top_k=self.top_k,
                        filter_memories=self.filter_memories,
                        enable_graph=True,
                        output_format="v1.1",
                    )
                else:
                    memories = self.mem0_client.search(
                        query, user_id=user_id, top_k=self.top_k, filter_memories=self.filter_memories
                    )
                break
            except Exception as e:
                logger.warning("Retrying search for user_id=%s after error: %s", user_id, e)
                retries += 1
                if retries >= max_retries:
                    raise e
                time.sleep(retry_delay)

        end_time = time.time()
        if not self.is_graph:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories
            ]
            graph_memories = None
        else:
            semantic_memories = [
                {
                    "memory": memory["memory"],
                    "timestamp": memory["metadata"]["timestamp"],
                    "score": round(memory["score"], 2),
                }
                for memory in memories["results"]
            ]
            graph_memories = [
                {"source": relation["source"], "relationship": relation["relationship"], "target": relation["target"]}
                for relation in memories["relations"]
            ]
        return semantic_memories, graph_memories, end_time - start_time

    def answer_question(
        self,
        speaker_1_user_id,
        speaker_2_user_id,
        question,
        answer,
        category,
        *,
        question_index=None,
    ):
        self.progress.log_step(
            phase="search",
            step="retrieving_memories",
            current_unit=question_index,
            note="searching memories for both speakers",
        )
        speaker_1_memories, speaker_1_graph_memories, speaker_1_memory_time = self.search_memory(
            speaker_1_user_id, question
        )
        speaker_2_memories, speaker_2_graph_memories, speaker_2_memory_time = self.search_memory(
            speaker_2_user_id, question
        )

        search_1_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_1_memories]
        search_2_memory = [f"{item['timestamp']}: {item['memory']}" for item in speaker_2_memories]

        template = Template(self.ANSWER_PROMPT)
        answer_prompt = template.render(
            speaker_1_user_id=speaker_1_user_id.split("_")[0],
            speaker_2_user_id=speaker_2_user_id.split("_")[0],
            speaker_1_memories=json.dumps(search_1_memory, indent=4),
            speaker_2_memories=json.dumps(search_2_memory, indent=4),
            speaker_1_graph_memories=json.dumps(speaker_1_graph_memories, indent=4),
            speaker_2_graph_memories=json.dumps(speaker_2_graph_memories, indent=4),
            question=question,
        )

        self.progress.log_step(
            phase="answer",
            step="generating_answer",
            current_unit=question_index,
            note="calling answer model",
        )
        t1 = time.time()
        response = self.openai_client.chat.completions.create(
            model=os.getenv("MODEL"), messages=[{"role": "system", "content": answer_prompt}], temperature=0.0
        )
        t2 = time.time()
        response_time = t2 - t1
        return (
            response.choices[0].message.content,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        )

    def process_question(self, val, speaker_a_user_id, speaker_b_user_id, *, question_index=None):
        question = val.get("question", "")
        answer = val.get("answer", "")
        category = val.get("category", -1)
        evidence = val.get("evidence", [])
        adversarial_answer = val.get("adversarial_answer", "")

        (
            response,
            speaker_1_memories,
            speaker_2_memories,
            speaker_1_memory_time,
            speaker_2_memory_time,
            speaker_1_graph_memories,
            speaker_2_graph_memories,
            response_time,
        ) = self.answer_question(
            speaker_a_user_id,
            speaker_b_user_id,
            question,
            answer,
            category,
            question_index=question_index,
        )

        result = {
            "question": question,
            "answer": answer,
            "category": category,
            "evidence": evidence,
            "response": response,
            "adversarial_answer": adversarial_answer,
            "speaker_1_memories": speaker_1_memories,
            "speaker_2_memories": speaker_2_memories,
            "num_speaker_1_memories": len(speaker_1_memories),
            "num_speaker_2_memories": len(speaker_2_memories),
            "speaker_1_memory_time": speaker_1_memory_time,
            "speaker_2_memory_time": speaker_2_memory_time,
            "speaker_1_graph_memories": speaker_1_graph_memories,
            "speaker_2_graph_memories": speaker_2_graph_memories,
            "response_time": response_time,
        }

        # Save results after each question is processed
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return result

    def process_data_file(self, file_path):
        with open(file_path, "r") as f:
            data = json.load(f)

        total_questions = sum(len(item["qa"]) for item in data)
        self.progress.start_run(
            total_conversations=len(data),
            total_units=total_questions,
            unit_label="question",
            note="starting mem0 hosted search benchmark",
        )

        for idx, item in tqdm(
            enumerate(data),
            total=len(data),
            desc="Processing conversations",
            disable=self.progress.tqdm_disabled,
        ):
            qa = item["qa"]
            conversation = item["conversation"]
            speaker_a = conversation["speaker_a"]
            speaker_b = conversation["speaker_b"]

            self.progress.start_conversation(
                idx,
                conversation_id=str(idx),
                total_units=len(qa),
                unit_label="question",
                note="starting question answering for conversation",
            )

            speaker_a_user_id = f"{speaker_a}_{idx}"
            speaker_b_user_id = f"{speaker_b}_{idx}"

            for question_index, question_item in enumerate(
                tqdm(
                    qa,
                    total=len(qa),
                    desc=f"Processing questions for conversation {idx}",
                    leave=False,
                    disable=self.progress.tqdm_disabled,
                ),
                start=1,
            ):
                result = self.process_question(
                    question_item,
                    speaker_a_user_id,
                    speaker_b_user_id,
                    question_index=question_index,
                )
                self.results[idx].append(result)

                # Save results after each question is processed
                self.progress.log_step(
                    phase="persist",
                    step="writing_partial_results",
                    current_unit=question_index,
                    note="writing incremental search results",
                )
                with open(self.output_path, "w") as f:
                    json.dump(self.results, f, indent=4)
                self.progress.mark_unit_complete(
                    phase="persist",
                    step="question_complete",
                    current_unit=question_index,
                    note="stored question result",
                )

            self.progress.finish_conversation(note="conversation search completed")

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)
        self.progress.finish_run(note="mem0 hosted search benchmark completed")

    def process_questions_parallel(self, qa_list, speaker_a_user_id, speaker_b_user_id, max_workers=1):
        def process_single_question(val):
            result = self.process_question(val, speaker_a_user_id, speaker_b_user_id)
            # Save results after each question is processed
            with open(self.output_path, "w") as f:
                json.dump(self.results, f, indent=4)
            return result

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(
                tqdm(executor.map(process_single_question, qa_list), total=len(qa_list), desc="Answering Questions")
            )

        # Final save at the end
        with open(self.output_path, "w") as f:
            json.dump(self.results, f, indent=4)

        return results
