from unittest.mock import Mock

from mem0.utils.llm_usage import LLMUsageCollector


class TestLLMUsageCollector:
    def test_record_response_accumulates_global_usage(self):
        collector = LLMUsageCollector()
        response = Mock()
        response.usage = Mock(prompt_tokens=11, completion_tokens=7, total_tokens=18)

        collector.record_response(response)

        assert collector.snapshot() == {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "calls": 1,
            "scopes": {},
        }

    def test_callback_records_active_scope(self):
        collector = LLMUsageCollector()
        response = Mock()
        response.usage = Mock(prompt_tokens=5, completion_tokens=3, total_tokens=8)

        with collector.scope("add.extraction"):
            collector.callback(None, response, {})

        assert collector.snapshot() == {
            "prompt_tokens": 5,
            "completion_tokens": 3,
            "total_tokens": 8,
            "calls": 1,
            "scopes": {
                "add.extraction": {
                    "prompt_tokens": 5,
                    "completion_tokens": 3,
                    "total_tokens": 8,
                    "calls": 1,
                }
            },
        }

    def test_nested_scope_uses_leaf_scope_name(self):
        collector = LLMUsageCollector()
        response = {"usage": {"prompt_tokens": 9, "completion_tokens": 1, "total_tokens": 10}}

        with collector.scope("add"):
            with collector.scope("add.procedural"):
                collector.record_response(response)

        assert collector.snapshot()["scopes"] == {
            "add.procedural": {
                "prompt_tokens": 9,
                "completion_tokens": 1,
                "total_tokens": 10,
                "calls": 1,
            }
        }

    def test_reset_clears_totals_and_scopes(self):
        collector = LLMUsageCollector()
        collector.record_usage(prompt_tokens=4, completion_tokens=2)

        collector.reset()

        assert collector.snapshot() == {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "calls": 0,
            "scopes": {},
        }
