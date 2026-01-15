"""Tests for EvaluatorAgent - response quality evaluation."""

import json

import pytest

from core.llm.types import CompletionMessage, CompletionResponse
from Evaluation import Evaluation
from EvaluatorAgent import EvaluatorAgent


class MockPersona:
    """Fake persona for testing."""

    def __init__(self, name: str = "Test User", summary: str = "A test summary"):
        self.name = name
        self.summary = summary


class TestEvaluatorAgentInit:
    """Test EvaluatorAgent initialization."""

    def test_init_sets_attributes(self, mock_llm_provider):
        """
        Verify init stores persona info, llm, and model.

        Why: EvaluatorAgent needs persona context to evaluate if responses
        match the persona's voice/style. If these aren't stored correctly,
        evaluations would be generic instead of persona-specific.
        """
        persona = MockPersona(name="Max", summary="Backend developer")
        evaluator = EvaluatorAgent(persona, mock_llm_provider, "gpt-5")

        assert evaluator.name == "Max"
        assert evaluator.summary == "Backend developer"
        assert evaluator.llm == mock_llm_provider
        assert evaluator.model == "gpt-5"

    def test_init_builds_system_prompt(self, mock_llm_provider):
        """
        Verify system prompt contains persona info.

        Why: The system prompt instructs the LLM how to evaluate responses.
        It must include persona details so the evaluator knows what "good"
        looks like for THIS specific persona, not generic quality.
        """
        persona = MockPersona(name="Max", summary="Backend developer")
        evaluator = EvaluatorAgent(persona, mock_llm_provider, "gpt-5")

        assert "Max" in evaluator.evaluator_system_prompt
        assert "Backend developer" in evaluator.evaluator_system_prompt
        assert "evaluator" in evaluator.evaluator_system_prompt.lower()


class TestEvaluatorUserPrompt:
    """Test user prompt generation."""

    def test_user_prompt_contains_all_parts(self, mock_llm_provider):
        """
        Verify user prompt includes history, message, and reply.

        Why: The evaluator needs full context to judge a response.
        Without history, it can't tell if the reply is consistent.
        Without the message, it can't tell if the reply is relevant.
        """
        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, mock_llm_provider, "gpt-5")

        prompt = evaluator.evaluator_user_prompt(
            reply="I'm a developer",
            message="What do you do?",
            history="User: Hello\nAssistant: Hi there",
        )

        assert "User: Hello" in prompt
        assert "What do you do?" in prompt
        assert "I'm a developer" in prompt


class TestEvaluate:
    """Test the evaluate method - the core functionality."""

    def test_evaluate_with_structured_output(self, mock_llm_provider):
        """
        Test evaluate when provider supports structured output (Path A).

        Why: Modern LLM providers can return structured Pydantic objects
        directly via parse(). This is the preferred path - cleaner, no
        JSON parsing needed. Tests the happy path with mock_llm_provider
        which has structured_output=True.
        """
        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, mock_llm_provider, "gpt-5")

        result = evaluator.evaluate(
            reply="I'm Max, a backend developer", message="Who are you?", history=[]
        )

        assert isinstance(result, Evaluation)
        assert result.is_acceptable is True
        assert result.feedback == "Looks good!"

    def test_evaluate_with_direct_evaluation_return(self):
        """
        Test evaluate when parse() returns Evaluation directly.

        Why: Some providers return the Pydantic model directly from parse(),
        others wrap it in response.choices[0].message.parsed. Line 52-53
        checks for direct Evaluation return first. This tests that branch.
        """

        class DirectReturnLLM:
            @property
            def capabilities(self):
                return {"structured_output": True}

            def parse(self, *, model, messages, response_format):
                return Evaluation(is_acceptable=False, feedback="Needs improvement")

        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, DirectReturnLLM(), "gpt-5")  # type: ignore[arg-type]

        result = evaluator.evaluate("reply", "message", [])

        assert result.is_acceptable is False
        assert result.feedback == "Needs improvement"

    def test_evaluate_fallback_to_json_extraction(self):
        """
        Test fallback when provider doesn't support structured output (Path B).

        Why: Not all providers support parse(). Gemini, local models, etc.
        may only support complete(). The code falls back to asking for JSON
        and manually parsing it. This tests that fallback path.
        """

        class NoStructuredOutputLLM:
            @property
            def capabilities(self):
                return {"structured_output": False}

            def complete(self, *, model, messages, tools=None):
                json_response = json.dumps(
                    {"is_acceptable": True, "feedback": "Response is acceptable"}
                )
                return CompletionResponse(
                    finish_reason="stop",
                    message=CompletionMessage(
                        role="assistant", content=json_response, tool_calls=None
                    ),
                )

        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, NoStructuredOutputLLM(), "gpt-5")  # type: ignore[arg-type]

        result = evaluator.evaluate("reply", "message", [])

        assert isinstance(result, Evaluation)
        assert result.is_acceptable is True
        assert result.feedback == "Response is acceptable"

    def test_evaluate_extracts_json_from_text(self):
        """
        Test JSON extraction when response has extra text around JSON.

        Why: LLMs don't always return clean JSON. They might say
        "Here is my evaluation: {...} Hope that helps!" The code uses
        find("{") and rfind("}") to extract just the JSON part.
        This tests that extraction logic (lines 70-73).
        """

        class TextAroundJsonLLM:
            @property
            def capabilities(self):
                return {"structured_output": False}

            def complete(self, *, model, messages, tools=None):
                response_text = 'Here is my evaluation: {"is_acceptable": false, "feedback": "Too short"} That is all.'
                return CompletionResponse(
                    finish_reason="stop",
                    message=CompletionMessage(
                        role="assistant", content=response_text, tool_calls=None
                    ),
                )

        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, TextAroundJsonLLM(), "gpt-5")  # type: ignore[arg-type]

        result = evaluator.evaluate("reply", "message", [])

        assert result.is_acceptable is False
        assert result.feedback == "Too short"

    def test_evaluate_fallback_when_parse_raises_not_implemented(self):
        """
        Test fallback when parse() raises NotImplementedError.

        Why: A provider might advertise structured_output=True but raise
        NotImplementedError for certain models or response formats.
        The code catches this (line 56) and falls back to JSON extraction.
        """

        class ParseFailsLLM:
            @property
            def capabilities(self):
                return {"structured_output": True}

            def parse(self, *, model, messages, response_format):
                raise NotImplementedError("Parse not supported")

            def complete(self, *, model, messages, tools=None):
                return CompletionResponse(
                    finish_reason="stop",
                    message=CompletionMessage(
                        role="assistant",
                        content='{"is_acceptable": true, "feedback": "Fallback worked"}',
                        tool_calls=None,
                    ),
                )

        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, ParseFailsLLM(), "gpt-5")  # type: ignore[arg-type]

        result = evaluator.evaluate("reply", "message", [])

        assert result.is_acceptable is True
        assert result.feedback == "Fallback worked"

    def test_evaluate_fallback_when_parse_raises_attribute_error(self):
        """
        Test fallback when parse() raises AttributeError.

        Why: The parse response might have unexpected structure - e.g.,
        missing .choices or .message.parsed attribute. Code catches
        AttributeError (line 56) and falls back to JSON extraction.
        """

        class ParseAttributeErrorLLM:
            @property
            def capabilities(self):
                return {"structured_output": True}

            def parse(self, *, model, messages, response_format):
                raise AttributeError("Missing attribute")

            def complete(self, *, model, messages, tools=None):
                return CompletionResponse(
                    finish_reason="stop",
                    message=CompletionMessage(
                        role="assistant",
                        content='{"is_acceptable": false, "feedback": "Attribute error fallback"}',
                        tool_calls=None,
                    ),
                )

        persona = MockPersona()
        evaluator = EvaluatorAgent(persona, ParseAttributeErrorLLM(), "gpt-5")  # type: ignore[arg-type]

        result = evaluator.evaluate("reply", "message", [])

        assert result.is_acceptable is False
        assert result.feedback == "Attribute error fallback"
