from core.persona import Persona


def test_persona_loads_persona(temp_persona_file):
    persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
    assert "Test User" in persona.summary


def test_persona_generates_system_prompt(temp_persona_file):
    persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
    assert persona.system_prompt
    assert isinstance(persona.system_prompt, str)


def test_persona_stores_name(temp_persona_file):
    persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
    assert persona.name == "Test User"


def test_persona_system_prompt_includes_name(temp_persona_file):
    persona = Persona(name="Test User", persona_yaml_file=temp_persona_file)
    assert "Test User" in persona.system_prompt
