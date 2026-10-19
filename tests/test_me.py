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


def test_persona_content_hash_depends_only_on_profile_text(temp_persona_file):
    first = Persona(name="Test User", persona_yaml_file=temp_persona_file)
    second = Persona(name="Someone Else", persona_yaml_file=temp_persona_file)

    assert first.content_hash() == second.content_hash()
    assert len(first.content_hash()) == 16
    assert set(first.content_hash()) <= set("0123456789abcdef")
