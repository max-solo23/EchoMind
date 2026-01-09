from Me import Me


def test_me_loads_persona(temp_persona_file):
    """Test that Me loads persona from YAML file."""
    me = Me(name="Test User", persona_yaml_file=temp_persona_file)
    assert "Test User" in me.summary

def test_me_generates_system_prompt(temp_persona_file):
    """Test that Me generates system prompt."""
    me = Me(name="Test User", persona_yaml_file=temp_persona_file)
    assert me.system_prompt
    assert isinstance(me.system_prompt, str)

def test_me_stores_name(temp_persona_file):
    """Test that Me stores the name."""
    me = Me(name="Test User", persona_yaml_file=temp_persona_file)
    assert me.name == "Test User"

def test_me_system_prompt_includes_name(temp_persona_file):
    """Test that prompt includes the persona name."""
    me = Me(name="Test User", persona_yaml_file=temp_persona_file)
    assert "Test User" in me.system_prompt
