import pytest
from pydantic import ValidationError
from nilai_common.api_model import ModelMetadata

def test_model_metadata_creation():
    """Test creating a ModelMetadata instance."""
    metadata = ModelMetadata(
        name="Test Model",
        version="1.0",
        description="A test model",
        author="Test Author",
        license="MIT",
        source="https://example.com",
        supported_features=["feature1", "feature2"]
    )

    assert metadata.id is not None
    assert metadata.name == "Test Model"
    assert metadata.version == "1.0"
    assert metadata.description == "A test model"
    assert metadata.author == "Test Author"
    assert metadata.license == "MIT"
    assert metadata.source == "https://example.com"
    assert metadata.supported_features == ["feature1", "feature2"]

def test_model_metadata_default_id():
    """Test that ModelMetadata generates a default UUID for id."""
    metadata = ModelMetadata(
        name="Test Model",
        version="1.0",
        description="A test model",
        author="Test Author",
        license="MIT",
        source="https://example.com",
        supported_features=["feature1", "feature2"]
    )

    assert metadata.id is not None
    assert len(metadata.id) == 36  # UUID length

def test_model_metadata_invalid_data():
    """Test creating ModelMetadata with invalid data."""
    with pytest.raises(ValidationError):
        ModelMetadata(
            name="",
            version="",
            description="",
            author="",
            license="",
            source="",
        ) # type: ignore