from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


class RawLLMRecord(BaseModel):
  """Pydantic schema to validate raw model records from the Hugging Face API."""

  id: str = Field(..., description="Unique repository ID (author/model_name)")
  downloads: int = Field(default=0, ge=0, description="Downloads in last 30 days")
  likes: int = Field(default=0, ge=0, description="Total community likes")
  pipeline_tag: Optional[str] = Field(
      default="Unknown", description="Task type"
  )
  library_name: Optional[str] = Field(
      default="Unknown", description="Framework/library"
  )
  tags: List[str] = Field(
      default_factory=list, description="Metadata string tags"
  )
  private: bool = Field(
      default=False, description="Whether the repo is private"
  )
  createdAt: Optional[datetime] = Field(
      default=None, description="Creation timestamp"
  )

  @field_validator("id")
  @classmethod
  def validate_model_id(cls, value: str) -> str:
    cleaned = value.strip()
    if not cleaned:
      raise ValueError("Model 'id' cannot be empty.")
    return cleaned

  @field_validator("pipeline_tag", "library_name", mode="before")
  @classmethod
  def fill_missing_strings(cls, value: Optional[str]) -> str:
    if value is None or str(value).strip() == "":
      return "Unknown"
    return str(value).strip()


def validate_records(raw_records: list) -> List[ dict ]:
  """Validates a list of raw JSON dicts using Pydantic and returns valid records."""
  valid_data = []
  invalid_count = 0

  for item in raw_records:
    try:
      validated_obj = RawLLMRecord(**item)
      valid_data.append(validated_obj.model_dump())
    except ValidationError:
      invalid_count += 1

  print(
      f"[VALIDATE] Pydantic validated {len(valid_data)} records"
      f" ({invalid_count} invalid records dropped)."
  )
  return valid_data