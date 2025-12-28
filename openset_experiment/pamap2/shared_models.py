"""
Pydantic models for PAMAP2 openset experiments
"""
from pydantic import BaseModel


class ActivityClassification(BaseModel):
    """Model for activity classification response"""
    class_label: str
    reasoning: str


class KnownUnknownClassification(BaseModel):
    """Model for known/unknown classification response"""
    is_unknown: bool
    reasoning: str


class SemanticMatching(BaseModel):
    """Model for semantic matching response"""
    most_similar: str
    reasoning: str


class StandardActivityClassification(BaseModel):
    """Model for standard activity classification (used in labeling experiments)"""
    label: str
    reasoning: str
