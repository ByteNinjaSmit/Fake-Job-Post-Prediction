"""
Pydantic schemas for the FastAPI request/response models.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class JobPostingRequest(BaseModel):
    """Input schema for a single job posting prediction."""

    title: str = Field(..., example="Marketing Intern")
    company_profile: Optional[str] = Field(None, example="A leading tech startup.")
    description: str = Field(..., example="Looking for an energetic intern.")
    requirements: Optional[str] = Field(None, example="1+ years experience.")
    benefits: Optional[str] = Field(None, example="Flexible hours.")
    telecommuting: int = Field(0, ge=0, le=1)
    has_company_logo: int = Field(0, ge=0, le=1)
    has_questions: int = Field(0, ge=0, le=1)
    employment_type: Optional[str] = Field(None, example="Full-time")
    required_experience: Optional[str] = Field(None, example="Entry level")
    required_education: Optional[str] = Field(None, example="Bachelor's Degree")
    industry: Optional[str] = Field(None, example="Marketing and Advertising")
    function: Optional[str] = Field(None, example="Marketing")


class PredictionResponse(BaseModel):
    """Output schema for a single prediction."""

    prediction: str = Field(..., example="Real")
    confidence: float = Field(..., ge=0.0, le=1.0, example=0.95)
    fraudulent_score: float = Field(..., ge=0.0, le=1.0, example=0.05)


class BatchRequest(BaseModel):
    """Input schema for batch predictions."""

    posts: List[JobPostingRequest]


class BatchResponse(BaseModel):
    """Output schema for batch predictions."""

    results: List[PredictionResponse]
    total: int


class FeatureWeight(BaseModel):
    """Schema for a single feature's contribution score."""
    word: str
    weight: float


class ExplainResponse(BaseModel):
    """Output schema for explainability endpoints."""

    prediction: str
    confidence: float
    fraudulent_score: float
    top_features: List[FeatureWeight]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    model_loaded: bool
    model_name: str


class DepartmentStat(BaseModel):
    """Stat for a single department."""
    name: str
    count: int
    fraud: int


class DistributionStat(BaseModel):
    """Stat for class distribution."""
    name: str
    value: int
    color: str


class DatasetInfoResponse(BaseModel):
    """Response for dataset information."""
    total_jobs: int
    real_posts: int
    fraudulent_posts: int
    departments: List[DepartmentStat]
    distribution: List[DistributionStat]
    locations: List[DepartmentStat] # Reusing DepartmentStat for name/count/fraud
    vocabulary_size: int
    fields_per_entry: int
    imbalance_ratio: str
