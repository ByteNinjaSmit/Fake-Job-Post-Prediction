from pydantic import BaseModel, Field
from typing import Optional

class JobPostingRequest(BaseModel):
    title: str = Field(..., example="Marketing Intern")
    company_profile: Optional[str] = Field(None, example="We are a cool startup.")
    description: str = Field(..., example="We need a marketing intern.")
    requirements: Optional[str] = Field(None, example="Experience with social media.")
    benefits: Optional[str] = Field(None, example="Free coffee.")
    telecommuting: int = Field(0, example=0)
    has_company_logo: int = Field(0, example=1)
    has_questions: int = Field(0, example=0)
    employment_type: Optional[str] = Field(None, example="Full-time")
    required_experience: Optional[str] = Field(None, example="Entry level")
    required_education: Optional[str] = Field(None, example="Bachelor's Degree")
    industry: Optional[str] = Field(None, example="Marketing")
    function: Optional[str] = Field(None, example="Marketing")

class PredictionResponse(BaseModel):
    prediction: str = Field(..., example="Real")
    probability: float = Field(..., example=0.95)
    fraudulent_score: float = Field(..., example=0.05)
