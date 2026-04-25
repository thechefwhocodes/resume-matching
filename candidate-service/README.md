# Candidate Service Implementation

This folder is where you should implement your resume matching service.

## Structure

Create your implementation in a subfolder with your name:

```
candidate-service/
├── main.py (or server.ts, etc.)
├── requirements.txt (or package.json, etc.)
├── README.md
```

## Requirements

Your service must:

1. Run on `http://localhost:8000`
2. Implement the `/match` endpoint (see API spec in main README)
3. Accept resume text and return top-k matching jobs
4. **Core output:** Return a ranked list of job titles with match scores for the resume

The API contract in the main README provides a suggested response format, but you have flexibility in your implementation approach.

## Getting Started

1. Create your implementation folder
2. Choose your language/framework (Python, TypeScript, Go, Rust, etc.)
3. Implement the matching service
4. Document your approach in your README.md
5. Write your evaluation and reflection in EVALUATION.md

For the full API specification and requirements, see the main README.md in the project root.
