from typing import Optional

from fastapi import FastAPI, Depends, HTTPException, Query, Response, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

app = FastAPI()


load_dotenv()

# Database connection
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise ValueError("DATABASE_URL environment variable not set")

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

app = FastAPI(title="Correlation Matrix API")

# Add CORS middleware to allow cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Set up templates directory
BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

# Mount static files directory
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


# Add an endpoint to serve the HTML page
@app.get("/visualize/", response_class=HTMLResponse)
async def visualize_page(request: Request):
    return templates.TemplateResponse("visualize.html", {"request": request})


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Endpoint to get available experiment IDs
@app.get("/subjects/")
def get_experiments(db: Session = Depends(get_db)):
    try:
        query = text(
            """
            SELECT DISTINCT subjectid 
            FROM stage2_run
            ORDER BY subjectid
        """
        )

        result = db.execute(query).fetchall()
        subjectids = [row[0] for row in result]

        return {"subjectids": subjectids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Health check endpoint
@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Correlation Matrix API",
        "endpoints": [
            "/correlation-matrix/?subjectid=<id>",
            "/runs/",
            "/health",
            "/docs",  # FastAPI's automatic documentation
        ],
    }


# Add this new endpoint
@app.get("/correlation-heatmap/")
def get_correlation_heatmap(
    subjectid: int = Query(..., description="Subject ID"),
    db: Session = Depends(get_db),
):
    try:
        # Query to fetch correlations for a specific subjectid
        query = text(
            """
            WITH latest_records AS (
                SELECT 
                    reg1id, 
                    reg2id, 
                    rho,
                    ROW_NUMBER() OVER (PARTITION BY reg1id, reg2id ORDER BY "end" DESC) as rn
                FROM stage2_run
                WHERE subjectid = :subjectid
            )
            SELECT reg1id, reg2id, rho
            FROM latest_records
            WHERE rn = 1
            """
        )
        result = db.execute(query, {"subjectid": subjectid}).fetchall()
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"No results found for subjectid: {subjectid}",
            )

        # Convert to pandas DataFrame
        df = pd.DataFrame(result, columns=["reg1id", "reg2id", "rho"])

        # Get all unique region IDs
        all_regions = sorted(list(set(df["reg1id"].tolist() + df["reg2id"].tolist())))

        # Create a pivot table to form the correlation matrix
        corr_matrix = df.pivot(index="reg1id", columns="reg2id", values="rho")

        # Ensure the matrix has all regions as both rows and columns
        for reg in all_regions:
            if reg not in corr_matrix.index:
                corr_matrix.loc[reg] = np.nan
            if reg not in corr_matrix.columns:
                corr_matrix[reg] = np.nan

        # Sort indices to make sure they're in the same order
        corr_matrix = corr_matrix.reindex(index=all_regions, columns=all_regions)

        # Fill diagonal with 1.0 (correlation of a region with itself is 1)
        np.fill_diagonal(corr_matrix.values, 1.0)

        # Make the matrix symmetric
        for i in range(len(all_regions)):
            for j in range(i + 1, len(all_regions)):
                if pd.isna(corr_matrix.iloc[i, j]) and not pd.isna(
                    corr_matrix.iloc[j, i]
                ):
                    corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i]
                elif pd.isna(corr_matrix.iloc[j, i]) and not pd.isna(
                    corr_matrix.iloc[i, j]
                ):
                    corr_matrix.iloc[j, i] = corr_matrix.iloc[i, j]

        # Create a figure and axis
        plt.figure(figsize=(10, 8))

        sns.heatmap(
            corr_matrix,
            mask=np.triu(np.ones_like(corr_matrix)),
            annot=False,
            cmap="RdBu",
            vmin=-1,  # Minimum value for color mapping
            vmax=1,  # Maximum value for color mapping
            square=True,  # Make cells square
            linewidths=0.5,  # Width of lines between cells
            cbar=True,
            cbar_kws={"shrink": 0.5},
        )

        # Set title and labels
        plt.title(f"Region Correlation Matrix - Subject {subjectid}")
        plt.tight_layout()

        # Save the figure to a bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=100)
        buf.seek(0)
        plt.close()

        # Return the image as a streaming response
        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/completion-status/")
def get_completion_status(
    subjectid: int = Query(..., description="Subject ID"),
    db: Session = Depends(get_db),
):
    try:
        # Query to count unique region pairs and get the latest timestamp
        query = text(
            """
            WITH latest_records AS (
                SELECT 
                    reg1id, 
                    reg2id,
                    ROW_NUMBER() OVER (PARTITION BY reg1id, reg2id ORDER BY "end" DESC) as rn
                FROM stage2_run
                WHERE subjectid = :subjectid
            )
            SELECT COUNT(*) as pair_count
            FROM latest_records
            WHERE rn = 1
            """
        )

        # Query to get the most recent timestamp for this subject
        timestamp_query = text(
            """
            SELECT "end"
            FROM stage2_run
            WHERE subjectid = :subjectid
            ORDER BY "end" DESC
            LIMIT 1
            """
        )

        pair_count_result = db.execute(query, {"subjectid": subjectid}).fetchone()
        timestamp_result = db.execute(
            timestamp_query, {"subjectid": subjectid}
        ).fetchone()

        if not pair_count_result:
            return {
                "pair_count": 0,
                "total_pairs": 4186,
                "completion_percentage": 0,
                "last_updated": None,
            }

        pair_count = pair_count_result[0]
        total_pairs = 4186  # 92 choose 2
        completion_percentage = (pair_count / total_pairs) * 100
        last_updated = timestamp_result[0] if timestamp_result else None

        return {
            "pair_count": pair_count,
            "total_pairs": total_pairs,
            "completion_percentage": round(completion_percentage, 2),
            "last_updated": last_updated,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
