from fastapi import FastAPI, Depends, HTTPException, Query, Request
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
    return templates.TemplateResponse("plot_interactive.html", {"request": request})


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


@app.get("/correlation-heatmap-data/")
def get_correlation_heatmap_data(
    subjectid: int = Query(..., description="Subject ID"),
    examid: int = Query(..., description="Exam ID"),
    db: Session = Depends(get_db),
):
    try:
        # Query to fetch correlations for a specific subjectid and examid
        correlation_query = text(
            """
            WITH latest_records AS (
                SELECT 
                    reg1id, 
                    reg2id, 
                    rho,
                    rho_ca,
                    ROW_NUMBER() OVER (PARTITION BY reg1id, reg2id ORDER BY "end" DESC) as rn
                FROM stage2_run
                WHERE subjectid = :subjectid AND examid = :examid
            )
            SELECT reg1id, reg2id, rho, rho_ca
            FROM latest_records
            WHERE rn = 1
            """
        )
        correlation_result = db.execute(
            correlation_query, {"subjectid": subjectid, "examid": examid}
        ).fetchall()
        if not correlation_result:
            raise HTTPException(
                status_code=404,
                detail=f"No results found for subjectid: {subjectid}, examid: {examid}",
            )

        # Convert to pandas DataFrame
        df = pd.DataFrame(
            correlation_result, columns=["reg1id", "reg2id", "rho", "rho_ca"]
        )

        # Get all unique region IDs from the correlation data
        all_regions = sorted(list(set(df["reg1id"].tolist() + df["reg2id"].tolist())))

        # Fetch region names for all regions in a single query
        region_query = text(
            """
            SELECT regid, name
            FROM region
            WHERE regid IN :region_ids
            """
        )
        region_result = db.execute(
            region_query, {"region_ids": tuple(all_regions)}
        ).fetchall()

        # Create a dictionary mapping region IDs to region names
        region_names = {row[0]: row[1] for row in region_result}

        # Add default names for any regions that weren't found in the region table
        for region_id in all_regions:
            if region_id not in region_names:
                region_names[region_id] = f"Region {region_id}"

        corr_matrix = df.pivot(index="reg1id", columns="reg2id", values="rho")
        corr_matrix_ca = df.pivot(index="reg1id", columns="reg2id", values="rho_ca")

        # Ensure the matrix has all regions as both rows and columns
        for reg in all_regions:
            if reg not in corr_matrix.index:
                corr_matrix.loc[reg] = np.nan
            if reg not in corr_matrix.columns:
                corr_matrix[reg] = np.nan

        # Sort indices to make sure they're in the same order
        corr_matrix = corr_matrix.reindex(index=all_regions, columns=all_regions)
        corr_matrix_ca = corr_matrix_ca.reindex(index=all_regions, columns=all_regions)

        # Make the matrix symmetric
        for i in range(len(all_regions)):
            for j in range(i + 1, len(all_regions)):
                # Handle symmetry for corr_matrix
                if pd.isna(corr_matrix.iloc[i, j]) and not pd.isna(
                    corr_matrix.iloc[j, i]
                ):
                    corr_matrix.iloc[i, j] = corr_matrix.iloc[j, i]
                elif pd.isna(corr_matrix.iloc[j, i]) and not pd.isna(
                    corr_matrix.iloc[i, j]
                ):
                    corr_matrix.iloc[j, i] = corr_matrix.iloc[i, j]

                # Handle symmetry for corr_matrix_ca
                if pd.isna(corr_matrix_ca.iloc[i, j]) and not pd.isna(
                    corr_matrix_ca.iloc[j, i]
                ):
                    corr_matrix_ca.iloc[i, j] = corr_matrix_ca.iloc[j, i]
                elif pd.isna(corr_matrix_ca.iloc[j, i]) and not pd.isna(
                    corr_matrix_ca.iloc[i, j]
                ):
                    corr_matrix_ca.iloc[j, i] = corr_matrix_ca.iloc[i, j]

        z_data = []
        z_data_ca = []

        # Create matrices of the correct size (number of actual regions)
        for i in range(len(all_regions)):
            row = []
            row_ca = []
            for j in range(len(all_regions)):
                if i == j:
                    val = 1.0
                    val_ca = 1.0
                else:
                    val = corr_matrix.iloc[i, j]
                    val_ca = corr_matrix_ca.iloc[i, j]
                row.append(None if pd.isna(val) else val)
                row_ca.append(None if pd.isna(val_ca) else val_ca)
            z_data.append(row)
            z_data_ca.append(row_ca)

        # Convert region IDs to strings for the axis labels
        region_labels = ["r" + str(region) for region in all_regions]

        return {
            "z": z_data,
            "z_ca": z_data_ca,
            "x": region_labels,
            "y": region_labels,
            "indices": all_regions,  # The actual region indices
            "region_names": region_names,  # Add region names mapping
            "subjectid": subjectid,
            "examid": examid,
        }

    except Exception as e:
        import traceback

        error_detail = str(e) + "\n" + traceback.format_exc()
        raise HTTPException(status_code=500, detail=error_detail)


@app.get("/completion-status/")
def get_completion_status(
    subjectid: int = Query(..., description="Subject ID"),
    examid: int = Query(..., description="Exam ID"),
    db: Session = Depends(get_db),
):
    try:
        # Update both queries to include examid filter
        query = text(
            """
            WITH latest_records AS (
                SELECT 
                    reg1id, 
                    reg2id,
                    ROW_NUMBER() OVER (PARTITION BY reg1id, reg2id ORDER BY "end" DESC) as rn
                FROM stage2_run
                WHERE subjectid = :subjectid AND examid = :examid
            )
            SELECT COUNT(*) as pair_count
            FROM latest_records
            WHERE rn = 1
            """
        )

        timestamp_query = text(
            """
            SELECT "end"
            FROM stage2_run
            WHERE subjectid = :subjectid AND examid = :examid
            ORDER BY "end" DESC
            LIMIT 1
            """
        )

        pair_count_result = db.execute(
            query, {"subjectid": subjectid, "examid": examid}
        ).fetchone()
        timestamp_result = db.execute(
            timestamp_query, {"subjectid": subjectid, "examid": examid}
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


@app.get("/exams/{subjectid}")
def get_exams(subjectid: int, db: Session = Depends(get_db)):
    try:
        query = text(
            """
            SELECT DISTINCT examid 
            FROM stage2_run
            WHERE subjectid = :subjectid
            ORDER BY examid
        """
        )

        result = db.execute(query, {"subjectid": subjectid}).fetchall()
        examids = [row[0] for row in result]

        return {"examids": examids}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
