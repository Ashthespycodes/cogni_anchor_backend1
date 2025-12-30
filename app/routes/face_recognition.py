"""
Face Recognition API Endpoints
Handles face detection, person enrollment, scanning, and matching
"""

import logging
import os
import uuid
from fastapi import APIRouter, HTTPException, status, UploadFile, File, Form
from typing import List, Optional
import json

from app.models.database_models import (
    PersonCreate,
    PersonUpdate,
    PersonInfo,
    PersonWithEmbedding,
    FaceScanRequest,
    FaceScanResponse,
    PeopleListResponse,
    SuccessResponse
)
from app.services.supabase_client import get_supabase_client
from app.services.face_recognition_service import get_face_recognition_service

logger = logging.getLogger("FaceRecognitionAPI")
router = APIRouter(prefix="/api/v1/face", tags=["Face Recognition"])

# Temporary directory for image processing
TEMP_DIR = "temp/face_images"
os.makedirs(TEMP_DIR, exist_ok=True)

# ===== HELPER FUNCTIONS =====

async def save_uploaded_image(image: UploadFile) -> str:
    """Save uploaded image to temporary location"""
    try:
        # Generate unique filename
        ext = image.filename.split(".")[-1] if "." in image.filename else "jpg"
        filename = f"{uuid.uuid4().hex}.{ext}"
        filepath = os.path.join(TEMP_DIR, filename)

        # Save image
        contents = await image.read()
        with open(filepath, "wb") as f:
            f.write(contents)

        logger.info(f"Saved uploaded image to {filepath}")
        return filepath

    except Exception as e:
        logger.error(f"Error saving uploaded image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save image: {str(e)}"
        )

async def upload_to_supabase_storage(filepath: str, pair_id: str, person_name: str) -> str:
    """Upload image to Supabase Storage"""
    try:
        supabase = get_supabase_client()

        # Generate storage path
        filename = os.path.basename(filepath)
        storage_path = f"{pair_id}/{person_name}_{filename}"

        # Read image file
        with open(filepath, "rb") as f:
            image_data = f.read()

        # Upload to Supabase Storage
        result = supabase.storage.from_("face-images").upload(
            path=storage_path,
            file=image_data,
            file_options={"content-type": "image/jpeg"}
        )

        # Get public URL
        public_url = supabase.storage.from_("face-images").get_public_url(storage_path)

        logger.info(f"Uploaded image to Supabase Storage: {storage_path}")
        return public_url

    except Exception as e:
        logger.error(f"Error uploading to Supabase Storage: {e}")
        # Return local path as fallback
        return filepath

# ===== API ENDPOINTS =====

@router.post("/addPerson", response_model=PersonInfo, status_code=status.HTTP_201_CREATED)
async def add_person(
    pair_id: str = Form(...),
    name: str = Form(...),
    relationship: str = Form(...),
    occupation: str = Form(...),
    age: Optional[int] = Form(None),
    notes: Optional[str] = Form(None),
    image: UploadFile = File(...)
):
    """
    Add a person to face recognition database

    - **pair_id**: Patient-caretaker pair ID
    - **name**: Person's name
    - **relationship**: Relationship to patient (e.g., "Son", "Daughter", "Friend")
    - **occupation**: Person's occupation
    - **age**: Person's age (optional)
    - **notes**: Additional notes (optional)
    - **image**: Face image file
    """
    try:
        logger.info(f"Adding person {name} for pair {pair_id}")

        # Save uploaded image
        temp_filepath = await save_uploaded_image(image)

        # Generate face embedding
        face_service = get_face_recognition_service()
        embedding = face_service.generate_embedding(temp_filepath)

        if not embedding:
            # Clean up temp file
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No face detected in image. Please upload a clear face photo."
            )

        # Upload image to Supabase Storage
        image_url = await upload_to_supabase_storage(temp_filepath, pair_id, name)

        supabase = get_supabase_client()

        # Insert person into database
        person_result = supabase.table("people").insert({
            "pair_id": pair_id,
            "name": name,
            "relationship": relationship,
            "occupation": occupation,
            "age": age,
            "notes": notes,
            "image_url": image_url
        }).execute()

        if not person_result.data or len(person_result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to add person"
            )

        person_id = person_result.data[0]["id"]

        # Insert face embedding
        embedding_result = supabase.table("face_embeddings").insert({
            "person_id": person_id,
            "embedding": embedding  # Supabase will handle the vector
        }).execute()

        if not embedding_result.data:
            logger.warning(f"Failed to save embedding for person {person_id}")

        # Clean up temp file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        logger.info(f"Person {name} added successfully with ID {person_id}")

        return PersonInfo(**person_result.data[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error adding person: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add person: {str(e)}"
        )

@router.get("/getPeople", response_model=PeopleListResponse)
async def get_people(pair_id: str):
    """
    Get all people for a pair

    - **pair_id**: Patient-caretaker pair ID
    """
    try:
        logger.info(f"Fetching people for pair {pair_id}")

        supabase = get_supabase_client()

        # Fetch people from database
        result = supabase.table("people") \
            .select("*") \
            .eq("pair_id", pair_id) \
            .execute()

        people = [PersonInfo(**person) for person in result.data]

        logger.info(f"Found {len(people)} people for pair {pair_id}")

        return PeopleListResponse(
            people=people,
            count=len(people)
        )

    except Exception as e:
        logger.error(f"Error fetching people: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch people: {str(e)}"
        )

@router.post("/scan", response_model=FaceScanResponse)
async def scan_face(scan_request: FaceScanRequest):
    """
    Scan and match a face against database

    - **pair_id**: Patient-caretaker pair ID
    - **embedding**: Face embedding vector (192 dimensions for Facenet512)
    """
    try:
        logger.info(f"Scanning face for pair {scan_request.pair_id}")

        supabase = get_supabase_client()

        # Get all people and their embeddings for this pair
        people_result = supabase.table("people") \
            .select("*") \
            .eq("pair_id", scan_request.pair_id) \
            .execute()

        if not people_result.data:
            logger.info("No people found in database")
            return FaceScanResponse(matched=False)

        # Get embeddings
        database_embeddings = []
        for person in people_result.data:
            person_id = person["id"]

            # Fetch embedding
            embedding_result = supabase.table("face_embeddings") \
                .select("embedding") \
                .eq("person_id", person_id) \
                .execute()

            if embedding_result.data and len(embedding_result.data) > 0:
                embedding = embedding_result.data[0]["embedding"]
                database_embeddings.append((person_id, embedding))

        if not database_embeddings:
            logger.info("No embeddings found in database")
            return FaceScanResponse(matched=False)

        # Find best match
        face_service = get_face_recognition_service()
        match_result = face_service.find_best_match(
            query_embedding=scan_request.embedding,
            database_embeddings=database_embeddings,
            threshold=0.6  # 60% similarity threshold
        )

        if not match_result:
            logger.info("No match found above threshold")
            return FaceScanResponse(matched=False)

        person_id, similarity_score = match_result

        # Get person info
        person = next((p for p in people_result.data if p["id"] == person_id), None)

        if not person:
            logger.error(f"Person {person_id} not found in database")
            return FaceScanResponse(matched=False)

        logger.info(f"Match found: {person['name']} (score: {similarity_score:.4f})")

        return FaceScanResponse(
            matched=True,
            score=similarity_score,
            person=PersonInfo(**person)
        )

    except Exception as e:
        logger.error(f"Error scanning face: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan face: {str(e)}"
        )

@router.post("/scanImage", response_model=FaceScanResponse)
async def scan_face_from_image(
    pair_id: str = Form(...),
    image: UploadFile = File(...)
):
    """
    Scan and match a face from uploaded image

    - **pair_id**: Patient-caretaker pair ID
    - **image**: Face image file
    """
    try:
        logger.info(f"Scanning face from image for pair {pair_id}")

        # Save uploaded image
        temp_filepath = await save_uploaded_image(image)

        # Generate face embedding
        face_service = get_face_recognition_service()
        embedding = face_service.generate_embedding(temp_filepath)

        # Clean up temp file
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

        if not embedding:
            return FaceScanResponse(matched=False)

        # Use scan endpoint logic
        scan_request = FaceScanRequest(pair_id=pair_id, embedding=embedding)
        return await scan_face(scan_request)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error scanning face from image: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to scan face: {str(e)}"
        )

@router.put("/updatePerson", response_model=PersonInfo)
async def update_person(
    person_id: int = Form(...),
    name: Optional[str] = Form(None),
    relationship: Optional[str] = Form(None),
    occupation: Optional[str] = Form(None),
    age: Optional[int] = Form(None),
    notes: Optional[str] = Form(None),
    image: Optional[UploadFile] = File(None)
):
    """
    Update person information

    - **person_id**: Person ID
    - **name**: New name (optional)
    - **relationship**: New relationship (optional)
    - **occupation**: New occupation (optional)
    - **age**: New age (optional)
    - **notes**: New notes (optional)
    - **image**: New face image (optional)
    """
    try:
        logger.info(f"Updating person {person_id}")

        supabase = get_supabase_client()

        # Build update data
        update_data = {}
        if name is not None:
            update_data["name"] = name
        if relationship is not None:
            update_data["relationship"] = relationship
        if occupation is not None:
            update_data["occupation"] = occupation
        if age is not None:
            update_data["age"] = age
        if notes is not None:
            update_data["notes"] = notes

        # Handle image update
        if image:
            # Get person's pair_id
            person_result = supabase.table("people").select("pair_id, name").eq("id", person_id).execute()

            if not person_result.data:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Person {person_id} not found"
                )

            pair_id = person_result.data[0]["pair_id"]
            person_name = person_result.data[0]["name"]

            # Save and process new image
            temp_filepath = await save_uploaded_image(image)

            # Generate new embedding
            face_service = get_face_recognition_service()
            new_embedding = face_service.generate_embedding(temp_filepath)

            if not new_embedding:
                if os.path.exists(temp_filepath):
                    os.remove(temp_filepath)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="No face detected in new image"
                )

            # Upload new image
            image_url = await upload_to_supabase_storage(temp_filepath, pair_id, person_name)
            update_data["image_url"] = image_url

            # Update embedding
            supabase.table("face_embeddings") \
                .update({"embedding": new_embedding}) \
                .eq("person_id", person_id) \
                .execute()

            # Clean up
            if os.path.exists(temp_filepath):
                os.remove(temp_filepath)

        if not update_data:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No fields to update"
            )

        # Update person
        result = supabase.table("people") \
            .update(update_data) \
            .eq("id", person_id) \
            .execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )

        logger.info(f"Person {person_id} updated successfully")

        return PersonInfo(**result.data[0])

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating person: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update person: {str(e)}"
        )

@router.delete("/deletePerson", response_model=SuccessResponse)
async def delete_person(person_id: int):
    """
    Delete a person from database

    - **person_id**: Person ID
    """
    try:
        logger.info(f"Deleting person {person_id}")

        supabase = get_supabase_client()

        # Delete embeddings first (foreign key constraint)
        supabase.table("face_embeddings").delete().eq("person_id", person_id).execute()

        # Delete person
        result = supabase.table("people").delete().eq("id", person_id).execute()

        if not result.data or len(result.data) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Person {person_id} not found"
            )

        logger.info(f"Person {person_id} deleted successfully")

        return SuccessResponse(message=f"Person {person_id} deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting person: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete person: {str(e)}"
        )
