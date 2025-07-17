from typing import List

import cv2
import numpy as np
from fastapi import APIRouter, UploadFile, File

from app.ocr.ocr import predict

router = APIRouter(
    prefix="/api",
    tags=["/api"],
    responses={404: {"description": "Not found"}},
)


@router.post("/ocr")
async def ocr(file: UploadFile = File(...)):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "이미지 디코딩 실패"}

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    results = predict(image)
    return results[0]["rec_texts"]


@router.post("/ocr/multi")
async def ocr_multiple(files: List[UploadFile] = File(...)):
    image_list = []
    filenames = []
    results = []

    # 1단계: 이미지 디코딩
    for file in files:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            # 디코딩 실패한 경우는 미리 결과에 추가
            results.append({"filename": file.filename, "error": "이미지 디코딩 실패"})
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_list.append(image)
            filenames.append(file.filename)

    # 2단계: OCR 예측 (이미지 리스트 한 번에 처리)
    if image_list:
        try:
            ocr_outputs = predict(image_list)  # image_list를 한 번에 전달
            for filename, output in zip(filenames, ocr_outputs):
                rec_texts = output.get("rec_texts", [])
                results.append({"filename": filename, "rec_texts": rec_texts})
        except Exception as e:
            for filename in filenames:
                results.append({"filename": filename, "error": str(e)})

    return results