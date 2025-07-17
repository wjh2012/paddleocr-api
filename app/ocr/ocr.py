from paddleocr import PaddleOCR

ocr = PaddleOCR(
    text_detection_model_name="PP-OCRv5_mobile_det",
    text_detection_model_dir="models/det/PP-OCRv5_mobile_det_infer",
    text_recognition_model_name="korean_PP-OCRv5_mobile_rec",
    text_recognition_model_dir="models/rec/korean_PP-OCRv5_mobile_rec_infer",
    use_doc_orientation_classify=False,
    use_doc_unwarping=False,
    use_textline_orientation=False)

def predict(image):
    result = ocr.predict(
        input=image)

    for res in result:
        # res.print()
        res.save_to_img("output")
        # res.save_to_json("output")

    return result