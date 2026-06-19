from paddleocr import PaddleOCR

ocr = PaddleOCR(
    use_angle_cls=True,
    lang="en"
)

result = ocr.ocr(
    r"../images/1. Level 1 session 1/page_1.png",
    cls=True
)

for line in result[0]:
    print(line[1][0])