from imutils.perspective import four_point_transform
import pytesseract
import argparse
import imutils
import cv2
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'



#construct argument parser https://pyimagesearch.com/2021/10/27/automatically-ocring-receipts-and-scans/

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to input receipt image")
ap.add_argument("-d", "--debug", type=int, default=-1, help="whether or not we are visualizing each step of the pipeline")
args = vars(ap.parse_args())

# begin to load our input image, alter it as necessary
#also compare the ratio of new width to the old width

orig = cv2.imread(args["image"])
image = orig.copy()
image = imutils.resize(image, width=500)
ratio = orig.shape[1] / float(image.shape[1])

#convert the image to grayscale, blur it very slightly and then apply edge detection

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5,5,),0)
edged = cv2.Canny(blurred, 75, 200)

if args["debug"] > 0:
    cv2.imshow("Input", image)
    cv2.imshow("Edged", edged)
    cv2.waitKey(0)

# looking for contours in edge map then using sorted() to get them in descending order

conts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
conts = imutils.grab_contours(conts)
conts = sorted(conts,key=cv2.contourArea, reverse=True)

receiptCntOutline = None

for c in conts:
    peri = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.02 * peri, True)

    if len(approx) == 4:
        receiptCntOutline = approx
        break

if receiptCntOutline is None:
    raise Exception("Couldn't find your receipt's outlines")

if args["debug"] > 0:
    output = image.copy()
    cv2.drawContours(output, [receiptCntOutline], -1, (0, 255, 0), 2)
    cv2.imshow("Receipt Outline", output)
    cv2.waitKey(0)

receipt = four_point_transform(orig, receiptCntOutline.reshape(4,2) * ratio)  # we use the original image as it's higher res and doesn't have all the noise from blur, etc.

cv2.imshow("Receipt Transformation", imutils.resize(receipt, width=500))
cv2.waitKey(0)

# going to apply OCR to the receipt now

options = "--psm 4"
text = pytesseract.image_to_string(cv2.cvtColor(receipt, cv2.COLOR_BGR2RGB), config = options)

print("[INFO] raw output: ")
print("------------------")
print(text)
print("\n") 

pricePatern = r"(\$\d+\.\d\d)"

#allPrices = re.findall(pricePatern, text)
#print(allPrices)

print("[INFO] Price line items: ")
print("------------------")

for row in text.split("\n"):
    if re.search(pricePatern, row) is not None:
        print(row)