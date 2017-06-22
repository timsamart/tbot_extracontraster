from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import os

def start(bot, update):
    update.message.reply_text('I am extract contrast bot! send me some picture and I will apply the extract black to white Filter!!')


def hello(bot, update):
    update.message.reply_text(
        'Hello {}'.format(update.message.from_user.first_name))

def applyclaheBGR(img, clipLimit=3., tileGridSize=(8, 8)):
    clahe = cv2.createCLAHE(clipLimit, tileGridSize)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
    l, a, b = cv2.split(lab)  # split on 3 different channels
    l2 = clahe.apply(l)  # apply CLAHE to the L-channel
    lab = cv2.merge((l2, a, b))  # merge channels
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
    return img

def applyotsuBGR(img_orig):
    img_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
    hist, bins = np.histogram(img_orig.flatten(), 256, [0, 256])

    sum1 = np.sum(range(0, 256) * hist)
    total = img_orig.shape[0] * img_orig.shape[1]
    sumB = 0
    wB = 0
    maximum = 0
    threshold1 = 0
    threshold2 = 0
    for i in range(0, 256):
        wB = wB + hist[i]
        if (wB == 0): continue
        wF = total - wB
        if (wF == 0): break
        sumB = sumB + i * hist[i]
        mB = sumB / wB
        mF = (sum1 - sumB) / wF
        between = wB * wF * (mB - mF) * (mB - mF)
        if (between >= maximum):
            threshold1 = i
            if (between > maximum):
                threshold2 = i
            maximum = between

    thresh = (threshold1 + threshold2) / 1.3

    tr = np.zeros(256, dtype=np.uint8)
    for i in range(0, 256):
        if (i < thresh):
            tr[i] = 0
        else:
            tr[i] = 255

    img_res = tr[img_orig]
    img_res = 255-img_res

    return img_res

def applycustomcontrastBGR(img):
    colors = np.array([[37, 1, 1],
                       [120, 1, 1],
                       [1, 1, 1]])
    colotsu = colors * np.array([[1, 255, 240],
                                 [1, 255, 240],
                                 [255, 255, 255]])
    colotsu = colotsu + np.array([[20, 0, 0],
                                  [20, 0, 0],
                                  [0, 0, 0]])
    colotsl = colors * np.array([[1, 50, 150],
                                 [1, 50, 100],
                                 [1, 50, 220]])
    colotsl = colotsl - np.array([[20, 0, 0],
                                  [20, 0, 0],
                                  [0, 0, 0]])

    phi = 1
    theta = 1
    maxInt = 255
    nimg = (maxInt / phi) * (img / (maxInt / theta)) ** 2
    nimg = np.array(nimg, dtype="uint8")
    nimg = (maxInt / phi) * (nimg / (maxInt / theta)) ** 2
    nimg = np.array(nimg, dtype="uint8")
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    img2 = applyclaheBGR(nimg, 3., (8, 8))
    # cv2.imshow("msk0",img2)
    # cv2.waitKey()
    converted = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

    msk0 = cv2.inRange(converted, colotsl[0], colotsu[0])
    msk1 = cv2.inRange(converted, colotsl[1], colotsu[1])
    msk2 = cv2.inRange(converted, colotsl[2], colotsu[2])

    final = 255 - cv2.merge((msk2 - msk0, msk2, msk2 - msk0))
    return final

def getimg(bot, update):
    photo = update.message.photo[-1].file_id
    im = bot.getFile(file_id=photo)
    im.download()
    img = cv2.imread(os.path.basename(im.file_path))

    final=applyotsuBGR(img)

    pil_im = Image.fromarray(final) # actually convert to pil image
    bio = BytesIO()
    bio.name = 'image.jpeg'
    pil_im.save(bio, 'JPEG')
    bio.seek(0)
    update.message.reply_photo(photo=bio)


updater = Updater('361559041:AAFHJzj7wN8Rn_PBMmGt8NNuXyJ9owvB6lQ')

updater.dispatcher.add_handler(CommandHandler('start', start))
updater.dispatcher.add_handler(CommandHandler('hello', hello))
updater.dispatcher.add_handler(MessageHandler(Filters.photo, getimg))

updater.start_polling()
updater.idle()