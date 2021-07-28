import cv2
import imutils
import time

def intial_setup():
    global net, name 
    weights = "yolo/yolov4.weights"
    config = "yolo/yolov4.cfg"
    labelsPath = "yolo/coco.names"
    name = open(labelsPath).read().strip().split("\n")  
    net = cv2.dnn_DetectionModel(config, weights)
    net.setInputSize(416, 416)
    net.setInputScale(1.0 / 255)
    net.setInputSwapRB(True)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

def check_distance(x,  y):
    dist = ((x[0] - y[0]) ** 2 + 500 / ((x[1] + y[1]) / 2) * (x[1] - y[1]) ** 2) ** 0.5    #Euclidean distance with depth calibration
    midpoint = (x[1] + y[1]) / 2       
    
    if 0 < dist < 0.25 * midpoint:
        return True
    else:
        return False

def image_processing(image):

    global processedImg
    (H, W) = (None, None)
    frame = image.copy()
    if W is None or H is None:
        (H, W) = frame.shape[:2]

    confidences1 = []
    outline = []
    # COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]
    classes, confidences, boxes= net.detect(frame, confThreshold=0.1, nmsThreshold=0.4)
    if(not len(classes) == 0):
        for classId, confidence, box in zip(classes.flatten(), confidences.flatten(), boxes):
            if(name[classId]=="person"):
                (x, y, width, height) = box.astype("int")
                x = int(x)
                y = int(y)
                outline.append([x, y, int(width), int(height)])
                confidences1.append(float(confidence))

    box_line = cv2.dnn.NMSBoxes(outline, confidences1, 0.4, 0.5)
    
    if len(box_line) > 0:
        flat_box = box_line.flatten()
        pairs = []
        center = []
        status = [] 
        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            center.append([int(x + w / 2), int(y + h / 2)])
            status.append(False)

        for i in range(len(center)):
            for j in range(len(center)):
                close = check_distance(center[i], center[j])

                if close:
                    pairs.append([center[i], center[j]])
                    status[i] = True
                    status[j] = True
        index = 0

        for i in flat_box:
            (x, y) = (outline[i][0], outline[i][1])
            (w, h) = (outline[i][2], outline[i][3])
            if status[index] == True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 150), 2)
            elif status[index] == False:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            index += 1
        for h in pairs:
            cv2.line(frame, tuple(h[0]), tuple(h[1]), (0, 0, 255), 2)
    processedImg = frame.copy()

def main():
    frame_number = 0
    filename = "videos/example.mp4"
    cap = cv2.VideoCapture(filename)
    # cap = cv2.VideoCapture(0)
    # size = (480,374)
    # fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, size)
    intial_setup()
    while(cap.isOpened()):
        ret, frame = cap.read()
        if not ret:
            break
        timer = time.time()

        current_img = frame.copy()
        current_img = imutils.resize(current_img, width=480)
        video = current_img.shape
        frame_number += 1
        Frame = current_img

        image_processing(current_img)
        Frame = processedImg
        out.write(processedImg)
        print(processedImg.shape)
        print('[Info] Time Taken: {} | FPS: {}'.format(time.time() - timer, 1/(time.time() - timer)), end='\r')
        cv2.imshow('Frame',processedImg)

        cv2.waitKey(1)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    main()