import cv2

def visualize(image, results, labels):
    for obj in results:
        # Draw bounding box
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * image.shape[1])
        xmax = int(xmax * image.shape[1])
        ymin = int(ymin * image.shape[0])
        ymax = int(ymax * image.shape[0])
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (10, 255, 0), 2)

        # Draw label
        class_id = int(obj['class_id'])
        label = "{}: {:.0f}%".format(labels[class_id], obj['score'] * 100)
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        label_ymin = max(ymin, labelSize[1] + 10)
        cv2.rectangle(image, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0],
            label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
        cv2.putText(image, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
            (0, 0, 0), 2)

    return image