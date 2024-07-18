import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):
    """
    A class used to define the architecture of our Object Detector

    - In this experiment, we use two-stage-detector FasterRCNN for object detection 
    with the pretrained ResNet50 as the feature pyramid network backbone for feature 
    extraction

    Attributes
    ----------
    num_classes: int
        - The size of the classification head of FasterRCNN used to determine how many
        classes of object the model should classify
        - In this case, num_classes = 2 because we want to detect only Strawberry and background
    backbone : ResNet50
        - Feature Pyramid Network used to extract features which will be passed forward
        as features to FasterRCNN to detect objects.
    nms_threshold : float
        - Non-minimum suppression threshold: used to select the detected bounding 
        boxes that best fit to an object in an image.

    Methods
    -------
    detect(img)
        Generate a set of bounding boxes indicating the detections in the input image
    """

    def __init__(self, num_classes, nms_threshold = 0.5):
        """
        Parameters
        ----------
        num_classes: int
            - The size of the classification head of FasterRCNN used to determine how many
            classes of object the model should classify
            - In this case, num_classes = 2 because we want to detect only Strawberry and background
        nms_threshold : float
            - Non-minimum suppression threshold: used to select the detected bounding 
            boxes that best fit to an object in an image.
        """
        backbone = resnet_fpn_backbone('resnet50', pretrained=True)
        # super(FRCNN_FPN, self).__init__(backbone, num_classes)
        super(FRCNN_FPN, self).__init__(backbone, num_classes)
        self.roi_heads.nms_thresh = nms_threshold
    
    def detect(self, img):
        """
        Generate a set of bounding boxes indicating the detections in the input image

        Parameters
        ----------
        image : torch.Tensor([3, H, W]))
            - Input image we want to run the detection on
            - H: Height of the image
            - W: Width of the image

        Returns
        -------
        bounding_boxes: torch.Tensor([N, 4])
            - Set of Detections in the input image described as bounding boxes
            - Bounding box dimension: (x_min, x_max, y_min, y_max)
        scores: torch.Tensor([N, ])
            - Set of probability describing the objectness of each bounding boxes
        
        """
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]
        return detections['boxes'].detach().cpu(),  detections['scores'].detach().cpu()