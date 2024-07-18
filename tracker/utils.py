import os
import cv2

from tqdm.auto import tqdm
import torch
from torchvision.transforms import functional as F
from torchvision.models.detection.roi_heads import fastrcnn_loss
from torchvision.models.detection.rpn import concat_box_prediction_layers
import math
import time

import random

def evaluate_obj_detection(model, dataloader):
    """
    Perform the evaluation on the object detection model on 5
    evaluation metrics including:
    - Average Precision
    - Precision
    - Recall
    - True Positives
    - False Positives

    Parameters
    ----------
    model : torchvision.models.detection
      - The object detection model of choice
    dataloader : torch.utils.data.DataLoader
      - Dataloader we used to test our detection performance

    Returns
    -------
    dict
      - Dictionary containing the values of the aforementioned evaluation metrics
    """
    model.eval()
    device = list(model.parameters())[0].device
    results = {}
    for imgs, targets in tqdm(dataloader):
        imgs = [img.to(device) for img in imgs]
        with torch.no_grad():
            preds = model(imgs)
        
        for pred, target in zip(preds, targets):
            results[target['image_id'].item()] = {'boxes': pred['boxes'].cpu(), 
                                                  'scores': pred['scores'].cpu()
                                                  }
            
    eval_results = dataloader.dataset.print_eval(results)
    return eval_results


def obj_detection_transforms(train):
    """
    Transform and augment the data ready for training the object detector

    Transform the data ready for validation and test the object detector

    Parameters
    ----------
    train : bool
      - Toggle indicating whether to apply transforms for training

    Returns
    -------
    Compose([Transform])
      - Sequence of the Transform object applied on the Dataset
    """
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(RandomHorizontalFlip(0.5))
    return Compose(transforms)


def train_one_epoch(model, optimizer, dataloader, device, epoch, print_freq, lr_scheduler = None, tb_writer = None, scaler = None):
    """
    Training pipeline per epoch for our object detect

    Parameters
    ----------
    model : torchvision.models.detection
      - The object detection model of choice
    optimizer: torch.optim
      - Optimization algorithm used to update model parameters
    dataloader : torch.utils.data.DataLoader
      - Dataloader we used to train our detection performance
    device: torch.device
      - Device for the training pipeline to run on
    epoch: int
      - The number of current epoch
    print_freq:
      - Frequency to print out the training loss
    lr_scheduler: torch.optim.lr_scheduler, optional
      - Scheduler used to adjust the learning rate based on the number of epochs
    tb_writer: torch.utils.SummaryWriter, optional
      - Summary Writer to log/record the progress of training process
    scaler: torch.cuda.amp.GradScaler, optional
      - Gradient scaler used to improves convergence for networks 
      with float16 gradients by minimizing gradient underflow


    Returns
    -------
    last_loss: float
      - Final loss function value after 1 epoch is completed
    """
    model.train()
    device = list(model.parameters())[0].device
    running_loss = 0.
    last_loss = 0.
    results = {}

    start_time = time.time()

    for i, data in enumerate(dataloader):
        # Setup Optimizre
        optimizer.zero_grad()
        imgs, targets = data
        imgs = [img.to(device) for img in imgs]

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]

        loss_dict = model(imgs, targets)
        losses = sum(loss for loss in loss_dict.values())
    
        loss_value = losses.item()
        
        # Update parameters
        if scaler is not None:
            scaler.scale(losses).backward()
            scaler.step(optimizer)
            scaler.update()
        else:   
            losses.backward()
            optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        running_loss += loss_value
        if (i + 1) % print_freq == 0:
            last_loss= running_loss / print_freq # Loss per batch

            current_time = time.time()
            elapsed_time = current_time - start_time
            start_time = current_time
            
            if i == len(dataloader) - 1:
                print(f"epoch: [{epoch['cur_epoch']}/{epoch['total_epochs']}] [{len(dataloader.dataset)}/{len(dataloader.dataset)}] time: {elapsed_time:.2f} sec. Loss: {last_loss:.4f}")
            else:
                print(f"epoch: [{epoch['cur_epoch']}/{epoch['total_epochs']}] [{(i + 1) * len(imgs)}/{len(dataloader.dataset)}] time: {elapsed_time:.2f} sec. Loss: {last_loss:.4f}")

            if tb_writer is not None:
                tb_writer.add_scalar('Detection Loss/ Train', last_loss, epoch * len(dataloader) + i + 1)
            running_loss = 0.
    
    return last_loss


def eval_forward(model, images, targets):
    # Note: This function is copied from https://github.com/pytorch/vision/blob/f40c8df02c197d1a9e194210e40dee0e6a6cb1c3/torchvision/models/detection/generalized_rcnn.py#L46
    # I copied this function because we need to perform validation to find optimal set of 
    # parameters for our object detecor but fasterRCNN cannot return the loss value 
    # once it is set to eval mode --> using this function solves the problem
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[str, Tensor]]): ground-truth boxes present in the image (optional)
    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            It returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).
    """
    model.eval()

    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])
    model.rpn.training=True
    #model.roi_heads.training=True


    #####proposals, proposal_losses = model.rpn(images, features, targets)
    features_rpn = list(features.values())
    objectness, pred_bbox_deltas = model.rpn.head(features_rpn)
    anchors = model.rpn.anchor_generator(images, features_rpn)

    num_images = len(anchors)
    num_anchors_per_level_shape_tensors = [o[0].shape for o in objectness]
    num_anchors_per_level = [s[0] * s[1] * s[2] for s in num_anchors_per_level_shape_tensors]
    objectness, pred_bbox_deltas = concat_box_prediction_layers(objectness, pred_bbox_deltas)
    # apply pred_bbox_deltas to anchors to obtain the decoded proposals
    # note that we detach the deltas because Faster R-CNN do not backprop through
    # the proposals
    proposals = model.rpn.box_coder.decode(pred_bbox_deltas.detach(), anchors)
    proposals = proposals.view(num_images, -1, 4)
    proposals, scores = model.rpn.filter_proposals(proposals, objectness, images.image_sizes, num_anchors_per_level)

    proposal_losses = {}
    assert targets is not None
    labels, matched_gt_boxes = model.rpn.assign_targets_to_anchors(anchors, targets)
    regression_targets = model.rpn.box_coder.encode(matched_gt_boxes, anchors)
    loss_objectness, loss_rpn_box_reg = model.rpn.compute_loss(
        objectness, pred_bbox_deltas, labels, regression_targets
    )
    proposal_losses = {
        "loss_objectness": loss_objectness,
        "loss_rpn_box_reg": loss_rpn_box_reg,
    }

    #####detections, detector_losses = model.roi_heads(features, proposals, images.image_sizes, targets)
    image_shapes = images.image_sizes
    proposals, matched_idxs, labels, regression_targets = model.roi_heads.select_training_samples(proposals, targets)
    box_features = model.roi_heads.box_roi_pool(features, proposals, image_shapes)
    box_features = model.roi_heads.box_head(box_features)
    class_logits, box_regression = model.roi_heads.box_predictor(box_features)

    result: List[Dict[str, torch.Tensor]] = []
    detector_losses = {}
    loss_classifier, loss_box_reg = fastrcnn_loss(class_logits, box_regression, labels, regression_targets)
    detector_losses = {"loss_classifier": loss_classifier, "loss_box_reg": loss_box_reg}
    boxes, scores, labels = model.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
    num_images = len(boxes)
    for i in range(num_images):
        result.append(
            {
                "boxes": boxes[i],
                "labels": labels[i],
                "scores": scores[i],
            }
        )
    detections = result
    detections = model.transform.postprocess(detections, images.image_sizes, original_image_sizes)  # type: ignore[operator]
    model.rpn.training=False
    model.roi_heads.training=False
    losses = {}
    losses.update(detector_losses)
    losses.update(proposal_losses)
    return losses, detections


def validate_obj_detector(model, val_dataloader, device):
    """
    Validating our object detect to find optimal paremters while
    training on multiple epochs

    Parameters
    ----------
    model : torchvision.models.detection
      - The object detection model of choice
    val_dataloader : torch.utils.data.DataLoader
      - Dataloader we used to validate our detection performance
    device: torch.device
      - Device for the training pipeline to run on

    Returns
    -------
    validation_loss: float
      - Validation loss function value after predicting every samples
      in the validation set
    """
    model.eval()
    running_loss = 0.
    for i, data in enumerate(val_dataloader):
        imgs, targets = data
        imgs = [img.to(device) for img in imgs]

        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in t.items()} for t in targets]
        loss_dict, detections = eval_forward(model, imgs, targets) 

        losses = sum(loss for loss in loss_dict.values())

        running_loss += losses.item()

    validation_loss =  running_loss / len(val_dataloader)
    return validation_loss


def filter_bboxes_by_prob(predictions, prob_threshold):
    """
    Filter out bounding boxes with the probability lower than the threshold
    
    Parameters
    ----------
    predictions : list(torch.Tensor)
      - Predicted bounding boxes torch.tensor([N, 4]) and the
      probability associated to it torch.tensor([N, ])
    prob_threshold : float
      - Probability threshold to select the bounding boxes

    Returns
    -------
    filtered_bboxes: torch.tensor([N, 4])
      - List of bounding boxes with the probability > prob_threshold
    filtered_probs: torch.tensor([N, ])
      - List of probabilitues corresponding to filtered_bboxes
    """
    bounding_boxes, probs = predictions[0], predictions[1]
    # Apply the mask to the bounding boxes and probabilities
    filtered_bboxes = bounding_boxes[probs > prob_threshold]
    filtered_probs = probs[probs > prob_threshold]

    return filtered_bboxes, filtered_probs


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    def __init__(self, prob):
        self.prob = prob

    def __call__(self, image, target):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]
            target["boxes"] = bbox
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

        return image, target


class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

