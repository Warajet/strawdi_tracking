import collections

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from scipy.optimize import linear_sum_assignment as linear_assignment

from tracker.utils import filter_bboxes_by_prob
from tracker import utils

import warnings

def quiet_divide(a, b):
    # Note: This function is copied from py-motmetrics library due to the
    # dependencies problem (motmetrics not supported by the current Python version on Colab (3.10))
    # For further details, Please consult: https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/math_util.py
    """Quiet divide function that does not warn about (0 / 0)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        return np.true_divide(a, b)


def rect_min_max(r):
    # Note: This function is copied from py-motmetrics library due to the
    # dependencies problem (motmetrics not supported by the current Python version on Colab (3.10))
    # For further details, Please consult: https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/distances.py
    min_pt = r[..., :2]
    size = r[..., 2:]
    max_pt = min_pt + size
    return min_pt, max_pt


def boxiou(a, b):
    # Note: This function is copied from py-motmetrics library due to the
    # dependencies problem (motmetrics not supported by the current Python version on Colab (3.10))
    # For further details, Please consult: https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/distances.py
    """Computes IOU of two rectangles."""
    a_min, a_max = rect_min_max(a)
    b_min, b_max = rect_min_max(b)
    # Compute intersection.
    i_min = np.maximum(a_min, b_min)
    i_max = np.minimum(a_max, b_max)
    i_size = np.maximum(i_max - i_min, 0)
    i_vol = np.prod(i_size, axis=-1)
    # Get volume of union.
    a_size = np.maximum(a_max - a_min, 0)
    b_size = np.maximum(b_max - b_min, 0)
    a_vol = np.prod(a_size, axis=-1)
    b_vol = np.prod(b_size, axis=-1)
    u_vol = a_vol + b_vol - i_vol
    return np.where(i_vol == 0, np.zeros_like(i_vol, dtype=np.float64),
                    quiet_divide(i_vol, u_vol))


def iou_matrix(objs, hyps, max_iou=1., return_dist=True):
    # Note: This function is copied from py-motmetrics library due to the
    # dependencies problem (motmetrics not supported by the current Python version on Colab (3.10))
    # For further details, Please consult: https://github.com/cheind/py-motmetrics/blob/develop/motmetrics/distances.py
    """Computes 'intersection over union (IoU)' distance matrix between object and hypothesis rectangles.

    The IoU is computed as

        IoU(a,b) = 1. - isect(a, b) / union(a, b)

    where isect(a,b) is the area of intersection of two rectangles and union(a, b) the area of union. The
    IoU is bounded between zero and one. 0 when the rectangles overlap perfectly and 1 when the overlap is
    zero.

    Params
    ------
    objs : Nx4 array
        Object rectangles (x,y,w,h) in rows
    hyps : Kx4 array
        Hypothesis rectangles (x,y,w,h) in rows

    Kwargs
    ------
    max_iou : float
        Maximum tolerable overlap distance. Object / hypothesis points
        with larger distance are set to np.nan signalling do-not-pair. Defaults
        to 0.5
    return_dist : bool
        If true, return distance matrix. If false, return similarity (IoU) matrix.

    Returns
    -------
    C : NxK array
        Distance matrix containing pairwise distances or np.nan.
        if `return_dist` is False, then the matrix contains the pairwise IoU.
    """

    if np.size(objs) == 0 or np.size(hyps) == 0:
        return np.empty((0, 0))

    objs = np.asfarray(objs)
    hyps = np.asfarray(hyps)
    assert objs.shape[1] == 4
    assert hyps.shape[1] == 4
    iou = boxiou(objs[:, None], hyps[None, :])
    if return_dist:
        dist = 1 - iou
        return np.where(dist > max_iou, np.nan, dist)
    return iou


class Tracker:
    """
    Base class describing the overall behaviour of our Multiple Object Tracker

    Attributes
    ----------
    obj_detector : torchvision.models.detection
        - Object Detector used to generate detections at each frame
    prob_threshold : float
        - Threshold value used to select the bounding box with the objectness probability
        higher than the threshold
    tracks: list(Track)
        - List of Track object
    track_num: int
        - Number of tracks in the video sequence --> i.e. number of strawberries in the video
    im_index: int
        - Index of an image in the video sequence
    results: dict
        - the result dictionary containing the track corresponding to each object
        - track contains the list of frame as the key and its corresponding bounding box as value
    

    Methods
    -------
    reset(hard):
        - reset the tracks before staring over a new sequence
    
    add(new_boxes, new_scores):
        - Initializes new Track objects and saves them
    
    get_pos():
        - Get the positions of all active tracks

    data_association(boxes, scores):
        - Associate the data in the tracks to the detections in the new frame
    
    step(frame):
        - Perform tracking with a blob containing the image information
    
    get_results():
        - Get the results describing the track corresponding to each object

    """

    def __init__(self, obj_detector, prob_threshold = 0.75):
        """
        Parameters
        ----------
        obj_detector : torchvision.models.detection
          - Object Detector used to generate detections at each frame
        prob_threshold : float
          - Threshold value used to select the bounding box with the objectness probability
        higher than the threshold (default = 0.75)
        """
        self.obj_detector = obj_detector
        self.prob_threshold = prob_threshold
        self.tracks = []
        self.track_num = 0
        self.im_index = 0
        self.results = {}

    def reset(self, hard=True):
        """
        reset the tracks before staring over a new sequence

        Parameters
        ----------
        hard : bool
          - Toggle to reset the number of tracks, results, and image index
          to initial state
        """
        self.tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def add(self, new_boxes, new_scores):
        """
        Initializes new Track objects and saves them

        Parameters
        ----------
        new_boxes : list(torch.Tensor([4]))
          - List of selected bounding boxes to be added into the new track
        new_scores : list(torch.Tensor([1]))
          - List of probability corresponding to the bounding box 
          to be added into the new track
        """
        num_new = len(new_boxes)
        for i in range(num_new):
            self.tracks.append(Track(
                new_boxes[i],
                new_scores[i],
                self.track_num + i
            ))
        self.track_num += num_new

    def get_pos(self):
        """
        Get the positions of all active tracks

        Returns
        -------
        box: torch.Tensor([N, 4])
            - Position of all active tracks

        """
        if len(self.tracks) == 1:
            box = self.tracks[0].box
        elif len(self.tracks) > 1:
            box = torch.stack([t.box for t in self.tracks], 0)
        else:
            box = torch.zeros(0).cuda()
        return box

    def data_association(self, boxes, scores):
        """
        Associate the data in the tracks to the detections in the new frame

        Parameters
        ----------
        boxes: torch.Tensor([N, 4])
            - Set of Detections in the input image described as bounding boxes
            - Bounding box dimension: (x_min, x_max, y_min, y_max)
        scores: torch.Tensor([N, ])
            - Set of probability describing the objectness of each bounding boxes
        """
        self.tracks = []
        self.add(boxes, scores)

    def step(self, frame):
        """
        This function should be called every timestep to perform tracking with a blob
        containing the image information

        Parameters
        ----------
        image : torch.Tensor([3, H, W]))
            - Input image we want to run the detection on
            - H: Height of the image
            - W: Width of the image
        """
        # object detection
        boxes, scores = filter_bboxes_by_prob(self.obj_detector.detect(frame), self.prob_threshold)

        self.data_association(boxes, scores)

        # results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate([t.box.cpu().numpy(), np.array([t.score])])

        self.im_index += 1

    def get_results(self):
        """
        Get the results describing the track corresponding to each object

        Returns
        -------
        results: dict
            - Dictionary describing the locations of each detection across
            multiple frames
        """
        return self.results

# - If the costs are equal to _UNMATCHED_COST, it's not a match.
_UNMATCHED_COST = 255.0
class HungarianIoUTracker(Tracker):
    """
    Multiple Object Tracker that uses Hungarian algorithm to match the 
    detections between frames based on IoU score between the predictions and
    detections.

    Attributes
    ----------
    Same number of attributes to the base class Tracker

    Methods
    -------
    data_association(boxes, scores):
        - Associate the data in the tracks to the detections in the new frame
        using Hungarian algorithm based on IoU score between the predictions and the detections
    """
    def __init__(self, obj_detector, prob_threshold = 0.75):
        super(HungarianIoUTracker, self).__init__(obj_detector, prob_threshold)

    def data_association(self, boxes, scores):
        """
        Associate the data in the tracks to the detections in the new frame

        Parameters
        ----------
        boxes: torch.Tensor([N, 4])
            - Set of Detections in the input image described as bounding boxes
            - Bounding box dimension: (x_min, x_max, y_min, y_max)
        scores: torch.Tensor([N, ])
            - Set of probability describing the objectness of each bounding boxes
        """
        # Check for non-empty tracks
        if self.tracks:
            track_ids = [t.id for t in self.tracks]
            track_boxes = np.stack([t.box.numpy() for t in self.tracks], axis=0)
            
            # Build cost matrix.
            distance = iou_matrix(track_boxes, boxes.numpy(), max_iou=0.5)

            # Set all unmatched costs to _UNMATCHED_COST.
            distance = np.where(np.isnan(distance), _UNMATCHED_COST, distance)

            # Perform Hungarian matching.
            # Indices to track_boxes and boxes: row_idx and col_idx
            # A match: row_idx[i] and col_idx[i]
            # Cost for the match: distance[row_idx[i], col_idx[i]]
            row_idx, col_idx = linear_assignment(distance)

            # Update existing tracks and remove unmatched tracks.
            remove_track_ids = []
            seen_track_ids = []
            seen_box_idx = []
            # Iterate through each pair of matches
            # box_idx: index of the newly detected bounding boxes
            # track_idx: index of the bounding boxes in the existing track list
            for track_idx, box_idx in zip(row_idx, col_idx):
                costs = distance[track_idx, box_idx] 
                internal_track_id = track_ids[track_idx]
                seen_track_ids.append(internal_track_id)
                # If the match is not really a match
                if costs == _UNMATCHED_COST:
                    remove_track_ids.append(internal_track_id)
                # If it is a match, assign the detection to the existing track
                else:
                    self.tracks[track_idx].box = boxes[box_idx]
                    seen_box_idx.append(box_idx)
            
            # Get the list of existing tracks that are not seen in the new frame
            unseen_track_ids = set(track_ids) - set(seen_track_ids)
            remove_track_ids.extend(list(unseen_track_ids))
            # Remove all the tracks that is not seen in the new frame any longer
            self.tracks = [t for t in self.tracks
                            if t.id not in remove_track_ids]

            # Add new tracks into the existing tracks list
            new_boxes_idx = set(range(len(boxes))) - set(seen_box_idx)
            new_boxes = [boxes[i] for i in new_boxes_idx]
            new_scores = [scores[i] for i in new_boxes_idx]

            self.add(new_boxes, new_scores)
        else:
            # No tracks exist.
            self.add(boxes, scores)


class Track(object):
    """
    A class used to represent an individual Track

    Attributes
    ----------
    id : int
      - the id of the track
    box : list(torch.tensor([4]))
      - the list of bounding boxes across multiple frames
    score : list(torch.tensor([4]))
        the list of probability corresponding to each bounding box
    """

    def __init__(self, box, score, track_id):
        self.id = track_id
        self.box = box
        self.score = score