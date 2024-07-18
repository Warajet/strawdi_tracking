import csv
import os
import torch
from PIL import Image

from torchvision.ops import masks_to_boxes
from torchvision.transforms.functional import pil_to_tensor
from tracker.utils import obj_detection_transforms

def sort_files(file_list):
    """
    Sort the list of files based on the number in the filename
    
    - Used to sort the list of image, segmentation masks, and the images from
    tested video sequence (i.e. test.mp4)

    Parameters
    ----------
    file_list : list(str)
        - List of the absolute path to each file in a directory

    Returns
    -------
    list(str):
        - Sorted version of a list of the absolute path to each file in a directory 
    """
    # Function to extract the numerical part of the filename
    def extract_number(filename):
        return int(os.path.splitext(filename)[0])
    
    file_list.sort(key=lambda x: extract_number(os.path.basename(x)))
    return file_list


class StrawDIObjDetect(torch.utils.data.Dataset):
    """
    A class used to represent StrawDI Dataset in Pytorch

    - This class will be passed into DataLoader used for training, validating, and
    test the strawberry detection model.
    

    Attributes
    ----------
    root : str
        - Path to the corresponding StrawDI dataset
    transforms : torchvision.transforms
        - Series of transforms object used to transform or augment data for training 
        or inference of strawberry detection (default None)
    classes: tuple
        - Tuple used define the set of classes in the dataset
    img_paths: list
        - List of absolute path to each of image sample (dimension: 756 x 1008 x 3)
    tgt_paths: list
        - List of absolute path to each of ground truth segmentation mask (dimension: 756 x 1008 x 3)
        - In the ground truth images, the value 0 is used for non-strawberry pixels, and strawberry 
        pixels have the index of their strawberry, from 1 to the number of strawberries in the image.

    Methods
    -------
    num_classes():
        - Return number of classes available in the StrawDI dataset
    
    _get_annotation(idx):
        - Get the list of bounding boxes corresponding to the image at the specific index
    
    __getitem__(idx):
        - Get the pair of an image and bounding boxes corresponding to the image at the specific index
    
    __len__():
        - Get the number of samples in the Dataset object

    print_results(results, ovthresh):
        - Generate the multiple strawberry detection performance
    
    """
    def __init__(self, root, transforms = None):
        """
        Parameters
        ----------
        root : str
          - Path to the corresponding StrawDI dataset
        transforms : torchvision.transforms
          - Series of transforms object used to transform or augment data for training 
        or inference of strawberry detection (default None)
        """
        self.root = root
        self.transforms = transforms
        self._classes = ('background', 'strawberry')

        image_dir = os.path.join(root, 'img')
        label_dir = os.path.join(root, 'label')

        self._img_paths = sort_files([os.path.join(image_dir, image_path) for image_path in os.listdir(image_dir)])
        self._tgt_paths = sort_files([os.path.join(label_dir, label_path) for label_path in os.listdir(label_dir)])

    @property
    def num_classes(self):
        """
        Return number of classes available in the StrawDI dataset
        """
        return len(self._classes)
    
    def _get_annotation(self, idx):
        """
        Get the list of bounding boxes corresponding to the image at the specific index

        Parameters
        ----------
        idx : int
            The index of the groundtruth label in the list
        
        Returns
        -------
        boxes: list(torch.Tensor([N, 4]))
            - List of bounding boxes extracted from the segmentation mask
            - N = number of bounding boxes extracted from the segmentation mask
            - each tensor contains (x_min, y_min, x_max, y_max)
        
        labels: torch.Tensor([N])
            - Tensor filled with 1 corresponding to the detected strawberry
        
        image_id: torch.Tensor([1])
            - Index of the image in the groundtruth label list

        iscrowd: torch.Tensor([N])
            - Tensor filled with 1 indicating whether there is a crowded of strawberry
        """
        # Note: 
        # - Commented this code because we need to evaluate the strawberry detection
        # model on the test set (the dataset has available groundtruth)
        # - For inference on unseen data, the below code should be un-commented 

        # if 'test' in self.root:
        #     num_objs = 0
        #     boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        #     return {'boxes': boxes,
        #             'labels': torch.ones((num_objs, ), dtype=torch.int64),
        #             'image_id': torch.tensor([idx]),
        #             'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
        #             'iscrowd': torch.zeros((num_objs, ), dtype=torch.int64, )
        #             }
        
        tgt_path = self._tgt_paths[idx]

        # Convert PIL Image to torch.Tensor
        train_label = pil_to_tensor(Image.open(tgt_path).convert('RGB'))

        # Check if every entries of the groundtruth segmentation mask are identical
        is_same_matrix  = (train_label[0, :,:] == train_label[1, :,:]).all() and (train_label[0, :,:] == train_label[2, :,:]).all()

        # If every entries across every channels are identical
        # - we convert the segmentation masks to a list of bounding boxes on the image
        if is_same_matrix:
            unique_elements = torch.unique(train_label[0, :,:])
            # Initialize the segmentation mask filled with zeros
            train_masks = torch.zeros((unique_elements.shape[0], train_label.shape[1], train_label.shape[2]))

            # Create binary masks for each unique identity
            for i, val in enumerate(unique_elements):
                train_masks[i, :, :] = (train_label[0, :, :] == val)

            # Background at Channel 0 --> Select only train masks for the object of interest
            train_mask_foreground = train_masks[1:, :, :]
            num_objs = train_mask_foreground.shape[2]
            bounding_boxes = masks_to_boxes(train_mask_foreground > 0)

            # Get the current bounding box
            x_min, y_min = bounding_boxes[:, 0], bounding_boxes[:, 1]
            x_max, y_max = bounding_boxes[:, 2], bounding_boxes[:, 3]
            width, height  = x_max - x_min, y_max - y_min

        return {'boxes': bounding_boxes,
                'labels': torch.ones((num_objs, ), dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': (width) * (height),
                'iscrowd': torch.zeros((num_objs, ), dtype=torch.int64, )
                }

    def __getitem__(self, idx):
        """
        Get the pair of an image and bounding boxes corresponding to the image at the specific index

        Parameters
        ----------
        idx : int
            The index of the image and groundtruth label in the list
        
        Returns
        -------
        img: list(torch.Tensor([3, H, W]))
            - the image tensor at the specified index in the image path list
            - H: Height of the image (756)
            - W: Width of the image (1008)
        
        target: dict
            - Dictionary containing the values described in the function _get_annotation
        
        """

        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root, self._img_paths[idx])

        img = Image.open(img_path).convert('RGB')

        target = self._get_annotation(idx)

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = pil_to_tensor(img)

        return img, target
    

    def __len__(self):
        """
        Get the number of samples in the Dataset object
        
        Returns
        -------
        int:
            - number of samples in the Dataset
        
        """
        return len(self._img_paths)
    
    def print_eval(self, results, ovthresh=0.5):
        """
        Generate the multiple strawberry detection performance

        Evaluation Metrics: 
          - Average Precision (ap)
          - Precision (precision)
          - Recall (recall)
          - True Positives (tp)
          - False Positives (fp)
        
        Parameters
        ----------
        results : list(dict)
            - The list of detection predictions used to evaluate the detection performance
            - e.g. results[idx]['boxes'] = N x 4 array of detections in (x1, y1, x2, y2)
        
        Returns
        -------
        dict: 
            - ap: float
              - Average Precision
            - precision: float
              - Precision: TP / (TP + FP) percentage of your results which are relevant
            - recall: float
              - Recall:  TP / (TP + FN) the percentage of total relevant results correctly 
              classified by your algorithm
            - tp: float
              - Number of true positive detections
            - fp: float
              - Number of false positive detections
  
        """

        # Lists for tp and fp in the format tp[cls][image]
        tp = [[] for _ in range(len(self._img_paths))]
        fp = [[] for _ in range(len(self._img_paths))]

        npos = 0
        gt = []
        gt_found = []

        for idx in range(len(self._img_paths)):
            annotation = self._get_annotation(idx)
            bbox = annotation['boxes']
            found = torch.zeros(bbox.shape[0])
            gt.append(bbox.cpu())
            gt_found.append(found)

            npos += found.shape[0]

        # Loop through all images
        # for res in results:
        for im_index, (im_gt, found) in enumerate(zip(gt, gt_found)):
            # Loop through dets an mark TPs and FPs
            im_det = results[im_index]['boxes'].cpu()

            im_tp = torch.zeros(len(im_det))
            im_fp = torch.zeros(len(im_det))
            for i, d in enumerate(im_det):
                ovmax = -torch.inf
                if im_gt.shape[0] > 0:
                    # compute overlaps
                    # intersection
                    ixmin = torch.maximum(im_gt[:, 0], d[0])
                    iymin = torch.maximum(im_gt[:, 1], d[1])
                    ixmax = torch.minimum(im_gt[:, 2], d[2])
                    iymax = torch.minimum(im_gt[:, 3], d[3])
                    iw = torch.maximum(ixmax - ixmin + 1., torch.tensor(0.0))
                    ih = torch.maximum(iymax - iymin + 1., torch.tensor(0.0))
                    inters = iw * ih

                    # union
                    uni = ((d[2] - d[0] + 1.) * (d[3] - d[1] + 1.) +
                            (im_gt[:, 2] - im_gt[:, 0] + 1.) *
                            (im_gt[:, 3] - im_gt[:, 1] + 1.) - inters)

                    overlaps = inters / uni
                    ovmax = torch.max(overlaps)
                    jmax = torch.argmax(overlaps)

                if ovmax > ovthresh:
                    if found[jmax] == 0:
                        im_tp[i] = 1.
                        found[jmax] = 1.
                    else:
                        im_fp[i] = 1.
                else:
                    im_fp[i] = 1.

            tp[im_index] = im_tp
            fp[im_index] = im_fp

        # Flatten out tp and fp into a numpy array
        i = 0
        for im in tp:
            if type(im) != type([]):
                i += im.shape[0]

        tp_flat = torch.zeros(i)
        fp_flat = torch.zeros(i)

        i = 0
        for tp_im, fp_im in zip(tp, fp):
            if type(tp_im) != type([]):
                s = tp_im.shape[0]
                tp_flat[i:s+i] = tp_im
                fp_flat[i:s+i] = fp_im
                i += s

        # print(f"tp_flat: {tp_flat} shape: {tp_flat.shape}")
        # print(f"fp_flat: {fp_flat} shape: {fp_flat.shape}")

        tp = torch.cumsum(tp_flat, dim = 0)
        fp = torch.cumsum(fp_flat, dim = 0)
        rec = tp / float(npos)
        # avoid divide by zero in case the first detection matches a difficult
        # ground truth (probably not needed in my code but doesn't harm if left)
        eps = torch.tensor(torch.finfo(torch.float64).eps, dtype=torch.float64)
        prec = tp / torch.maximum(tp + fp, eps)
        tmp = torch.maximum(tp + fp, eps)

        # correct AP calculation
        # first append sentinel values at the end
        mrec = torch.concatenate((torch.tensor([0.]), rec, torch.tensor([1.])))
        mpre = torch.concatenate((torch.tensor([0.]), prec, torch.tensor([0.])))

        # compute the precision envelope
        for i in range(mpre.shape[0] - 1, 0, -1):
            mpre[i - 1] = torch.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = torch.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = torch.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

        tp, fp, prec, rec, ap = torch.max(tp), torch.max(fp), prec[-1], torch.max(rec), ap

        return {'ap': ap, 'precision': prec, 'recall': rec, 'tp': tp, 'fp': fp}
    

class StrawDIObjDetectSmall(StrawDIObjDetect):
    """
    A Small version of StrawDIObjDetect used for testing the training pipeline with small
    number of samples

    --> Goal: save time for setting up training pipeline by training the model on 
    small batch of data. Once the model perform works (good performance on training but poor
    performance at validation),  we can scale up to larger dataset later.

    Attributes
    ----------
    num_samples: int
    - Number of samples to have in the smaller version of StrawDI dataset

    """
    def __init__(self, root, num_samples, transforms = None):
        super(StrawDIObjDetectSmall, self).__init__(root, transforms)

        image_dir = os.path.join(root, 'img')
        label_dir = os.path.join(root, 'label')

        self._img_paths = [os.path.join(image_dir, image_path) for image_path in os.listdir(image_dir)]
        self._tgt_paths = [os.path.join(label_dir, label_path) for label_path in os.listdir(label_dir)]

        self._img_paths = self._img_paths[: num_samples]
        self._tgt_paths = self._tgt_paths[: num_samples]

    
    