from einops import rearrange
from torchvision.models.feature_extraction import create_feature_extractor
from torch import nn
import torch
from torchvision.ops.misc import FrozenBatchNorm2d
from torchvision import datasets
import pycocotools.cocoeval as cocoeval
from torch.utils.data import DataLoader
import os
import torchvision.ops as ops
from torchvision.ops import nms
import numpy as np
import matplotlib.pyplot as plt
from torch.optim import AdamW
from DETR_losses import compute_sample_loss
import config

"""
Based on https://github.com/dimiz51/DETR-Factory-PyTorch/
"""
def class_based_nms(boxes, probs, iou_threshold=0.5):
    """
    Performs non-maximum suppression (NMS) on bounding boxes to filter out overlapping
    boxes for each class. This is usually not needed for DETR as it usually does not produce
    overlapping boxes (if trained long enough).

    Args:
        boxes (torch.Tensor): Bounding boxes in the format (xmin, ymin, xmax, ymax). Shape: [num_boxes, 4]
        probs (torch.Tensor): Class probabilities for each bounding box. [num_boxes, num_classes]
        iou_threshold (float, optional): IOU threshold for NMS. Defaults to 0.5.

    Returns:
        torch.Tensor: Bounding boxes after NMS.
        torch.Tensor: Predicted class scores after NMS.
        torch.Tensor: Predicted class indices after NMS.
    """

    # Get the class with the highest probability for each box
    scores, class_ids = torch.max(probs, dim=1)

    # Apply NMS
    keep_ids = nms(boxes, scores, iou_threshold)

    # Get the boxes and class scores after NMS
    boxes = boxes[keep_ids]
    scores = scores[keep_ids]
    class_ids = class_ids[keep_ids]

    return boxes, scores, class_ids
def run_inference(
    model,
    device,
    inputs,
    nms_threshold=0.3,
    image_size=480,
    empty_class_id=0,
    out_format="xyxy",
    scale_boxes=True,
):
    """
    Utility function that wraps the inference and post-processing and returns the results for the
    batch of inputs. The inference will be run using the passed model and device while post-processing
    will be done on the CPU.

    Args:
        model (torch.nn.Module): The trained model for inference.
        device (torch.device): The device to run inference on.
        inputs (torch.Tensor): Batch of input images.
        nms_threshold (float, optional): NMS threshold for removing overlapping boxes. Default is 0.3.
        image_size (int, optional): Image size for transformations. Default is 480.
        empty_class_id (int, optional): The class ID representing 'no object'. Default is 0.
        out_format (str, optional): Output format for bounding boxes. Default is "xyxy".
        scale_boxes (bool, optional): Whether to scale the bounding boxes. Default is True.
    Returns:
        List of tuples: Each tuple contains (nms_boxes, nms_probs, nms_classes) for a batch item.
    """
    if model and device:
        model.eval()
        model.to(device)
        inputs = inputs.to(device)
    else:
        raise ValueError("No model or device provided for inference!")

    with torch.no_grad():
        out_cl, out_bbox = model(inputs)

    # Get the outputs from the last decoder layer..
    out_cl = out_cl[:, -1, :]
    out_bbox = out_bbox[:, -1, :]
    out_bbox = out_bbox.sigmoid().cpu()
    out_cl_probs = out_cl.cpu()

    scale_factors = torch.tensor([image_size, image_size, image_size, image_size])
    results = []

    for i in range(inputs.shape[0]):
        o_bbox = out_bbox[i]
        o_cl = out_cl_probs[i].softmax(dim=-1)
        o_bbox = ops.box_convert(o_bbox, in_fmt="cxcywh", out_fmt=out_format)

        # Scale boxes if needed...
        if scale_boxes:
            o_bbox = o_bbox * scale_factors

        # Filter out boxes with no object...
        o_keep = o_cl.argmax(dim=-1) != empty_class_id
        if o_keep.sum() == 0:
            results.append((np.array([]), np.array([]), np.array([])))
            continue
        keep_probs = o_cl[o_keep]
        keep_boxes = o_bbox[o_keep]

        # Apply NMS
        nms_boxes, nms_probs, nms_classes = class_based_nms(
            keep_boxes, keep_probs, nms_threshold
        )
        results.append((nms_boxes, nms_probs, nms_classes))

    return results

def use_frozen_batchnorm(module):
    """Recursively replace all BatchNorm2d layers with FrozenBatchNorm2d."""
    for name, child in module.named_children():
        if isinstance(child, torch.nn.BatchNorm2d):
            # Copy existing parameters from the BatchNorm2d....
            frozen_bn = FrozenBatchNorm2d(child.num_features)
            frozen_bn.weight.data = child.weight.data
            frozen_bn.bias.data = child.bias.data
            frozen_bn.running_mean.data = child.running_mean.data
            frozen_bn.running_var.data = child.running_var.data

            # Replace the layer in the model inplace...
            setattr(module, name, frozen_bn)
        else:
            # Recursively apply to child modules
            use_frozen_batchnorm(child)


class DETR(nn.Module):
    """Detection Transformer (DETR) model with a ResNet50 backbone.

    Paper: https://arxiv.org/abs/2005.12872

    Args:
        d_model (int, optional): Embedding dimension. Defaults to 256.
        n_classes (int, optional): Number of classes. Defaults to 92.
        n_tokens (int, optional): Number of tokens. Defaults to 225.
        n_layers (int, optional): Number of layers. Defaults to 6.
        n_heads (int, optional): Number of heads. Defaults to 8.
        n_queries (int, optional): Number of queries/max objects. Defaults to 100.

    Returns:
        DETR: DETR model
    """

    def __init__(
        self,
        d_model=256,
        n_classes=92,
        n_tokens=225,
        n_layers=6,
        n_heads=8,
        n_queries=100,
        use_frozen_bn=False,
    ):
        super().__init__()

        self.backbone = create_feature_extractor(
            torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True),
            return_nodes={"layer4": "layer4"},
        )

        # Replace BatchNorm2d with FrozenBatchNorm2d...
        # BatchNorm2D makes inference unstable for DETR...
        if use_frozen_bn:
            use_frozen_batchnorm(self.backbone)

        self.conv1x1 = nn.Conv2d(2048, d_model, kernel_size=1, stride=1)

        self.pe_encoder = nn.Parameter(
            torch.rand((1, n_tokens, d_model)), requires_grad=True
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=0.1,
        )

        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )

        self.queries = nn.Parameter(
            torch.rand((1, n_queries, d_model)), requires_grad=True
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            batch_first=True,
            dropout=0.1,
        )

        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Each of the decoder's outputs will be passed through
        # linear layers for prediction of boxes/classes.
        self.linear_class = nn.Linear(d_model, n_classes)
        self.linear_bbox = nn.Linear(d_model, 4)

    def forward(self, x):
        # Pass inputs through the CNN backbone...
        tokens = self.backbone(x)["layer4"]

        # Pass outputs from the backbone through a simple conv...
        tokens = self.conv1x1(tokens)

        # Re-order in patches format
        tokens = rearrange(tokens, "b c h w -> b (h w) c")

        # Pass encoded patches through encoder...
        out_encoder = self.transformer_encoder((tokens + self.pe_encoder))

        # We expand so each image of each batch get's it's own copy of the
        # query embeddings. So from (1, 100, 256) to (4, 100, 256) for example
        # for batch size=4, with 100 queries of embedding dimension 256.
        queries = self.queries.repeat(out_encoder.shape[0], 1, 1)

        # Compute outcomes for all intermediate
        # decoder's layers...
        class_preds = []
        bbox_preds = []

        for layer in self.transformer_decoder.layers:
            queries = layer(queries, out_encoder)
            class_preds.append(self.linear_class(queries))
            bbox_preds.append(self.linear_bbox(queries))

        # Stack and return
        class_preds = torch.stack(class_preds, dim=1)
        bbox_preds = torch.stack(bbox_preds, dim=1)

        return class_preds, bbox_preds




class DETREvaluator:
    def __init__(
        self,
        model: torch.nn.Module,
        coco_dataset: datasets.CocoDetection,
        device: torch.device,
        empty_class_id: int,
        collate_fn: callable,
        nms_iou_threshold: float = 0.5,
        batch_size: int = 2,
        image_size: int = 480,
    ):
        """
        Evaluator for DETR using COCO evaluation metrics.

        Args:
            model (torch.nn.Module): The trained DETR model.
            coco_dataset (torch.utils.data.Dataset): The COCO dataset used for evaluation.
            device (torch.device): The device to run the model on.
            empty_class_id (int): The class ID for the empty class (background).
            collate_fn (callable): The collate function for the DataLoader.
            nms_iou_threshold (float, optional): The IOU threshold for NMS. Defaults to 0.5.
        """
        self.model = model.to(device)
        self.device = device
        self.coco_gt = coco_dataset.coco  # COCO ground truth annotations
        self.empty_class_id = empty_class_id
        self.nms_iou_threshold = nms_iou_threshold
        self.image_size = image_size

        # Create DataLoader and no shuffling
        self.dataloader = DataLoader(
            coco_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
        )

    def evaluate(self):
        """
        Runs evaluation on the dataset and computes COCO metrics.
        """
        self.model.eval()
        results = []

        # Evaluation dataset information
        print(f"Number of images in evaluation COCO dataset: {len(self.coco_gt.imgs)}")
        print(
            f"Number of objects in the evaluation COCO dataset: {len(self.coco_gt.anns)}"
        )

        print(f"Evaluating DETR model on device: {self.device}")

        with torch.no_grad():
            for ix, (input_, (_, _, _, image_ids)) in enumerate(self.dataloader):
                # Extract inference results
                batch_results = run_inference(
                    self.model,
                    self.device,
                    input_,
                    nms_threshold=self.nms_iou_threshold,
                    image_size=self.image_size,
                    empty_class_id=self.empty_class_id,
                    out_format="xywh",  # COCO format for boxes
                    scale_boxes=False,  # We don't want to scale to inference image size as those might differ from the COCO ground truths
                )

                # Process each image in the batch...
                for img_idx, (nms_boxes, nms_probs, nms_classes) in enumerate(
                    batch_results
                ):
                    img_id = image_ids[img_idx].item()

                    # Skip images where no objects are detected
                    if len(nms_boxes) == 0:
                        continue

                    # Get the scaling factors
                    scale_factors = np.array(
                        [
                            self.coco_gt.imgs[img_id]["width"],
                            self.coco_gt.imgs[img_id]["height"],
                            self.coco_gt.imgs[img_id]["width"],
                            self.coco_gt.imgs[img_id]["height"],
                        ],
                        dtype=np.float32,
                    )

                    # Scale the boxes to image size...
                    nms_boxes = nms_boxes * scale_factors

                    # Convert detections to COCO format
                    for j in range(len(nms_classes)):
                        results.append(
                            {
                                "image_id": img_id,
                                "category_id": nms_classes[j].item(),
                                "bbox": nms_boxes[j].tolist(),
                                "score": nms_probs[j].item(),
                            }
                        )

        if len(results) == 0:
            raise ValueError(
                "No objects were found, something could be wrong with the model provided!"
            )

        # Create COCO results object
        coco_dt = self.coco_gt.loadRes(results)

        # Initialize COCO evaluator
        coco_eval = cocoeval.COCOeval(self.coco_gt, coco_dt, iouType="bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        return coco_eval.stats




class DETRTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device,
        epochs: int,
        batch_size: int,
        log_freq: int = 1,
        save_freq: int = 10,
        weight_decay: float = 1e-4,
        checkpoint_dir: str = "ckpts",
        freeze_backbone: bool = False,
        backbone_lr: float = 1e-5,
        transformer_lr: float = 1e-4,
        num_queries: int = 100,
        empty_class_id: int = 0,
    ):
        """
        Initializes the DETR trainer class.

        Public API:
        - train() : Start the training
        - visualize_losses() : Plot the training losses and save plots

        Args:
            model: The DETR model to train
            train_loader: The Data Loader for the training data set
            val_loader: The Data Loader for the validation data set
            device: The device to run the model on
            epochs: The number of epochs to train for
            batch_size: The number of samples in a batch
            log_freq: How often to log the loss (default: 1)
            save_freq: How often to save the model (default: 10)
            weight_decay: The weight decay for the AdamW optimizer (default: 1e-4)
            checkpoint_dir: The directory to save the model checkpoints (default: "ckpts")
            freeze_backbone: Whether to freeze the backbone during training (default: False)
            backbone_lr: The learning rate for the backbone (default: 1e-5)
            transformer_lr: The learning rate for the transformer (default: 1e-4)
            num_queries: The number of object queries (default: 100)
            empty_class_id: The class id for the empty class (default: 0)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.num_train_batches = len(self.train_loader)
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.batch_size = batch_size
        self.log_freq = log_freq
        self.save_freq = save_freq
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        self.num_queries = num_queries
        self.empty_class_id = empty_class_id

        # History objects to hold training time metrics
        self.hist = []
        self.hist_detailed_losses = []

        # Create the optimizer with different learning rates for backbone/Transformer head and
        # optionally freeze the backbone during training.
        backbone_params = [p for n, p in model.named_parameters() if "backbone." in n]

        if freeze_backbone:
            print("Freezing CNN backbone...")
            for p in model.backbone.parameters():
                p.requires_grad = False
        else:
            # This is needed to re-enable the training of the backbone in case a previous
            # training iteration kept it frozen...
            for p in model.backbone.parameters():
                p.requires_grad = True
        print(f"CNN backbone is trainable: {not freeze_backbone}")

        transformer_params = [
            p for n, p in model.named_parameters() if "backbone." not in n
        ]

        self.optimizer = AdamW(
            [
                {"params": transformer_params, "lr": transformer_lr},
                {"params": backbone_params, "lr": backbone_lr},
            ],
            weight_decay=weight_decay,
        )

        # Log the number of total trainable parameters
        nparams = (
            sum([p.nelement() for p in model.parameters() if p.requires_grad]) / 1e6
        )
        print(f"DETR trainable parameters: {nparams:.1f}M")

        # Create the checkpoint dir if it does not exist...
        os.makedirs(self.checkpoint_dir, exist_ok=True)

    def compute_loss(self, o_bbox, t_bbox, o_cl, t_cl, t_mask):
        """
        Computes the total loss for a single sample (image and corresponding GT labels).

        Args:
            o_bbox (torch.Tensor): The predicted bounding boxes (Shape: torch.Size([100, 4]))
            t_bbox (torch.Tensor): The ground truth bounding boxes (Shape: torch.Size([100, 4]))
            o_cl (torch.Tensor): The predicted class labels (Shape: torch.Size([100, num_classes]))
            t_cl (torch.Tensor): The ground truth class labels (Shape: torch.Size([100]))
            t_mask (torch.Tensor): The mask for the ground truth bounding boxes (Shape: torch.Size([100]))

        Returns:
            torch.Tensor: The total loss for the sample
        """
        return compute_sample_loss(
            o_bbox,
            t_bbox,
            o_cl,
            t_cl,
            t_mask,
            n_queries=self.num_queries,
            empty_class_id=self.empty_class_id,
            device=self.device,
        )

    def log_epoch_losses(self, epoch, losses, class_losses, box_losses, giou_losses):
        """Logs and stores loss values for an epoch based on the set log frequency.

        Args:
            epoch(int) : Current epoch idx
            losses(torch.Tensor): The tensor holding the total DETR losses objects (per-batch)
            class_losses(torch.Tensor): The tensor holding the class losses objects (per-batch)
            box_losses(torch.Tensor): The tensor holding the bounding box L1 losses objects (per-batch)
            giou_losses(torch.Tensor): The tensor holding the GIoU objects (per-batch)
        """
        if (epoch + 1) % self.log_freq == 0:
            loss_avg = losses[-self.num_train_batches :].mean().item()
            epoch_loss_class = class_losses[-self.num_train_batches :].mean().item()
            epoch_loss_bbox = box_losses[-self.num_train_batches :].mean().item()
            epoch_loss_giou = giou_losses[-self.num_train_batches :].mean().item()

            print(f"Epoch: {epoch+1}/{self.epochs}, DETR Loss: {loss_avg:.4f}")
            print(
                f"→ Class Loss: {epoch_loss_class:.4f}, BBox Loss: {epoch_loss_bbox:.4f}, GIoU Loss: {epoch_loss_giou:.4f}"
            )

            self.hist.append(loss_avg)
            self.hist_detailed_losses.append(
                (epoch_loss_class, epoch_loss_bbox, epoch_loss_giou)
            )

    def save_checkpoint(self, epoch):
        """Saves model checkpoints and training history at specified intervals."""
        if (epoch + 1) % self.save_freq == 0:
            os.makedirs(self.checkpoint_dir, exist_ok=True)
            torch.save(
                self.model.state_dict(),
                f"{self.checkpoint_dir}/model_epoch{epoch+1}.pt",
            )

    def load_loss_history(self, hist_file=None, detail_hist_file=None):
        """
        Loads training loss and detailed loss history from .npy files and updates the corresponding attributes.

        Args:
            hist_file (str, optional): Path to the .npy file containing the total loss history.
            detail_hist_file (str, optional): Path to the .npy file containing detailed loss history
                                            (class loss, bbox loss, GIoU loss).
        """
        if hist_file:
            try:
                self.hist = np.load(hist_file).tolist()
                print(f"Loaded loss history from {hist_file}.")
            except Exception as e:
                print(f"Error loading loss history file: {e}")

        if detail_hist_file:
            try:
                self.hist_detailed_losses = np.load(detail_hist_file).tolist()
                print(f"Loaded detailed loss history from {detail_hist_file}.")
            except Exception as e:
                print(f"Error loading detailed loss history file: {e}")

    def visualize_losses(self, save_dir=None):
        """
        Plots training loss over epochs and optionally saves the figure.

        Args:
            save_dir (str, optional): Directory to save the plots. If None, it only displays the plots.
        """

        # Create save directory if it doesn't exist
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        epochs = np.arange(1, len(self.hist) + 1) * self.log_freq

        plt.figure(figsize=(10, 5))
        plt.plot(epochs, self.hist, label="Total Loss", marker="o", linestyle="-")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Over Epochs")
        plt.legend()
        plt.grid()

        if save_dir:
            plt.savefig(os.path.join(save_dir, "DETR_training_loss.png"))
        #plt.show()

        # If detailed loss is provided, plot them separately
        if self.hist_detailed_losses:
            class_loss, bbox_loss, giou_loss = zip(*self.hist_detailed_losses)

            plt.figure(figsize=(10, 5))
            plt.plot(epochs, class_loss, label="Class Loss", linestyle="--")
            plt.plot(epochs, bbox_loss, label="BBox Loss", linestyle="--")
            plt.plot(epochs, giou_loss, label="GIoU Loss", linestyle="--")
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.title("Detailed Training Loss Over Epochs")
            plt.legend()
            plt.grid()

            if save_dir:
                plt.savefig(os.path.join(save_dir, "DETR_training_losses.png"))
            #plt.show()

    def train(self):
        """Trains the DETR model for a specified number of epochs, with checkpoint/log callbacks."""
        torch.set_grad_enabled(True)
        self.model.train()
        print(
            f"Starting training for {self.epochs} epochs... Using device : {self.device}"
        )

        losses = torch.tensor([], device=self.device)
        class_losses = torch.tensor([], device=self.device)
        box_losses = torch.tensor([], device=self.device)
        giou_losses = torch.tensor([], device=self.device)

        # Clear the training history from previous trainings..
        self.hist = []
        self.hist_detailed_losses = []

        for epoch in range(self.epochs):
            for batch_idx, (input_, (tgt_cl, tgt_bbox, tgt_mask, _)) in enumerate(
                self.train_loader
            ):
                # Move data to device
                input_ = input_.to(self.device)
                tgt_cl = tgt_cl.to(self.device)
                tgt_bbox = tgt_bbox.to(self.device)
                tgt_mask = tgt_mask.bool().to(self.device)

                # Run inference
                class_preds, bbox_preds = self.model(input_)

                # Accumulate losses
                loss = torch.tensor(0.0, device=self.device)
                loss_class_batch = torch.tensor(0.0, device=self.device)
                loss_bbox_batch = torch.tensor(0.0, device=self.device)
                loss_giou_batch = torch.tensor(0.0, device=self.device)

                num_dec_layers = class_preds.shape[1]

                for i in range(num_dec_layers):
                    o_bbox = bbox_preds[:, i, :, :].sigmoid().to(self.device)
                    o_cl = class_preds[:, i, :, :].to(self.device)

                    for o_bbox_i, t_bbox, o_cl_i, t_cl, t_mask in zip(
                        o_bbox, tgt_bbox, o_cl, tgt_cl, tgt_mask
                    ):

                        loss_class, loss_bbox, loss_giou = self.compute_loss(
                            o_bbox_i, t_bbox, o_cl_i, t_cl, t_mask
                        )

                        sample_loss = 1 * loss_class + 5 * loss_bbox + 2 * loss_giou

                        loss += sample_loss / self.batch_size / num_dec_layers

                        # Track individual losses per batch
                        loss_class_batch += (
                            loss_class / self.batch_size / num_dec_layers
                        )
                        loss_bbox_batch += loss_bbox / self.batch_size / num_dec_layers
                        loss_giou_batch += loss_giou / self.batch_size / num_dec_layers

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()

                # Clip gradient norms
                nn.utils.clip_grad_norm_(self.model.parameters(), 0.1)
                self.optimizer.step()

                # Gather batch-level losses
                losses = torch.cat((losses, loss.unsqueeze(0)))
                class_losses = torch.cat((class_losses, loss_class_batch.unsqueeze(0)))
                box_losses = torch.cat((box_losses, loss_bbox_batch.unsqueeze(0)))
                giou_losses = torch.cat((giou_losses, loss_giou_batch.unsqueeze(0)))

            # If the epoch is done check if it's time to log the training metrics...
            # Then check if it's time to save a checkpoint..
            self.log_epoch_losses(
                epoch=epoch,
                losses=losses,
                class_losses=class_losses,
                box_losses=box_losses,
                giou_losses=giou_losses,
            )
            self.save_checkpoint(epoch=epoch)


if __name__ == "__main__":
    import torch
    from torch import nn
    from torch.utils.data import DataLoader
    from torchvision import ops
    from DETR_coco_pytorch import TorchCOCOLoader, collate_fn
    # Batch size for dataloaders and image size for model/pre-processing
    BATCH_SIZE = 100
    IMAGE_SIZE = 480
    MAX_OBJECTS = 100
    FREEZE_BACKBONE = True
    EPOCHS = 150
    LOG_FREQUENCY = 5 # Training-time losses will be logged according to this frequency
    SAVE_FREQUENCY = 20 # Model weights will be saved according to this frequency
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Training device
    # NOTE: You can load the COCO dataset infromation or any other from the available datasets 
    #       if it's in the DATASET_CLASSES class map. This map is a lookup dictionary with the
    #       dataset name as key where each instance has the following attributes:
    #           - "class_names" : The list of class names
    #           - "empty_class_id": The ID of the class to be treated as the "empty" class for boxes
    #           - "links": Contains some sort of link to download the dataset
    # NOTE: All the available datasets are listed in the project README file.
    FASHIONPEDIA_CLASSES = ['background','shirt, blouse','top, t-shirt, sweatshirt','sweater','cardigan','jacket','vest','pants','shorts','skirt','coat','dress','jumpsuit','cape','glasses','hat','headband, head covering, hair accessory','tie','glove','watch','belt','leg warmer','tights, stockings','sock','shoe','bag, wallet','scarf','umbrella']
    DATASET_CLASSES ={
        "fashionpedia": {
            "class_names": FASHIONPEDIA_CLASSES,
            "empty_class_id": 0,
        }
    }

    CLASSES = DATASET_CLASSES["fashionpedia"]["class_names"]
    EMPTY_CLASS_ID = DATASET_CLASSES["fashionpedia"]["empty_class_id"]


    # Or explicitly set the  class labels/empty class ID for your custom dataset if its not added to the DATASET_CLASSES map...
    # CLASSES = ["N/A", "something"]
    # EMPTY_CLASS_ID = 0 # ID of the dataset classes to treat as "empty" class


    # Load and COCO dataset (adjust the paths accordingly)
    coco_ds_train = TorchCOCOLoader(
        config.TRAIN_IMG,
        config.ANNOTATIONS_TRAIN,
        max_boxes=MAX_OBJECTS,
        empty_class_id=EMPTY_CLASS_ID,
        image_size=IMAGE_SIZE,
        augment=True
    )

    coco_ds_val = TorchCOCOLoader(
        config.TEST_IMG,
        config.ANNOTATIONS_TEST,
        max_boxes=MAX_OBJECTS,
        empty_class_id=EMPTY_CLASS_ID,
        image_size=IMAGE_SIZE,
    )

    train_loader = DataLoader(
        coco_ds_train, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    val_loader = DataLoader(
        coco_ds_val, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn
    )

    print(f"Training dataset size: {len(coco_ds_train)}")
    print(f"Validation dataset size: {len(coco_ds_val)}")

    import matplotlib.pyplot as plt
    from DETR_visualizer import DETRBoxVisualizer

    # Create a visualizer
    visualizer = DETRBoxVisualizer(class_labels= CLASSES,
                                empty_class_id=0)

    # Visualize batches
    dataloader_iter = iter(train_loader)
    for i in range(1):
        input_, (classes, boxes, masks, _) = next(dataloader_iter)
        fig = plt.figure(figsize=(10, 10), constrained_layout=True)

        for ix in range(4):
            t_cl = classes[ix]
            t_bbox = boxes[ix]
            mask = masks[ix].bool()

            # Filter padded classes/boxes using the binary mask...
            t_cl = t_cl[mask]
            t_bbox = t_bbox[mask] * IMAGE_SIZE

            # Convert to x1y1x2y2 for visualization and denormalize boxes..
            t_bbox = ops.box_convert(
                t_bbox, in_fmt='cxcywh', out_fmt='xyxy')
            
            im = input_[ix]

            ax = fig.add_subplot(2, 2, ix+1)
            visualizer._visualize_image(im, t_bbox, t_cl, ax=ax)
    detr_model = DETR(
        d_model=256, n_classes=92, n_tokens=225, 
        n_layers=6, n_heads=8, n_queries=MAX_OBJECTS, use_frozen_bn=True
    )
    """
    CHECKPOINT_PATH = "<YOUR_DETR_WEIGHTS.pt>"

    # Load the checkpoint
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=torch.device("cpu"))

    # Load the weights into the model
    # We don't use strict matching as you might want to use FrozenBatchNorm2D...
    # Some pre-trained weights come from trainings using BatchNorm2D
    print(detr_model.load_state_dict(checkpoint['state'], strict=False))

    # Adapt the class prediction head to our new dataset
    detr_model.linear_class = nn.Linear(detr_model.linear_class.in_features, len(CLASSES))
    """


    # Create a trainer for DETR
    trainer = DETRTrainer(model = detr_model,
                        train_loader= train_loader,
                        val_loader=val_loader,
                        device=device,
                        epochs=EPOCHS,
                        batch_size=BATCH_SIZE,
                        log_freq=LOG_FREQUENCY,
                        save_freq=SAVE_FREQUENCY,
                        freeze_backbone= FREEZE_BACKBONE,
                        num_queries=MAX_OBJECTS,
                        empty_class_id=EMPTY_CLASS_ID)

    # Start the training
    trainer.train()
    trainer.visualize_losses(save_dir = "./")