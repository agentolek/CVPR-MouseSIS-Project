import ultralytics
from transformers import SamModel, SamProcessor
import torch

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

import hydra

from ..utils import suppress_stdout_stderr

# change hydra path 
hydra.core.global_hydra.GlobalHydra.instance().clear()
# reinit hydra with a new search path for configs
hydra.initialize_config_module('/net/tscratch/people/plgagentolek/CVPR-MouseSIS-Project', version_base='1.2')

class SamYoloDetector:
    def __init__(self, yolo_path, device='cuda:0') -> None:
        self.detector = ultralytics.YOLO(yolo_path)
        # self.sam_model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
        # self.sam_processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")
        self.device = device
        self.sam_predictor = self.buildSam2Predictor()

    def buildSam2Predictor(self):
        sam2_checkpoint = "/net/tscratch/people/plgagentolek/sam2/checkpoints/sam2.1_hiera_large.pt"
        model_cfg = "./configs/sam2.1/sam2.1_hiera_l.yaml"

        sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=self.device)

        predictor = SAM2ImagePredictor(sam2_model)
        return predictor

    def run(self, img):
        with suppress_stdout_stderr():
            result = self.detector(img)[0]
        
        boxes = result.boxes.xyxy.detach().cpu().numpy()  # x1, y1, x2, y2
        scores = result.boxes.conf.detach().cpu().numpy()
        
        if not len(boxes):
            return None, None
            
        boxes_list = [[boxes.tolist()]]
        # inputs = self.sam_processor(img.transpose(2, 0, 1), input_boxes=[boxes_list], return_tensors="pt").to(self.device)
        
        with torch.inference_mode(), torch.autocast("cuda:0", dtype=bfloat16):
            # outputs = self.sam_model(**inputs)
            # masks = self.sam_processor.image_processor.post_process_masks(
            #     outputs.pred_masks.cpu(),
            #     inputs["original_sizes"].cpu(),
            #     inputs["reshaped_input_sizes"].cpu()
            # )[0]
            self.sam_predictor.set_image(img.transpose(2, 0, 1))
            masks, iou_scores, _ = self.sam_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=boxes,
                multimask_output=False
            )
            
            iou_scores.cpu()
            # iou_scores = outputs.iou_scores.cpu()[0]
            num_instances, nb_predictions, height, width = masks.shape
            max_indices = iou_scores.argmax(dim=1, keepdim=True)
            gather_indices = max_indices[..., None, None].expand(-1, 1, height, width)
            selected_masks = torch.gather(masks, 1, gather_indices).squeeze(1)
            
        return selected_masks.cpu().numpy(), scores