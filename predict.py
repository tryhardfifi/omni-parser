from cog import BasePredictor, Input, Path, BaseModel
import io
import os
import torch
import base64
from PIL import Image
from huggingface_hub import snapshot_download

MODEL_CACHE = "weights"
# Set all possible PaddleOCR related environment variables
os.environ["HF_HOME"] = MODEL_CACHE
os.environ["HF_HUB_CACHE"] = MODEL_CACHE

from util.utils import get_yolo_model, get_caption_model_processor, check_ocr_box, get_som_labeled_img

class Output(BaseModel):
    img: Path
    elements: str

class Predictor(BasePredictor):
    def setup(self) -> None:
        """Load the model into memory to make running multiple predictions efficient"""
        # make folder checkpoints
        os.makedirs(MODEL_CACHE, exist_ok=True)
    
        # download the weights if not exists
        if not os.path.exists(MODEL_CACHE+"/OmniParser-v2.0"):
            print("Downloading OmniParser-v2.0")
            snapshot_download(
                repo_id="microsoft/OmniParser-v2.0",
                local_dir=MODEL_CACHE+"/OmniParser-v2.0",
                local_dir_use_symlinks=False
            )

        if not os.path.exists(MODEL_CACHE+"/Florence-2-base"):
            print("Downloading Florence-2-base")
            snapshot_download(
                repo_id="microsoft/Florence-2-base",
                local_dir=MODEL_CACHE+"/Florence-2-base",
                local_dir_use_symlinks=False
            )

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Fix paths to use MODEL_CACHE instead of checkpoints
        self.yolo_model = get_yolo_model(model_path=os.path.join(MODEL_CACHE, 'OmniParser-v2.0/icon_detect/model.pt'))
        self.caption_model_processor = get_caption_model_processor(
            model_name="florence2", 
            model_name_or_path=os.path.join(MODEL_CACHE, 'OmniParser-v2.0/icon_caption')
        )
        print("Finished setup")


    @torch.inference_mode()
    def predict(
        self,
        image: Path = Input(description="Input image to process"),
        box_threshold: float = Input(
            description="Threshold for removing bounding boxes with low confidence",
            default=0.05,
            ge=0.01,
            le=1.0,
        ),
        iou_threshold: float = Input(
            description="Threshold for removing bounding boxes with large overlap",
            default=0.1,
            ge=0.01,
            le=1.0,
        ),
        imgsz: int = Input(
            description="Icon detection image size",
            default=640,
            ge=640,
            le=1920,
        ),
    ) -> Output:
        """Run a single prediction on the model"""
        # Load and process the image
        image_input = Image.open(image)
        
        # Calculate box overlay ratio
        box_overlay_ratio = image_input.size[0] / 3200
        draw_bbox_config = {
            'text_scale': 0.8 * box_overlay_ratio,
            'text_thickness': max(int(2 * box_overlay_ratio), 1),
            'text_padding': max(int(3 * box_overlay_ratio), 1),
            'thickness': max(int(3 * box_overlay_ratio), 1),
        }

        # Process the image
        ocr_bbox_rslt, is_goal_filtered = check_ocr_box(
            image_input,
            display_img=False,
            output_bb_format='xyxy',
            goal_filtering=None,
            easyocr_args={'paragraph': False, 'text_threshold': 0.9}
        )
        
        text, ocr_bbox = ocr_bbox_rslt
        dino_labled_img, label_coordinates, parsed_content_list = get_som_labeled_img(
            image_input,
            self.yolo_model,
            BOX_TRESHOLD=box_threshold,
            output_coord_in_ratio=True,
            ocr_bbox=ocr_bbox,
            draw_bbox_config=draw_bbox_config,
            caption_model_processor=self.caption_model_processor,
            ocr_text=text,
            iou_threshold=iou_threshold,
            imgsz=imgsz,
        )
        output_img = Image.open(io.BytesIO(base64.b64decode(dino_labled_img)))
        img_path = "/tmp/output.png"
        output_img.save(img_path)
        parsed_content_list = '\n'.join([f'icon {i}: ' + str(v) for i,v in enumerate(parsed_content_list)])
        
        # Return the image path and elements
        return Output(img=Path(img_path), elements=str(parsed_content_list))