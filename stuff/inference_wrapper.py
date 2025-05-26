import ultralytics
assert "__multilabel__" in dir(ultralytics) and ultralytics.__multilabel__==True, "Please use a verion of ultralytics that supports multilabel"
import concurrent.futures
import torch
import copy
import cv2
import stuff.image as image_stuff
import stuff.ultralytics as ultralytics_stuff
import numpy as np
import os
import requests

from ensemble_boxes import weighted_boxes_fusion
try:
    from rfdetr import RFDETRBase, RFDETRLarge
    from rfdetr.util.coco_classes import COCO_CLASSES
    rfdetr_ok=True
except ImportError:
    rfdetr_ok=False
try:
    from detectron2.engine import DefaultPredictor
    from detectron2.config import get_cfg
    from detectron2 import model_zoo
    from detectron2.data import MetadataCatalog
    from detectron2.data import MetadataCatalog
    detectron2_ok=True
except ImportError:
    detectron2_ok=False
try:
    from mmdet.apis import init_detector, inference_detector
    from mmengine.structures import InstanceData
    mmdet_ok = True
except ImportError:
    mmdet_ok = False

def download_mmdet_config(config_path, repo="https://raw.githubusercontent.com/open-mmlab/mmdetection/main/configs"):
    """
    Download MMDetection config if it doesn't exist locally.
    Args:
        config_path (str): Relative path like 'rtmdet/rtmdet-h_8xb32-300e_coco.py'
        repo (str): Base URL to fetch from
    Returns:
        local_path (str): Path to saved config
    """
    config_dir = os.path.join("mmdetection", "configs")
    local_path = os.path.join(config_dir, config_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)

    if not os.path.isfile(local_path):
        print(f"Downloading MMDetection config: {config_path}")
        url = f"{repo}/{config_path}"
        response = requests.get(url)
        if response.status_code != 200:
            raise FileNotFoundError(f"Failed to download config: {url}")
        with open(local_path, "w") as f:
            f.write(response.text)
    return local_path

def merge_detections_wbf(detectors_outputs, iou_thr=0.55, skip_box_thr=0.001):
    all_boxes = []
    all_scores = []
    all_labels = []

    for detections in detectors_outputs:
        boxes = []
        scores = []
        labels = []
        for det in detections:
            boxes.append(det["box"])
            scores.append(det["confidence"])
            labels.append(det["class"])
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    merged_boxes, merged_scores, merged_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr
    )

    # Build output detections
    merged_detections = []
    for box, score, label in zip(merged_boxes, merged_scores, merged_labels):
        merged_detections.append({
            "box": box,
            "confidence": score,
            "class": int(label)
        })

    return merged_detections

def ensemble_merge_results(results):
    num=len(results[0])
    out=[]
    for i in range(num):
        dets=[]
        for r in results:
            dets.append(r[i])
        out.append(merge_detections_wbf(dets))
    return out

def github_blob_to_raw_url(blob_url):
    return blob_url.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")

class inference_wrapper:
    class_synonyms={'person':['man','woman','boy','girl'],
                    'vehicle':['car','bicycle','motorcycle','train','truck','bus','airplane','boat'],
                    'animal':['cat','dog','horse','sheep','cow','bird','elephant','bear','zebra','giraffe'],
                    'weapon':['gun', 'dagger', 'pistol', 'handgun', 'rifle', 'revolver'],
                    'face':[]}

    def __init__(self,
                 model_name,
                 class_names=None,
                 thr=0.5,
                 nms_iou=0.45,
                 half=True,
                 rect=True,
                 max_det=400,
                 batch_size=32,
                 imgsz=640,
                 attributes=False,
                 yolo_add_faces=False,
                 face_kp=False,
                 pose_kp=False,
                 facepose_kp=False,
                 fold_attributes=False):

        self.ensemble_models=None
        self.class_names=class_names
        self.yolo_batch_size=batch_size
        self.yolo_num_params=0
        self.yolo_num_flops=0
        self.yolo_det_conf=thr
        self.yolo_cache={}
        self.num_threads=1
        self.detectron2_predictor=None
        self.rf_detr_model=None

        # set model size
        if ":" in model_name:
            x=model_name.split(":")
            model_name=x[0]
            params=[x[1]]
            if "," in x[1]:
                params=x[1].split(",")
            for p in params:
                if p=="flip-lr":
                    aug_params["flip-lr"]=True
                else:
                    imgsz=int(p)
        self.imgsz=imgsz

        if model_name=="ensemble2":
            model_name=["/mldata/weights/good/yolo11l.pt:960",
                        "/mldata/weights/good/yolo12l.pt:640",
                        "/mldata/weights/good/yolov9e.pt:864",
                        "/mldata/weights/good/yolov5x6u.pt:1280"]
        if model_name=="ensemble3":
            model_name=["/mldata/weights/good/yolov9e.pt:512",
                        "/mldata/weights/good/yolov9e.pt:640,flip-lr",
                        "/mldata/weights/good/yolov9e.pt:864",
                        "/mldata/weights/good/yolov9e.pt:960,flip-lr"]

        # Support roboflow DETR models (https://github.com/roboflow/rf-detr)
        if model_name=="RFDETR-B" or model_name=="RFDETR-L":
            assert rfdetr_ok, "try pip install rfdetr"
            if model_name=="RFDETR-B":
                self.rf_detr_model=RFDETRBase(resolution=self.imgsz)
            else:
                self.rf_detr_model=RFDETRLarge(resolution=self.imgsz)
            num_classes=max([i for i in COCO_CLASSES])+1

            self.yolo_class_names=["-"]*num_classes
            for i in COCO_CLASSES:
                self.yolo_class_names[i]=COCO_CLASSES[i]
            self.set_class_remap()
            return

        # Support models using detectron2
        detectron2_models=["faster_rcnn_X_101_32x8d_FPN_3x",
                           "faster_rcnn_R_101_FPN_3x"]
        if model_name in detectron2_models:
            assert detectron2_ok, "try python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            self.detectron2_cfg = get_cfg()
            model="COCO-Detection/"+model_name+".yaml"
            self.detectron2_cfg.merge_from_file(model_zoo.get_config_file(model))
            self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.yolo_det_conf  # confidence threshold
            self.detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
            self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_iou
            self.detectron2_predictor = DefaultPredictor(self.detectron2_cfg)
            metadata = MetadataCatalog.get(self.detectron2_cfg.DATASETS.TRAIN[0])
            class_names = metadata.get("thing_classes", None)
            self.yolo_class_names=class_names
            self.set_class_remap()
            return

         # Support MMDetection models
        if model_name in mmdet_models:
            assert mmdet_ok, "try pip install 'mmdet' and 'mmengine'"
            mmdet_path="/home/mark/stuff/ai/mmdetection/configs"
            mmdet_models = {
                "cascade-mask-rcnn_x101-64x4d_fpn_ms-3x": (mmdet_path+"/cascade_rcnn/cascade-mask-rcnn_x101-64x4d_fpn_ms-3x_coco.py",
                        "https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco/cascade_mask_rcnn_x101_64x4d_fpn_mstrain_3x_coco_20210719_210311-d3e64ba0.pth"),
                "grounding-dino-b": (mmdet_path+"/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco.py",
                                    "https://download.openmmlab.com/mmdetection/v3.0/grounding_dino/grounding_dino_swin-b_finetune_16xb2_1x_coco/grounding_dino_swin-b_finetune_16xb2_1x_coco_20230921_153201-f219e0c0.pth"),
                "dino-5scale": (mmdet_path+"/dino/dino-5scale_swin-l_8xb2-36e_coco.py",
                                "https://github.com/RistoranteRist/mmlab-weights/releases/download/dino-swinl/dino-5scale_swin-l_8xb2-36e_coco-5486e051.pth"),
                "crowddet2": (mmdet_path+"/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman.py",
                            "https://download.openmmlab.com/mmdetection/v3.0/crowddet/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman/crowddet-rcnn_r50_fpn_8xb2-30e_crowdhuman_20221023_174954-dc319c2d.pth"),
            }

            cfg_path, weight_url = mmdet_models[model_name]
            self.mmdet_model = init_detector(cfg_path, weight_url, device="cuda:0")
            self.yolo_class_names = self.mmdet_model.dataset_meta['classes']
            self.set_class_remap()
            return

        if isinstance(model_name, list):
            self.ensemble_models=[]
            for m in model_name:
                yolo=inference_wrapper(m,
                                       class_names=class_names,
                                       thr=thr,
                                       nms_iou=nms_iou,
                                       max_det=max_det,
                                       rect=rect,
                                       imgsz=imgsz,
                                       batch_size=batch_size//2)
                self.ensemble_models.append(yolo)
            self.num_threads=self.ensemble_models[0].num_threads
            self.yolo_batch_size=self.ensemble_models[0].yolo_batch_size
            self.yolo_num_params=sum([x.yolo_num_params for x in self.ensemble_models])
            self.yolo_num_flops=sum([x.yolo_num_flops for x in self.ensemble_models])
            self.yolo_cache={}
            return

        aug_params={}
        self.ensemble_models=None
        self.model=self.load_ultralytics_model(model_name)
        self.num_gpus=torch.cuda.device_count()
        self.num_threads=self.num_gpus
        self.yolo=[None]*self.num_threads
        self.model_params=[aug_params]*self.num_threads
        self.yolo_det_conf=thr
        self.yolo_nms_iou=nms_iou
        self.yolo_half=half
        self.yolo_rect=rect
        self.yolo_max_det=max_det
        self.yolo_batch_size=batch_size
        self.attributes=attributes
        self.yolo_add_faces=yolo_add_faces
        self.face_kp=face_kp
        self.pose_kp=pose_kp
        self.facepose_kp=facepose_kp
        self.fold_attributes=fold_attributes
        self.yolo_cache={}

        for i in range(self.num_threads):
            self.yolo[i]=self.load_ultralytics_model(model_name)
            self.yolo[i]=self.yolo[i].to("cuda:"+str(i))
            assert(self.yolo[i] is not None)

        self.yolo_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]

        self.set_class_remap()

    def set_class_remap(self):
        self.det_class_remap=[-1]*len(self.yolo_class_names)
        if self.class_names is None:
            self.class_names=self.yolo_class_names
        self.can_detect=[False]*len(self.class_names)

        # map the detected classes back to our class set
        # assume if our class set has things like 'vehicle' then we would want any standard
        # coco classes like 'car' to map to that

        for i,x in enumerate(self.yolo_class_names):
            if x in self.class_names:
                self.det_class_remap[i]=self.class_names.index(x)
                self.can_detect[self.class_names.index(x)]=True
            else:
                for y in self.class_synonyms:
                    if y in self.class_names:
                        if x in self.class_synonyms[y]:
                            self.det_class_remap[i]=self.class_names.index(y)
                            self.can_detect[self.class_names.index(y)]=True

    def load_ultralytics_model(self, name):
        task="detect"
        if "world" in name:
            extended_classes=copy.deepcopy(self.class_names)
            for c in self.class_names:
                if c in self.class_synonyms:
                    for s in self.class_synonyms[c]:
                        if not s in extended_classes:
                            extended_classes.append(s)

            model=ultralytics.YOLOWorld(name, verbose=False)
            model.set_classes(extended_classes)
            return model
        if "yoloe" in name:
            model=ultralytics.YOLOE(name, task=task, verbose=False)
            model.set_classes(self.class_names, model.get_text_pe(self.class_names))
        if "nas" in name:
            return ultralytics.NAS(name)
        if "rtdetr" in name:
            return ultralytics.RTDETR(name)
        if "pose" in name or "face" in name or "full" in name or "attributes" in name or "dpa" in name:
            task="pose"
        if "dpar" in name:
            task="posereid"

        model=ultralytics.YOLO(name, task=task, verbose=False)
        info=model.info(verbose=False)

        self.yolo_num_params=0
        self.yolo_num_flops=0
        if info is not None:
            self.yolo_num_params=info[1]
            self.yolo_num_flops=info[3]

        #try:
        #    self.yolo_num_params = sum(p.numel() for p in self.yolo[0].model.parameters())
        #except AttributeError:
        #    self.yolo_num_params = 0
        return model

    def infer_rfdetr(self, input_frames, thr=None):
        out_det=[]
        #detections_list = self.rf_detr_model.predict(input_frames, threshold=self.yolo_det_conf)
        detections_list=[]
        for i in input_frames:
            detections_list.append(self.rf_detr_model.predict(i, threshold=self.yolo_det_conf))
        ret=[]
        for j,detections in enumerate(detections_list):
            num_detections=len(detections.xyxy)
            out_det=[]
            w,h=image_stuff.get_image_size(input_frames[j])
            for i in range(num_detections):
                xyxy=detections.xyxy[i]
                box=[xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h]
                #print(int(detections.class_id[i]), COCO_CLASSES[detections.class_id[i]])
                det={"box":box,
                    "id":None,
                    "class":self.det_class_remap[int(detections.class_id[i]+0.01)],
                    "confidence":detections.confidence[i]}
                out_det.append(det)
            ret.append(out_det)
        return ret

    def infer_mmdet(self, input_frames, thr=None):
        ret = []
        for img in input_frames:
            if isinstance(img, str):
                image = cv2.imread(img)
            else:
                image = img

            w,h=image_stuff.get_image_size(image)
            det_sample = inference_detector(self.mmdet_model, image)
            # Extract instances (boxes, labels, scores)
            instances = det_sample.pred_instances  # InstanceData object
            boxes = instances.bboxes.cpu().numpy()        # shape: [N, 4]
            scores = instances.scores.cpu().numpy()       # shape: [N]
            labels = instances.labels.cpu().numpy()       # shape: [N]

            out_det = []
            for box, score, label in zip(boxes, scores, labels):
                if score < self.yolo_det_conf:
                    continue
                box_norm = [box[0]/w, box[1]/h, box[2]/w, box[3]/h]
                out_det.append({
                    "box": box_norm,
                    "id": None,
                    "class": self.det_class_remap[int(label)],
                    "confidence": score
                })
            ret.append(out_det)
        return ret

    def infer_detectron2(self, input_frames, thr=None):
        ret=[]
        for i in input_frames:
            if not isinstance(i, np.ndarray):
                image = cv2.imread(i)
            else:
                image=i

            # Run detection
            outputs = self.detectron2_predictor(image)
            # Extract only person boxes
            instances = outputs["instances"]
            person_scores = instances.scores.cpu().numpy()
            boxes = instances.pred_boxes.tensor.cpu().numpy()
            pred_classes = instances.pred_classes.cpu().numpy()
            w,h=image_stuff.get_image_size(i)
            out_det=[]
            for i in range(len(boxes)):
                xyxy=boxes[i]
                box=[xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h]
                det={"box":box,
                    "id":None,
                    "class":self.det_class_remap[pred_classes[i]],
                    "confidence":person_scores[i]}
                out_det.append(det)
            ret.append(out_det)
            #print(boxes,person_scores)
            #print(f"thr {self.yolo_det_conf}")
        return ret

    def infer(self, input_frames, thr=None):
        if self.ensemble_models is not None:
            results=[]
            for m in self.ensemble_models:
                results.append(m.infer(input_frames, thr=thr))
            return ensemble_merge_results(results)

        if self.rf_detr_model!=None:
            return self.infer_rfdetr(input_frames, thr)

        if self.detectron2_predictor is not None:
            return self.infer_detectron2(input_frames, thr)

        if hasattr(self, "mmdet_model"):
            return self.infer_mmdet(input_frames, thr)

        def process_batch(yolo_fn, frames, params):
            if params is not None:
                if params.get("flip-lr", False):
                    out_frames=[]
                    for f in frames:
                        if isinstance(f, str):
                            f = cv2.imread(f)
                        f=cv2.flip(f, 1)
                        out_frames.append(f)
                    frames=out_frames
            return yolo_fn(
                frames,
                conf=self.yolo_det_conf,
                iou=self.yolo_nms_iou,
                max_det=self.yolo_max_det,
                agnostic_nms=False,
                half=self.yolo_half,
                imgsz=self.imgsz,
                verbose=False,
                rect=self.yolo_rect
            )

        results = []
        index = 0
        if thr is not None:
            self.yolo_det_conf=thr

        if self.num_threads == 1:
            while index < len(input_frames):
                end = min(len(input_frames), index + self.yolo_batch_size)
                batch = input_frames[index:end]
                results += process_batch(self.yolo[0], batch, self.model_params[0])
                index = end
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while index < len(input_frames):
                    futures = []
                    for i in range(self.num_threads):
                        end = min(len(input_frames), index + self.yolo_batch_size)
                        if index < end:
                            batch = input_frames[index:end]
                            futures.append(executor.submit(process_batch, self.yolo[i], batch, self.model_params[i]))
                        index = end
                    for future in futures:
                        results += future.result()

        ret=[]
        for r in results:
            out_det=ultralytics_stuff.yolo_results_to_dets(r,
                                             det_thr=self.yolo_det_conf,
                                             det_class_remap=self.det_class_remap,
                                             yolo_class_names=self.yolo_class_names,
                                             class_names=self.class_names,
                                             attributes=self.attributes,
                                             add_faces=False,
                                             face_kp=self.face_kp,
                                             pose_kp=self.pose_kp,
                                             facepose_kp=self.facepose_kp,
                                             fold_attributes=self.fold_attributes)

            if self.model_params[0] is not None:
                if self.model_params[0].get("flip-lr", False):
                    for d in out_det:
                        d["box"]=[1.0-d["box"][0],d["box"][1],1.0-d["box"][2],d["box"][3]]
            ret.append(out_det)
        return ret

    def infer_cached(self, index, num_images, get_image_fn, thr=None):
        if index in self.yolo_cache:
            return self.yolo_cache[index]
        self.yolo_cache={}
        num=self.yolo_batch_size*self.num_threads
        num=min(num, num_images-index)
        image_list=[get_image_fn(i+index) for i in range(num)]
        batch_results=self.infer(image_list, thr=thr)
        self.yolo_cache={(i+index):batch_results[i] for i in range(num)}
        assert index in self.yolo_cache
        return self.yolo_cache[index]