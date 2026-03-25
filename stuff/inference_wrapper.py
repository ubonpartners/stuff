import ultralytics
import concurrent.futures
import copy
import cv2
import stuff
import stuff.image as image_stuff
import stuff.ultralytics as ultralytics_stuff
import numpy as np
import os
import requests
import time
from functools import partial

from PIL import Image

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
    from ensemble_boxes import weighted_boxes_fusion
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

def infer_model_name(x):
    ext=""
    if "," in x:
        x=x.split(",")[0]
    if isinstance(x, str):
        if "+" in x:
            t=x.split("+")
            x=t[0]
            ext+=t[1]
        if ":" in x:
            t=x.split(":")
            x=t[0]
            ext+=t[1]+" "
        if x.endswith(".engine"):
            ext+="TRT "
        if x.endswith(".trt"):
            ext+="UPYC "
        if len(ext)>0:
            ext="("+ext[:-1]+")"
    ret=os.path.splitext(os.path.basename(x))[0]+ext
    #print(f"infer model name {x} = {ret}")
    return ret

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
                 nms_thr=0.45,
                 half=True,
                 rect=True,
                 max_det=1200,
                 batch_size=32,
                 imgsz=640,
                 attributes=None,
                 face_kp=False,
                 pose_kp=False,
                 facepose_kp=False,
                 fold_attributes=False,
                 get_feats=False):

        self.ensemble_models=None
        self.class_names=class_names
        self.infer_batch_size=batch_size
        self.yolo_num_params=0
        self.yolo_num_flops=0
        self.yolo_det_conf=thr
        self.infer_cache={}
        self.num_threads=1
        self.detectron2_predictor=None
        self.rf_detr_model=None
        self.get_feats=get_feats

        # set model size
        if ":" in model_name:
            x = model_name.split(":")
            model_name = x[0]
            params = [x[1]]
            if "," in x[1]:
                params = x[1].split(",")
            aug_params = {}
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
        # https://rfdetr.roboflow.com/learn/pretrained/
        if model_name in ["rfdetr-nano","rfdetr-small","rfdetr-medium","rfdetr-large","rfdetr-base"]:
            try:
                from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRSmall, RFDETRNano
                from rfdetr.util.coco_classes import COCO_CLASSES
            except ImportError:
                assert False, "try pip install rfdetr"
            model_map={"rfdetr-nano":(RFDETRNano, 16),
                       "rfdetr-small":(RFDETRSmall, 16),
                       "rfdetr-medium":(RFDETRMedium, 16),
                       "rfdetr-large":(RFDETRLarge, 8),
                       }
            self.rf_detr_model=model_map[model_name][0](resolution=self.imgsz)
            self.rf_detr_batch_size=model_map[model_name][1]
            self.rf_detr_model.optimize_for_inference(batch_size=self.rf_detr_batch_size)
            num_classes=max([i for i in COCO_CLASSES])+1

            self.infer_model_class_names=["-"]*num_classes
            for i in COCO_CLASSES:
                self.infer_model_class_names[i]=COCO_CLASSES[i]
            self.set_class_remap()
            return

        # Support models using detectron2
        detectron2_models=["faster_rcnn_X_101_32x8d_FPN_3x",
                           "faster_rcnn_R_101_FPN_3x"]
        if model_name in detectron2_models:
            try:
                from detectron2.engine import DefaultPredictor
                from detectron2.config import get_cfg
                from detectron2 import model_zoo
                from detectron2.data import MetadataCatalog
                detectron2_ok=True
            except ImportError:
                detectron2_ok=False
            assert detectron2_ok, "try python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'"
            self.detectron2_cfg = get_cfg()
            model="COCO-Detection/"+model_name+".yaml"
            self.detectron2_cfg.merge_from_file(model_zoo.get_config_file(model))
            self.detectron2_cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.yolo_det_conf  # confidence threshold
            self.detectron2_cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(model)
            self.detectron2_cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = nms_thr
            self.detectron2_predictor = DefaultPredictor(self.detectron2_cfg)
            metadata = MetadataCatalog.get(self.detectron2_cfg.DATASETS.TRAIN[0])
            class_names = metadata.get("thing_classes", None)
            self.infer_model_class_names=class_names
            self.set_class_remap()
            return

        if "," in model_name:
            model_name_list=model_name.split(",")
            model_name=model_name_list[0]
        else:
            model_name_list=[model_name]

         # Support MMDetection models
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

        if model_name in mmdet_models:
            try:
                from mmdet.apis import init_detector
                #from mmengine.structures import InstanceData
                from mmdet.apis import inference_detector
            except ImportError:
                assert False, "try pip install 'mmdet'"

            cfg_path, weight_url = mmdet_models[model_name]
            self.mmdet_inference_detector=inference_detector
            self.mmdet_model = init_detector(cfg_path, weight_url, device="cuda:0")
            self.infer_model_class_names = self.mmdet_model.dataset_meta['classes']
            self.set_class_remap()
            return

        if model_name.endswith(".trt"):
            try:
                import ubon_pycstuff.ubon_pycstuff as upyc
                import json
            except:
                assert False, "TRT models require ubon_pycstuff and JSON"

            self.upyc=upyc
            param_file=""
            if len(model_name_list)>1:
                param_file=model_name_list[1]
            self.fold_attributes=fold_attributes
            self.upyc_infer = upyc.c_infer(model_name, param_file)
            self.model_description=self.upyc_infer.get_model_description()
            self.model_description["engineInfo"]=json.loads(self.model_description["engineInfo"])
            #print(self.model_description)
            self.infer_model_class_names=self.model_description["class_names"]+self.model_description["person_attribute_names"]
            self.first_attribute_class=len(self.model_description["class_names"])
            self.set_class_remap()
            self.upyc_infer.configure({"det_thr":thr,
                                       "nms_thr":nms_thr,
                                       "max_detections":max_det,
                                       "allow_upscale":True})
            return

        if model_name=="upyc_track":
            try:
                import ubon_pycstuff.ubon_pycstuff as upyc
                import json
            except:
                assert False, "upyc_track models require ubon_pycstuff and JSON"

            self.upyc=upyc
            param_file="/mldata/config/track/trackers/uc_reid.yaml"
            if len(model_name_list)>1:
                param_file=model_name_list[1]
            self.track_shared=upyc.c_track_shared_state("/mldata/config/track/trackers/uc_reid.yaml")
            self.track_stream=upyc.c_track_stream(self.track_shared)
            self.infer_model_class_names=["person"]
            self.set_class_remap()

            return

        if isinstance(model_name, list):
            self.ensemble_models=[]
            for m in model_name:
                yolo=inference_wrapper(m,
                                       class_names=class_names,
                                       thr=thr,
                                       nms_thr=nms_thr,
                                       max_det=max_det,
                                       rect=rect,
                                       imgsz=imgsz,
                                       batch_size=batch_size//2)
                self.ensemble_models.append(yolo)
            self.num_threads=self.ensemble_models[0].num_threads
            self.infer_batch_size=self.ensemble_models[0].infer_batch_size
            self.yolo_num_params=sum([x.yolo_num_params for x in self.ensemble_models])
            self.yolo_num_flops=sum([x.yolo_num_flops for x in self.ensemble_models])
            self.infer_cache={}
            return

        import torchvision # needed to make it use fast NMS!
        aug_params={}
        self.ensemble_models=None
        self.model=self.load_ultralytics_model(model_name)
        self.yolo_num_flops=self.yolo_num_flops*self.imgsz*self.imgsz/(640.0*640.0)
        self.num_gpus=stuff.platform_stuff.platform_num_gpus()
        self.num_threads=self.num_gpus
        self.yolo=[None]*self.num_threads
        self.model_params=[aug_params]*self.num_threads
        self.yolo_det_conf=thr
        self.yolo_nms_thr=nms_thr
        self.yolo_half=half
        self.yolo_rect=rect
        self.yolo_max_det=max_det
        self.infer_batch_size=batch_size

        self.face_kp=face_kp
        self.pose_kp=pose_kp
        self.facepose_kp=facepose_kp
        self.fold_attributes=fold_attributes
        self.det_attributes_remap=None
        self.infer_cache={}

        def on_predict_start(predictor: object, persist: bool = False) -> None:
            """
            Enable feature extraction for post-NMS models.

            YOLO26 models default to end-to-end mode (integrated postprocess). In that mode Ultralytics' NMS helper does
            not return anchor indices, so the "save_feats" pipeline in DetectionPredictor can mis-handle outputs.
            We therefore disable feature extraction automatically for end-to-end models.
            """
            # end-to-end models (e.g. YOLO26) do not expose kept anchor indices -> do not enable feats hooks
            is_end2end = bool(getattr(getattr(predictor, "model", None), "end2end", False))
            if is_end2end:
                predictor.save_feats = False
                predictor.expanded_feats = False
                predictor._feats = None
                predictor._feats2 = None
                return

            predictor.save_feats = True
            predictor.expanded_feats = True
            predictor._feats = None
            predictor._feats2 = None

            # Install hooks once per predictor instance; repeated registration leaks memory.
            if getattr(predictor, "_feat_hooks_installed", False):
                return

            # Register hooks to extract input and output of Detect layer
            def pre_hook(module, input):
                predictor._feats = [t.detach() for t in input[0]]

            def post_hook(module, input, output):
                # For most models output is (y, preds_dict); we want raw head output tensor in output[0].
                out0 = output[0] if isinstance(output, (tuple, list)) else output
                if hasattr(out0, "clone"):
                    predictor._feats2 = out0.detach()

            head = predictor.model.model.model[-1]
            predictor._feat_pre_hook_handle = head.register_forward_pre_hook(pre_hook)
            predictor._feat_post_hook_handle = head.register_forward_hook(post_hook)
            predictor._feat_hooks_installed = True

        for i in range(self.num_threads):
            if i==0:
                self.yolo[i]=self.model
            else:
                self.yolo[i]=self.load_ultralytics_model(model_name)
            if self.num_threads!=1:
                self.yolo[i]=self.yolo[i].to("cuda:"+str(i))
            if self.get_feats:
                self.yolo[i].add_callback("on_predict_start", partial(on_predict_start, persist=False))
            assert(self.yolo[i] is not None)
        self.infer_model_class_names=[self.yolo[0].names[i] for i in range(len(self.yolo[0].names))]
        model_attr_names = getattr(getattr(self.yolo[0], "model", None), "attr_names", None)
        if fold_attributes and attributes is None:
            if model_attr_names is not None and len(model_attr_names) > 0:
                # Prefer checkpoint-provided attribute metadata when available.
                attributes = list(model_attr_names)
            else:
                attributes = stuff.attributes_from_class_names(self.infer_model_class_names)
        if model_attr_names is not None and attributes is not None:
            attr_index = {name: i for i, name in enumerate(attributes)}
            # -1 means "attribute exists in model output but is not tracked in target schema".
            self.det_attributes_remap = [attr_index.get(name, -1) for name in model_attr_names]
        self.attributes=attributes
        self.set_class_remap()

    def set_class_remap(self):
        self.det_class_remap=[-1]*len(self.infer_model_class_names)
        if self.class_names is None:
            self.class_names=self.infer_model_class_names
        self.can_detect=[False]*len(self.class_names)

        # map the detected classes back to our class set
        # assume if our class set has things like 'vehicle' then we would want any standard
        # coco classes like 'car' to map to that

        for i,x in enumerate(self.infer_model_class_names):
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
        info=None
        if name.endswith(".engine")==False:
            info=model.info(verbose=False)

        self.yolo_num_params=0
        self.yolo_num_flops=0
        if info is not None:
            self.yolo_num_params=info[1]
            self.yolo_num_flops=info[3]

        return model

    def infer_rfdetr(self, input_frames):
        out_det=[]
        #detections_list = self.rf_detr_model.predict(input_frames, threshold=self.yolo_det_conf)
        detections_list=[]
        img_list=[]
        for i in range(0, len(input_frames), self.rf_detr_batch_size):
            img_batch=input_frames[i:i+self.rf_detr_batch_size]
            for idx, img in enumerate(img_batch):
                if isinstance(img, str):
                    img=Image.open(img).convert("RGB")
                    img_batch[idx]=img
            real_batch_size=len(img_batch)
            for i in range(self.rf_detr_batch_size-real_batch_size):
                img_batch.append(img_batch[real_batch_size-1])
            dets=self.rf_detr_model.predict(img_batch, threshold=self.yolo_det_conf)
            dets=dets[0:real_batch_size]
            detections_list+=dets

        ret=[]
        for j,detections in enumerate(detections_list):
            num_detections=len(detections.xyxy)
            out_det=[]
            w,h=image_stuff.get_image_size(input_frames[j])
            for i in range(num_detections):
                xyxy=detections.xyxy[i]
                box=[xyxy[0]/w, xyxy[1]/h, xyxy[2]/w, xyxy[3]/h]
                #print(int(detections.class_id[i]), COCO_CLASSES[detections.class_id[i]])
                cl=self.det_class_remap[int(detections.class_id[i]+0.01)]
                det={"box":box,
                    "id":None,
                    "class":cl,
                    "confidence":detections.confidence[i]}
                if cl!=-1:
                    out_det.append(det)
            ret.append(out_det)
        return ret

    def infer_mmdet(self, input_frames):
        ret = []
        for img in input_frames:
            if isinstance(img, str):
                image = cv2.imread(img)
            else:
                image = img

            w,h=image_stuff.get_image_size(image)
            det_sample = self.mmdet_inference_detector(self.mmdet_model, image)
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

    def infer_detectron2(self, input_frames):
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

    def infer_upyc_track(self, input_frames):
        from ubon_pycstuff.ubon_pycstuff import c_image
        images=[]
        for i in input_frames:
            if isinstance(i, c_image):
                images.append(i)
            elif isinstance(i, str):
                images.append(self.upyc.load_jpeg(i))
            elif isinstance(i, Image.Image):
                np_img = np.array(i.convert("RGB"))  # Ensures it's in RGB mode
                images.append(self.upyc.c_image.from_numpy(np_img))
            else:
                bgr_img=cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                images.append(self.upyc.c_image.from_numpy(bgr_img))
        self.track_stream.run_on_individual_images(images)
        ret=[]
        for i in range(500):
            track_results=self.track_stream.get_results(True)
            for r in track_results:
                ret.append(r["track_dets"])
            if len(ret)==len(input_frames):
                break
            if i>100:
                time.sleep(0.005)
        assert len(ret)==len(input_frames), "Didn't get all results"
        return ret

    def infer_upyc(self, input_frames):
        from ubon_pycstuff.ubon_pycstuff import c_image
        assert len(input_frames)>0, "Infer UPYC no input frames"
        # build images
        images=[]
        start = time.perf_counter()
        for i in input_frames:
            if isinstance(i, c_image):
                images.append(i)
            elif isinstance(i, str):
                images.append(self.upyc.load_jpeg(i))
            elif isinstance(i, Image.Image):
                np_img = np.array(i.convert("RGB"))  # Ensures it's in RGB mode
                images.append(self.upyc.c_image.from_numpy(np_img))
            else:
                bgr_img=cv2.cvtColor(i, cv2.COLOR_BGR2RGB)
                images.append(self.upyc.c_image.from_numpy(bgr_img))

        # run inference on all images

        start = time.perf_counter()
        batch_dets=self.upyc_infer.run_batch(images)

        # postprocess results
        start = time.perf_counter()
        out_dets = []
        remap = self.det_class_remap
        conf_thresh = self.yolo_det_conf
        first_attr_class = self.first_attribute_class
        fold_attrs = self.fold_attributes

        for dets in batch_dets:
            processed_dets = []
            for d in dets:
                orig_class = remap[d["class"]]
                if orig_class != -1:
                    d["class"] = orig_class
                    processed_dets.append(d)

                if not fold_attrs and "attrs" in d:
                    base_det = {k: v for k, v in d.items() if k != "class" and k != "confidence"}  # shallow copy
                    for i, v in enumerate(d["attrs"]):
                        if v > conf_thresh:
                            new_class = remap[first_attr_class + i]
                            if new_class != -1:
                                det_copy = base_det.copy()
                                det_copy["class"] = new_class
                                det_copy["confidence"] = v
                                processed_dets.append(det_copy)

            out_dets.append(processed_dets)
        elapsed_ms = (time.perf_counter() - start) * 1000
        if elapsed_ms > 5*len(input_frames):
            print(f"inference_wrapper: took {elapsed_ms:.3f} ms for {len(input_frames)} images")
        return out_dets

    def infer_ultralytics(self, input_frames):
        def _flip_kp_inplace(kp):
            # kp format: [x,y,conf] repeated, normalized [0,1]
            if kp is None:
                return
            for i in range(0, len(kp), 3):
                kp[i] = 1.0 - kp[i]

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
            kwargs = dict(
                conf=self.yolo_det_conf,
                iou=self.yolo_nms_thr,
                max_det=self.yolo_max_det,
                agnostic_nms=False,
                half=self.yolo_half,
                imgsz=self.imgsz,
                verbose=False,
                rect=self.yolo_rect,
            )
            # If caller requested features, force non-end2end so kept-anchor indices are available.
            # Otherwise YOLO26 end2end mode breaks the feature extraction pipeline.
            if self.get_feats:
                kwargs["end2end"] = False
            return yolo_fn(frames, **kwargs)

        results = []
        index = 0

        if self.num_threads == 1:
            while index < len(input_frames):
                end = min(len(input_frames), index + self.infer_batch_size)
                batch = input_frames[index:end]
                results += process_batch(self.yolo[0], batch, self.model_params[0])
                index = end
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                while index < len(input_frames):
                    futures = []
                    for i in range(self.num_threads):
                        end = min(len(input_frames), index + self.infer_batch_size)
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
                                             det_attributes_remap=self.det_attributes_remap,
                                             yolo_class_names=self.infer_model_class_names,
                                             class_names=self.class_names,
                                             attributes=self.attributes,
                                             face_kp=self.face_kp,
                                             pose_kp=self.pose_kp,
                                             facepose_kp=self.facepose_kp,
                                             fold_attributes=self.fold_attributes)

            if self.model_params[0] is not None:
                if self.model_params[0].get("flip-lr", False):
                    for d in out_det:
                        # flip normalized xyxy box
                        x1, y1, x2, y2 = d["box"]
                        d["box"] = [1.0 - x2, y1, 1.0 - x1, y2]
                        # flip keypoints (normalized)
                        if "pose_points" in d:
                            _flip_kp_inplace(d["pose_points"])
                        if "face_points" in d:
                            _flip_kp_inplace(d["face_points"])
                        if "facepose_points" in d:
                            _flip_kp_inplace(d["facepose_points"])
            ret.append(out_det)
        return ret

    def infer(self, input_frames):
        if self.ensemble_models is not None:
            results=[]
            for m in self.ensemble_models:
                results.append(m.infer(input_frames))
            return ensemble_merge_results(results)

        if self.rf_detr_model!=None:
            return self.infer_rfdetr(input_frames)

        if self.detectron2_predictor is not None:
            return self.infer_detectron2(input_frames)

        if hasattr(self, "mmdet_model"):
            return self.infer_mmdet(input_frames)

        if hasattr(self, "upyc_infer"):
            return self.infer_upyc(input_frames)

        if hasattr(self, "track_stream"):
            return self.infer_upyc_track(input_frames)

        return self.infer_ultralytics(input_frames)

    def infer_cached(self, index, num_images, get_image_fn):
        if index in self.infer_cache:
            return self.infer_cache[index]
        self.infer_cache={}
        num=self.infer_batch_size*self.num_threads
        num=min(num, num_images-index)
        image_list=[get_image_fn(i+index) for i in range(num)]
        batch_results=self.infer(image_list)
        self.infer_cache={(i+index):batch_results[i] for i in range(num)}
        assert index in self.infer_cache
        return self.infer_cache[index]