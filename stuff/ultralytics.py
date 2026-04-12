import copy
import stuff.coord as coord
import stuff.match as match

def basic_nms(dets, iou_thresh=0.5):
    # Sort detections by confidence descending
    dets = sorted(dets, key=lambda d: d['confidence'], reverse=True)
    kept = []

    while dets:
        best = dets.pop(0)
        kept.append(best)
        remaining = []
        for d in dets:
            # Only suppress boxes of the same class
            if best['class'] != d['class'] or coord.box_iou(best['box'], d['box']) <= iou_thresh:
                remaining.append(d)
        dets = remaining

    return kept

def is_large(x):
    return x["box"][2]-x["box"][0]>=0.1

def has_face_points(x):
    """
    Return True if the annotation has non-trivial face points (i.e. not all 0)
    """
    t=0
    if "face_points" in x:
        for i in range(5):
            t+=x["face_points"][3*i+2]
    if t<0.1:
        return False
    return True

def has_pose_points(x):
    """
    Return True if the annotation has non-trivial pose points (i.e. not all 0)
    """
    t=0
    if "pose_points" in x:
        for i in range(17):
            t+=x["pose_points"][3*i+2]
    if t<0.1:
        return False
    return True

def better_annotation(a1, a2):
    """
    return True if a1 is 'better' than a2
    better is currently the one that has pose, face points if the other doesn't else the largest
    """
    a1_pp=has_pose_points(a1)
    a2_pp=has_pose_points(a2)
    if a1_pp!=a2_pp:
        return a1_pp
    a1_fp=has_face_points(a1)
    a2_fp=has_face_points(a2)
    if a1_fp!=a2_fp:
        return a1_fp
    return coord.box_a(a1["box"])>coord.box_a(a2["box"])

def to_the_left(gt, point_type, index_of_first):
    if point_type in gt:
        pt_x_1=gt[point_type][3*index_of_first+0]
        pt_x_2=gt[point_type][3*index_of_first+3]
        box_w=gt["box"][2]-gt["box"][1]
        if pt_x_2>pt_x_1 and gt[point_type][3*index_of_first+2]>0 and gt[point_type][3*index_of_first+2]>0:
            print(f"Weird pose: {point_type} {index_of_first} {pt_x_2}>{pt_x_1} {pt_x_2-pt_x_1} {box_w}")

def to_the_right(gt, point_type, index_of_first):
    if point_type in gt:
        pt_x_1=gt[point_type][3*index_of_first+3]
        pt_x_2=gt[point_type][3*index_of_first+0]
        if pt_x_2>pt_x_1 and gt[point_type][3*index_of_first+2]>0 and gt[point_type][3*index_of_first+2]>0:
            print(f"Weird pose: {point_type} {index_of_first} {pt_x_2}>{pt_x_1}")

def check_pose_points(gt):
    return
    to_the_left(gt, "pose_points", 1)
    to_the_left(gt, "pose_points", 3)
    to_the_left(gt, "pose_points", 5)
    to_the_left(gt, "pose_points", 7)
    to_the_left(gt, "pose_points", 9)
    to_the_left(gt, "pose_points", 11)
    to_the_left(gt, "pose_points", 13)
    to_the_left(gt, "pose_points", 15)
    to_the_right(gt, "face_points", 0)
    to_the_right(gt, "face_points", 3)


def get_face_triangle_points(gt):
    # returns nose, left eye, right eye - can be from face points, pose points
    if "pose_points" in gt:
        return gt["pose_points"][0:3*3]
    if "face_points" in gt:
        ret=[0]*3*3
        ret[0:3]=gt["face_points"][6:9]
        ret[3:6]=gt["face_points"][3:6]
        ret[6:9]=gt["face_points"][0:3]
        return ret
    return None

def map_one_gt_keypoints(gt, face_points, pose_points, facepose_points, delete_others=True):
    # Coco/Facepose order          Facepoint order
    # 0 - Nose                     0-Right eye
    # 1 - Left eye                 1-Left eye
    # 2 - Right eye                2-Nose
    # 3 - Left Ear                 3-Right mouth
    # 4 - Right Ear                4-Left mouth
    # 5 - Left Shoulder
    # 6 - Right Shoulder
    # 7 - Left Elbow
    # 8 - Right Elbow
    # 9 - Left Wrist
    # 10 -Right Wrist
    # 11 -Left Hip
    # 12 -Right Hip
    # 13 -Left Knee
    # 14 -Right Knee
    # 15 -Left Ankle
    # 16 -Right Ankle
    # 17 -Left mouth
    # 18 -Right mouth

    # make face points from facepose points
    if face_points and not "face_points" in gt:
        gt["face_points"]=[0]*5*3

    # make pose points from facepose points
    if pose_points and not "pose_points" in gt:
        gt["pose_points"]=[0]*17*3

    if delete_others:
        if pose_points is False:
            if "pose_points" in gt:
                del gt["pose_points"]

        if face_points is False:
            if "face_points" in gt:
                del gt["face_points"]

def map_keypoints(gts, face_points, pose_points, facepose_points):
    for gt in gts:
        map_one_gt_keypoints(gt, face_points, pose_points, facepose_points)

def unpack_yolo_keypoints(det_kp_list, det_kp_conf_list, index):
    if det_kp_list is None:
        return None, None, None, None

    det_kp = det_kp_list[index]
    num_kp = len(det_kp)

    # Use confidence values if provided, else default to 1.0
    det_kp_conf = (
        det_kp_conf_list[index]
        if det_kp_conf_list is not None
        else [1.0] * num_kp
    )

    flat_kp = []
    for (x, y), conf in zip(det_kp, det_kp_conf):
        flat_kp.extend([x, y, 0.0 if x <= 0 and y <= 0 else conf])

    length = len(flat_kp)

    if length == 51:      # 17 keypoints (pose)
        return None, flat_kp, None, None
    elif length == 66:    # 22 keypoints (face + pose)
        return flat_kp[:15], flat_kp[15:], None, None
    elif length == 15:    # 5 keypoints (face only)
        return flat_kp, None, None, None
    else:
        print(f"Bad number of yolo keypoints: {length}")
        return None, None, None, None

def attr_match(d,d2,class_to_attribute_map):
    # match attribute detections to base detections
    # returns a score for if d2 is a base object for d
    # i.e. if d is a "person_with_hat" detection and d2 is "person" then
    # the iou of d2 with d is returned

    m=class_to_attribute_map[d["class"]]
    if m is None:
        return 0.0
    base_class_index=m["base_class_index"]
    attr_index=m["attr_index"]
    if attr_index is None:
        return 0.0
    if d2["class"]!=base_class_index:
        return 0.0
    return coord.box_iou(d2["box"], d["box"])

def attr_partition_fn(obj, context):
    return obj["partition_mask"]

def fold_detections_to_attributes(gts, class_names, attributes):
    """
    Remove GT boxes that correspond to class attributes and build
    and 'attr' vector attched to the primary GT instead
    e.g. separate person_male GT will become the person:male
    attribute of the person GT with the same box as the original
    person_make
    """

    # set up class_to_attribute_map
    # this finds object detector classes that are really detecting
    # attributes for objects, e.g. the attribute person:is_female is
    # detected with a class person_is_female
    # this is going to let us remove those detections and map them
    # back to a vector of attributes in the base object

    class_to_attribute_map=None
    if attributes is not None:
        class_to_attribute_map=[None]*len(class_names)
        for i,a in enumerate(attributes):
            base_class=a.split(":")[0]
            if base_class in class_names:
                base_class_index=class_names.index(base_class)
                an=a.replace(":","_")
                if an in class_names:
                    j=class_names.index(an)
                    class_to_attribute_map[j]={"base_class_index":base_class_index,
                                                    "attr_index":i}
    if class_to_attribute_map is not None:

        # replace attribute detections with attribute properties in the
        # underlying object

        if True:
            # optimized assignment as this is a miserable N^2 thing by
            # default. If you have 200 detections this gets pretty ugly to
            # match all with all....
            for d in gts:
                d["partition_mask"]=match.uniform_grid_partition(d["box"], context=[4,16])

            new_ind, old_ind, scores=match.match_lsa2(gts, gts,
                            mfn=attr_match, mfn_context=class_to_attribute_map,
                            partition_fn=attr_partition_fn, max_partitions=64,
                            match_method="greedy_multi_match")

            for i,n in enumerate(new_ind):
                d=gts[new_ind[i]]
                d2=gts[old_ind[i]]
                m=class_to_attribute_map[d["class"]]
                assert m is not None
                base_class_index=m["base_class_index"]
                attr_index=m["attr_index"]
                assert d2["class"]==base_class_index
                if not "attrs" in d2:
                    d2["attrs"]=[0]*len(attributes)
                d2["attrs"][attr_index]=max(d2["attrs"][attr_index], d["confidence"])
                d["confidence"]=0
        else:
            for d in gts:
                m=class_to_attribute_map[d["class"]]
                base_class_index=m["base_class_index"]
                attr_index=m["attr_index"]
                if attr_index is not None:
                    best_iou=0
                    for i,d2 in enumerate(gts):
                        if d2["class"]==base_class_index:
                            iou=coord.box_iou(d2["box"], d["box"])
                            if iou>best_iou:
                                best_iou=iou
                                best_match=i
                    if best_iou>0.3:
                        print(best_iou)
                        d2=gts[best_match]
                        if not "attrs" in d2:
                            d2["attrs"]=[0]*len(attributes)
                        d2["attrs"][attr_index]=max(d2["attrs"][attr_index], d["confidence"])
                        d["confidence"]=0
        # delete attribute detections
        gts=[x for x in gts if x["confidence"]!=0]
    return gts

def yolo_results_to_dets(results,
                         det_thr=0.01,
                         det_class_remap=None,
                         det_attributes_remap=None,
                         yolo_class_names=None,
                         class_names=None,
                         attributes=None,
                         face_kp=False,
                         pose_kp=False,
                         facepose_kp=False,
                         fold_attributes=True,
                         params=None):

    det_boxes = results.boxes.xyxyn.tolist() # center
    det_classes = results.boxes.cls.tolist()
    #print(det_classes)
    det_confidences = results.boxes.conf.tolist()
    out_det=[]

    if det_class_remap is None:
        nc=1
        if len(det_classes)>0:
            nc=int(max(det_classes))+1
        det_class_remap=list(range(0, nc))

    det_kp_list=None
    det_kp_conf_list=None
    if hasattr(results, "keypoints") and results.keypoints is not None:
        det_kp_list=results.keypoints.xyn.tolist()
        if results.keypoints.has_visible:
            det_kp_conf_list=results.keypoints.conf.tolist()

    num_det=len(det_boxes)
    indexes=[i for i in range(num_det) if det_class_remap[int(det_classes[i])]!=-1
             and det_confidences[i]>det_thr] # remove detected classes we are not interested in

    if hasattr(results.boxes, "id") and results.boxes.id is not None:
        det_ids=results.boxes.id.tolist()
    else:
        det_ids=[None]*len(det_boxes)

    if hasattr(results, "feats"):
        feats=results.feats
    else:
        feats=[None]*len(det_boxes)

    if hasattr(results, "reid_embeddings"):
        det_reid_embeddings=results.reid_embeddings.tolist()
    else:
        det_reid_embeddings=[None]*len(det_boxes)

    # new v10 attribute head support
    if hasattr(results, "attributes") and results.attributes is not None:
        det_attributes=results.attributes.tolist()
    else:
        det_attributes=None

    for i in indexes:
        det={"box":det_boxes[i],
             "id":det_ids[i],
             "class":det_class_remap[int(det_classes[i])],
             "confidence":det_confidences[i],
             "feats":feats[i],
             "reid_vector":det_reid_embeddings[i]}

        # remap the attributes vector from the supported model attributes to the dataset attributes
        if det_attributes is not None and det_attributes[i] is not None:
            if attributes is None:
                attributes = []
            if det_attributes_remap is None:
                det_attributes_remap = list(range(len(det_attributes[i])))
                if len(attributes) == 0:
                    attributes = [f"attr_{k}" for k in range(len(det_attributes[i]))]
            det["attrs"]=[0]*len(attributes)
            for j,v in enumerate(det_attributes[i]):
                if j >= len(det_attributes_remap):
                    continue
                ridx = det_attributes_remap[j]
                if ridx is None or ridx < 0 or ridx >= len(det["attrs"]):
                    continue
                det["attrs"][ridx]=det_attributes[i][j]

        fp, pp, fpp, attr=unpack_yolo_keypoints(det_kp_list, det_kp_conf_list, i)
        if fp is not None:
            det["face_points"]=fp
        if pp is not None:
            det["pose_points"]=pp
        if attr is not None:
            assert (len(attr)%3)==0
            num_poseattr=len(attr)//3
            poseattr=[0]*num_poseattr
            for i in range(num_poseattr):
                poseattr[i]=attr[3*i+2]
            det["poseattr"]=poseattr
        if det["class"]!=-1:
            out_det.append(det)

    person_class=-1
    face_class=-1
    if "person" in class_names:
        person_class=class_names.index("person")
    if "face" in class_names:
        face_class=class_names.index("face")

    for d in out_det:
        map_one_gt_keypoints(d, face_kp, pose_kp, facepose_kp)

        if face_kp and d["class"]==face_class:
            if "pose_points" in d:
                del d["pose_points"]

    assert fold_attributes==True, "Must have folded attributes in yolo_results_to_dets"

    if fold_attributes and det_attributes is None:
        out_det=fold_detections_to_attributes(out_det, class_names, attributes)

    pose_nms=None
    pose_area_limit=None
    pose_expand=None
    nms_iou=None
    if params is not None:
        if "pose_nms" in params:
            pose_nms=params["pose_nms"]
        if "pose_area_limit" in params:
            pose_area_limit=params["pose_area_limit"]
        if "pose_expand" in params:
            pose_expand=params["pose_expand"]
        if "nms_iou" in params:
            nms_iou=params["nms_iou"]

    if pose_nms is not None or pose_area_limit is not None:
        p_index=[]
        for i,d in enumerate(out_det):
            if d["class"]==person_class:
                num=0
                if "pose_points" in d:
                    for j in range(len(d["pose_points"])//3):
                        if d["pose_points"][j*3+2]>0.01:
                            num+=1
                if num>=2:
                    p_index.append(i)
                else:
                    if pose_area_limit is not None and coord.box_a(d["box"])>pose_area_limit:
                        d["confidence"]=0
        if pose_expand is not None:
            for i in p_index:
                d=out_det[i]
                pose_box=d["box"].copy()
                for j in range(len(d["pose_points"])//3):
                    if d["pose_points"][j*3+2]>0.1:
                        pose_box[0]=min(pose_box[0], d["pose_points"][j*3+0])
                        pose_box[2]=max(pose_box[2], d["pose_points"][j*3+0])
                        pose_box[1]=min(pose_box[1], d["pose_points"][j*3+1])
                        pose_box[3]=max(pose_box[3], d["pose_points"][j*3+1])
                for i in range(4):
                    d["box"][i]=pose_expand*pose_box[i]+(1.0-pose_expand)*d["box"][i]
        #kp_iou2(kp_gt, kp_det, s, num_pt)
        if pose_nms is not None:
            for n,i in enumerate(p_index):
                d=out_det[i]
                if d["confidence"]==0:
                    continue
                for m in range(n+1, len(p_index)):
                    j=p_index[m]
                    d2=out_det[j]
                    ioma=coord.box_ioma(d["box"],d2["box"])
                    if ioma==0:
                        continue
                    #if ioma>0.9:
                    #    d2["confidence"]=0
                    #    continue
                    iou=kp_iou2(d["pose_points"], d2["pose_points"], max(coord.box_a(d["box"]), coord.box_a(d2["box"])), len(d["pose_points"])//3)
                    #print(d["pose_points"])
                    #print(d2["pose_points"])
                    #print(box_iou(d["box"],d2["box"]), iou, len(d["pose_points"])//3)
                    if iou>pose_nms:
                        f=d["confidence"]/(d["confidence"]+d2["confidence"])
                        for i in range(4):
                            d["box"][i]=f*d["box"][i]+(1.0-f)*d2["box"][i]
                        #print(f"remove keypoint! {iou} {d["confidence"]} {d2["confidence"]}")
                        #if d2["confidence"]>0.2:
                        #    img=results.orig_img
                        #    an=[d,d2]
                        #    img=draw_boxes(img, an)
                        #    display_image_wait_key(img)
                        d2["confidence"]=0

        out_det2=[]
        for d in out_det:
            if d["confidence"]!=0:
                out_det2.append(d)
        #print(f"removed {len(out_det)-len(out_det2)}")
        out_det=out_det2

        for i,d in enumerate(out_det):
            if d["class"]==person_class:
                for j in range(i+1, len(out_det)):
                    d2=out_det[j]
                    if d["confidence"]==0:
                        continue
                    if d2["class"]==person_class and coord.box_iou(d["box"],d2["box"])>nms_iou:
                        d2["confidence"]=0

        out_det2=[]
        for d in out_det:
            if d["confidence"]!=0:
                out_det2.append(d)
        out_det=out_det2

    return out_det

def kp_line(display, kp, pts, thickness=2, clr="half_blue"):
    a=pts[0]
    b=pts[1]
    c=None
    if len(pts)>2:
        c=pts[2]
    x0=kp[3*a+0]
    y0=kp[3*a+1]
    v0=kp[3*a+2]
    x1=kp[3*b+0]
    y1=kp[3*b+1]
    v1=kp[3*b+2]
    if v0==0:
        return
    if v1==0:
        return
    if c is not None:
        x2=kp[3*c+0]
        y2=kp[3*c+1]
        v2=kp[3*c+2]
        if v2==0:
            return
        x1=0.5*(x1+x2)
        y1=0.5*(y1+y2)
    display.draw_line([x0,y0], [x1,y1], clr, thickness=thickness)

def draw_pose(display, kp=None, pose_pos=None, pose_conf=None, thickness=2, clr="half_blue"):
    if kp is None:
        assert pose_pos is not None, "one of kp and pose_pos,conf needs to be set"
        n=len(pose_conf)
        kp=[0]*3*n
        for i in range(n):
            kp[3*i+0]=pose_pos[i][0]
            kp[3*i+1]=pose_pos[i][1]
            kp[3*i+2]=pose_conf[i+0]

    lines=[[0,1],[0,2],[0, 5, 6],[1, 3],[2, 4],[5, 6],
            [5, 11],[6,12],[11,12],[5,7],[7,9],[6,8],
            [8,10],[11,13],[13,15],[12,14],[14,16]]
    for l in lines:
        kp_line(display, kp, l, thickness=thickness, clr=clr)

def draw_boxes(display,
               an,
               class_names=None,
               highlight_index=None,
               alt_clr=False,
               attributes=None,
               extra_text=None
               ):

    for index,a in enumerate(an):
        highlight=index==highlight_index
        if alt_clr:
            clr="half_cyan"
        else:
            clr="half_green"
        thickness=2
        if highlight:
            clr="flashing_yellow"
            thickness=4
        display.draw_box(a["box"], clr=clr, thickness=thickness)
        if "subbox" in a:
            display.draw_box(a["subbox"], clr="half_cyan", thickness=thickness)
            display.draw_text("fsb", a["subbox"][0], a["subbox"][3])

        if class_names==None or not a["class"]<len(class_names):
            label=f"Class_{a['class']}"
        else:
            label=class_names[a["class"]]
        label+=" "
        label+="{:4.2f} ".format(a["confidence"])

        if "face_points" in a:
            fp=a["face_points"]
            for i in range(5):
                if fp[3*i+2]!=0:
                    clr="half_red"
                    if i==0 or i==3: # RIGHT points
                        clr="half_yellow"
                    display.draw_circle([fp[3*i+0], fp[3*i+1]], radius=0.002, clr=clr)

        if "pose_points" in a:
            kp=a["pose_points"]
            draw_pose(display, kp=kp, thickness=thickness)

        display.draw_text(label, a["box"][0], a["box"][3])

    if highlight_index is not None and extra_text is None:
        extra_text=""
        a=an[highlight_index]
        extra_text+=f"Class: {class_names[a['class']]}\n"
        if "track_id" in a and a["track_id"]!=0:
            extra_text+=f"ID : {a['track_id']:x}\n"
        extra_text+=f"Box: L:{a['box'][0]:0.3f} T:{a['box'][1]:0.3f} R:{a['box'][2]:0.3f} B:{a['box'][3]:0.3f}\n"
        if 'subbox' in a:
            extra_text+=f"SubBox: L:{a['subbox'][0]:0.3f} T:{a['subbox'][1]:0.3f} R:{a['subbox'][2]:0.3f} B:{a['subbox'][3]:0.3f}\n"
        extra_text+=f"W:{(a['box'][2]-a['box'][0]):0.3f} H:{(a['box'][3]-a['box'][1]):0.3f}"
        if 'subbox' in a:
            extra_text+=f"SW:{(a['subbox'][2]-a['subbox'][0]):0.3f} SH:{(a['subbox'][3]-a['subbox'][1]):0.3f} C:{a['subbox_conf']:0.2f} "
        if "fiqa_score" in a:
            extra_text+=f" FIQA {a['fiqa_score']:0.3f}"
        extra_text+="\n"
        attribute_text=""
        if attributes is not None:
            if "attrs" in a:
                l=list(range(len(a["attrs"])))
                l=[x for _, x in sorted(zip(a["attrs"], l), reverse=True)]
                for i in l:
                    attr=a["attrs"][i]
                    if attr>0.2:
                        attribute_text+=f"{attributes[i]}={attr:0.2f}\n"
        extra_text+=attribute_text
        if "feats" in a and a["feats"] is not None:
            f=a["feats"]
            extra_text+=f"\nFeat len {len(f)} :"
            for i in f[-8:-5]:
                extra_text+=f" {int(i):d}"

    if extra_text is not None:
        display.draw_text(extra_text, 0.05, 0.05, unmap=False, fontScale=0.5)

def find_gt_from_point(gts, x, y):
    ret=None
    best_d=1000000
    for i,gt in enumerate(gts):
        if x<gt["box"][0] or x>gt["box"][2] or y<gt["box"][1] or y>gt["box"][3]:
            continue
        cx=(gt["box"][0]+gt["box"][2])*0.5
        cy=(gt["box"][1]+gt["box"][3])*0.5
        d=(x-cx)*(x-cx)+(y-cy)*(y-cy)
        if d<best_d:
            ret=i
            best_d=d
    return ret, best_d

def make_class_remap_table(s_cl, d_cl, missing_val=None):
    remap=[missing_val]*len(s_cl)
    for i,cl in enumerate(d_cl):
        if cl in s_cl:
            remap[s_cl.index(cl)]=i
    return remap

def attributes_from_class_names(class_names):
    """
    Detect class names that correspond to attributes of a base
    class. Returns a list of attribtues
    e.g. person_male class -> person:male attribute
    """
    attributes=[]
    for c in class_names:
        if c.startswith("person_"):
            attributes.append(c.replace("person_", "person:"))
    for c in class_names:
        if c.startswith("face_"):
            attributes.append(c.replace("face_", "face:"))
    return attributes