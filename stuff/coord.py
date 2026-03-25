"""Axis-aligned boxes, points, and ROI mapping in normalized [0,1] coordinates."""
import copy


def clip01(x):
    if x<0:
        return 0
    if x>1:
        return 1
    return x

def unmap_roi_point_inplace(roi, pt):
    pt[0]=roi[0]+pt[0]*(roi[2]-roi[0])
    pt[1]=roi[1]+pt[1]*(roi[3]-roi[1])

def unmap_roi_box_inplace(roi, box):
    box[0]=roi[0]+box[0]*(roi[2]-roi[0])
    box[1]=roi[1]+box[1]*(roi[3]-roi[1])
    box[2]=roi[0]+box[2]*(roi[2]-roi[0])
    box[3]=roi[1]+box[3]*(roi[3]-roi[1])

def unmap_roi_point(roi, pt):
    ret=[roi[0]+pt[0]*(roi[2]-roi[0]),
         roi[1]+pt[1]*(roi[3]-roi[1])]
    return ret

def unmap_roi_box(roi, box):
    ret=[roi[0]+box[0]*(roi[2]-roi[0]),
         roi[1]+box[1]*(roi[3]-roi[1]),
         roi[0]+box[2]*(roi[2]-roi[0]),
         roi[1]+box[3]*(roi[3]-roi[1])]
    return ret

def unmap_roi_keypoints_inplace(roi, keypoints):
    for i in range(len(keypoints)//3):
        keypoints[i*3+0]=roi[0]+keypoints[i*3+0]*(roi[2]-roi[0])
        keypoints[i*3+1]=roi[1]+keypoints[i*3+1]*(roi[3]-roi[1])

def box_w(b1):
    """
    Return width of box
    """
    return b1[2]-b1[0]

def box_h(b1):
    """
    Return height of box
    """
    return b1[3]-b1[1]

def box_a(b1):
    """
    Return area of box
    """
    return (b1[3]-b1[1])*(b1[2]-b1[0])

def box_i(b1, b2):
    """
    Return area of box intersection
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    return ai

def box_union(b1, b2):
    """
    Return union of two boxes
    """
    return [min(b1[0], b2[0]), min(b1[1], b2[1]), max(b1[2], b2[2]), max(b1[3], b2[3])]

def box_expand(b, f):
    """
    Expand box by a scale factor (1.5 = 50% wider/taller)
    """
    w=b[2]-b[0]
    h=b[3]-b[1]
    s=(f-1)/2.0
    w=w*s
    h=h*s
    return [clip01(b[0]-w), clip01(b[1]-h), clip01(b[2]+w), clip01(b[3]+h)]

def box_iou(b1, b2):
    """
    Compute IoU between two boxes in [x1, y1, x2, y2] format.
    """
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    iw = x2 - x1
    ih = y2 - y1
    if iw <= 0 or ih <= 0:
        return 0.0

    inter_area = iw * ih
    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area

def box_iou_relaxed_person(b1, b2):
    iou=box_iou(b1,b2)
    if iou==0:
        return 0
    b1m=copy.copy(b1)
    b2m=copy.copy(b2)
    b1m[3]=min(b1[3], b1[1]+b1[2]-b1[0])
    b2m[3]=min(b2[3], b2[1]+b1[2]-b1[0])
    return 0.5*iou+0.5*box_iou(b1m, b2m)

def box_ioma(b1, b2):
    """
    Computes the ioma (intersection over minimum
    area) between two boxes

    Args:
        b1, b2: list [x1,y1,x2,y2] in xyxy format

    Returns:
        float iou
    """
    iw=max(0, min(b1[2],b2[2])-max(b1[0],b2[0]))
    ih=max(0, min(b1[3],b2[3])-max(b1[1],b2[1]))
    ai=iw*ih
    a1=(b1[2]-b1[0])*(b1[3]-b1[1])
    a2=(b2[2]-b2[0])*(b2[3]-b2[1])
    iou=(iw*ih)/(min(a1,a2)+1e-7)
    return iou

def point_in_box(pt, box):
    if pt[0]<box[0] or pt[0]>box[2]:
        return None
    if pt[1]<box[1] or pt[1]>box[3]:
        return None
    d=abs(pt[0]-0.5*(box[0]+box[2]))
    d+=abs(pt[1]-0.5*(box[1]+box[3]))
    return d

def interpolate2(x, y, f):
    """Linear interpolation: (1-f)*x + f*y. Supports scalar or list; returns new value(s)."""
    if isinstance(x, (float, int)) and isinstance(y, (float, int)):
        return (1.0 - f) * x + f * y
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            raise ValueError("x and y lists must have the same length")
        return [(1.0 - f) * x[i] + f * y[i] for i in range(len(x))]
    raise TypeError("interpolate2: x and y must be numeric or lists of same length")


def interpolate(x, y, f):
    """Linear interpolation in-place for lists: x[i] = (1-f)*x[i] + f*y[i]. No return for list; returns for scalar."""
    if isinstance(x, (float, int)) and isinstance(y, (float, int)):
        return (1.0 - f) * x + f * y
    if isinstance(x, list) and isinstance(y, list):
        if len(x) != len(y):
            raise ValueError("x and y lists must have the same length")
        for i in range(len(x)):
            x[i] = (1.0 - f) * x[i] + f * y[i]
        return
    raise TypeError("interpolate: x and y must be numeric or lists of same length")