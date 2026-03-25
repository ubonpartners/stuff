"""Unmap detection boxes and keypoints from a normalized ROI to full-image coordinates."""
import copy
import stuff.coord as coord
import stuff.match as match

def unmap_detection_inplace(det, roi):
    coord.unmap_roi_box_inplace(roi, det["box"])
    if "subbox" in det:
         coord.unmap_roi_box_inplace(roi, det["subbox"])
    if "pose_points" in det:
        coord.unmap_roi_keypoints_inplace(roi, det["pose_points"])
    if "face_points" in det:
        coord.unmap_roi_keypoints_inplace(roi, det["face_points"])