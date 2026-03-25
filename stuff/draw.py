"""Draw primitives (lines, boxes, circles, text) on OpenCV BGR images using normalized [0,1] coordinates."""
import cv2
import stuff.coord as coord
import time

def set_colour(clr, chan, default, default_alpha=255):

    colours={"red":[0,0,255],
             "orange":[0,128,255],
             "green":[0,255,0],
             "cyan":[255,255,0],
             "blue":[255,0,0],
             "yellow":[0,255,255],
             "white":[255,255,255],
             "black":[0,0,0]}

    alphas={"solid":255,
            "half":128,
            "flashing":128,
            "transparent":64}

    if clr is None:
        clr=default
    if isinstance(clr, str):
        alpha=None
        if "_" in clr:
            alpha,clr=clr.split("_")
        assert clr in colours, f"unknown colour {clr}"
        clr=colours[clr]
        if alpha in alphas and chan==4:
            clr=[alphas[alpha]]+clr
            if alpha=="flashing":
                t=int(time.time()*512) & 511
                if t>255:
                    t=511-t
                clr[0]=t
    if chan==4 and len(clr)==3:
        clr=[default_alpha]+clr
    assert chan==len(clr), "Bad colour size"
    return clr

def draw_line(img, start, stop, clr=None, thickness=1):
    height, width, chan = img.shape
    p0=[int(coord.clip01(start[0])*width), int(coord.clip01(start[1])*height)]
    p1=[int(coord.clip01(stop[0])*width), int(coord.clip01(stop[1])*height)]
    clr=set_colour(clr, chan, "white")
    cv2.line(img, p0, p1, clr, thickness=thickness)

def draw_box(img, box, clr=None, thickness=1):
    height, width, chan = img.shape
    p0=[int(coord.clip01(box[0])*width), int(coord.clip01(box[1])*height)]
    p1=[int(coord.clip01(box[2])*width), int(coord.clip01(box[3])*height)]
    clr=set_colour(clr, chan, "white")
    cv2.rectangle(img, p0, p1, clr, thickness)

def draw_circle(img, centre, radius, clr=None, thickness=1):
    height, width, chan= img.shape
    p=[int(coord.clip01(centre[0])*width), int(coord.clip01(centre[1])*height)]
    r=int(radius*width+0.5)
    clr=set_colour(clr, chan, "white")
    cv2.circle(img, p, r, clr, -1)

def draw_text(img, text, xc, yc, img_bg=None,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              fontScale=0.65,
              fontColor=None,
              bgColor=None,
              lineType=2,
              thickness=1
              ):
    height, width, chan = img.shape

    fontColor=set_colour(fontColor, chan, "white", default_alpha=128)
    bgColor=set_colour(bgColor, chan, "black", default_alpha=64)

    x=(int)(xc*width)
    y=(int)(yc*height)
    text_split=text.split("\n")

    if img_bg is None:
        img_bg=img

    for t in text_split:
        xp=x
        yp=y
        (text_width, text_height) = cv2.getTextSize(t,
                                                    font,
                                                    fontScale=fontScale,
                                                    thickness=thickness)[0]
        box_coords = ((xp, yp+2),
                      (xp + text_width + 2, yp - text_height - 2))
        y+=text_height+5

        cv2.rectangle(img_bg, box_coords[0], box_coords[1], bgColor, cv2.FILLED)
        cv2.putText(img,
                    t,
                    (xp, yp),
                    font,
                    fontScale,
                    fontColor,
                    thickness,
                    lineType)
