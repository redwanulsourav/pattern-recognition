import cv2 as cv
import argparse
import numpy as np

class Context:
    def __init__(self):
        self.center = None
        self.radii = None

    def set_center(self, center_coords):
        # assert type(center_coords) == '<class \'tuple\'>' , 'Coordinates should be of type tuple'
        assert len(center_coords) == 2, 'Tuple should have two members'

        self.center = center_coords
    
    def set_radii(self, radii):
        self.radii = radii
    
    
def main():

    ctx = Context()

    img = cv.imread('input/1.png')
    bgr_img = img.copy()

    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    _, img = cv.threshold(img, 125, 255, cv.THRESH_BINARY_INV)
    contours, hierarchy = cv.findContours(img, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    contour_areas = [(cv.contourArea(contour), contour) for contour in contours]
    contour_areas.sort(key = lambda tup: tup[0])
    
    (x, y), radius = cv.minEnclosingCircle(contour_areas[-1][1])
    (x1, y1), radius1 = cv.minEnclosingCircle(contour_areas[-2][1])
    (x2, y2), radius2 = cv.minEnclosingCircle(contour_areas[-3][1])
    
    center = (int(x), int(y))
    
    ctx.set_center(center)
    ctx.set_radii([int(radius), int(radius1), int(radius2)])

    masks = [np.zeros_like(img) for radius in ctx.radii]
    pts = []
    kernel = np.ones((2, ), np.uint8)

    for idx, mask in enumerate(masks):
        cv.circle(mask, center, ctx.radii[idx] - 4, 255, 2)
        temp_img = cv.bitwise_and(img, img, mask = mask)
        temp_img2 = mask - temp_img
        eroded = cv.erode(temp_img2, kernel, iterations = 1)
        poi_locations = np.argwhere(eroded == 255)
        pt1 = ctx.center
        pt2 = poi_locations.mean(0)
        pt2 = (int(pt2[1]), int(pt2[0]))
        cv.line(bgr_img, pt1, pt2, (0, 255, 0), 2)
        
        cv.imshow(f"Mask {idx}", mask)
        cv.imshow(f'Temp image {idx}', temp_img2)
        cv.imshow(f'Eroded {idx}', eroded)
        #print(pt2)
    
    
    # cv.circle(bgr_img, center1, radius1, (0, 255, 0), 2)
    # cv.circle(bgr_img, center2, radius2, (0, 255, 0), 2)
    
    # bgr_img = cv.drawContours(bgr_img, contours, -1, (0, 255, 0), -1)
    
    # cv.imshow('Threshold', img)
    # cv.imshow('Image 2', img2)
    cv.imshow('Output', bgr_img)

    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == '__main__':
    main()