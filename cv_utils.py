import numpy as np
import time
import cv2
import copy


hsv_white_range = [(0, 0, 99), (29, 52, 255)]
hsv_brown_range = [(0, 93, 23), (175, 119,  133)]


def get_cow_mask(cow_img):
    hsv_cow_img = cv2.cvtColor(cow_img, cv2.COLOR_RGB2HSV)
    brown_mask = cv2.inRange(hsv_cow_img, hsv_brown_range[0], hsv_brown_range[1])
    white_mask = cv2.inRange(hsv_cow_img, hsv_white_range[0], hsv_white_range[1])
    mask = brown_mask | white_mask
    
    kernel = np.ones((3, 3), np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)
    kernel = np.ones((5, 5), np.uint8)
    dilated_mask = cv2.dilate(eroded_mask, kernel, iterations=2)
    closed_mask = cv2.morphologyEx(dilated_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    
    return closed_mask


def get_bounding_boxes(binary_mask):
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    box_coords = []
    for bb in bounding_boxes:
        center = (bb[0] + bb[2]/2., bb[1] + bb[3]/2.)
        box_coords.append((center, bb))
        
    return box_coords

                    
def draw_boxes(img, bounding_boxes):
    for _, bb in bounding_boxes:
        top_left = (bb[0], bb[1])
        bottom_right = (bb[0] + bb[2], bb[1] + bb[3])
        cv2.rectangle(img, top_left, bottom_right, (255, 255, 255), 2)


def mask_to_img(mask):
     return np.stack([mask, mask, mask], axis=2)


def dist(p1, p2):
    return ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5


class ObjectTracker(object):
    def __init__(self, movement_threshold, size_threshold, time_threshold=5, n_hist=10):
        # Thresholds before a bounding box can no longer be
        # considered a continuation
        self.mt = movement_threshold
        self.st = size_threshold
        self.tt = time_threshold
        self.n_hist = n_hist # Number of previous frame data to keep
        self.curr_id = 0 # Next object ID to be generated
        self.curr_visible_id = 0 # Next object ID to be given to a confirmed object
        
        self.object_map = {} # {id: [shown_id, [(center_t-n, area_t-n), ..., (center_t, area_t)]]}
        
        
    def create_object(self):
        self.object_map[self.curr_id] = [-1, []]
        for _ in range(self.n_hist):
            self.object_map[self.curr_id][1].append(None)
        self.curr_id += 1
        
        return self.curr_id - 1
    
    
    def _get_latest_hist(self, object_id):
        # Gets the more recent non-none (center, area) tuple
        for i in reversed(range(len(self.object_map[object_id][1]))):
            if self.object_map[object_id][1][i]:
                return self.object_map[object_id][1][i]
            
        raise ValueError('There should not be an object with all `None` history!')
        
        
    def check_current_objects(self):
        # Check all current object for updates that need to be made
        del_object_ids = []
        for object_id, (visible_id, object_hist) in self.object_map.items():
            if object_hist == [None] * len(object_hist):
                del_object_ids.append(object_id)
            elif visible_id == -1 and not (None in object_hist[-self.tt:]):
                self.object_map[object_id][0] = self.curr_visible_id
                self.curr_visible_id += 1
        
        for object_id in del_object_ids:
            del self.object_map[object_id]
            
            
    def update_objects(self, bounding_boxes):
        # Check for matching bounding boxes for each object
        for object_id in self.object_map.keys():
            latest_hist = self._get_latest_hist(object_id)
            object_center = latest_hist[0]
            object_area = latest_hist[1]
            
            curr_cand = None # Most likely candidate to be the current object's bounding box
            for i, (box_center, box_coords) in enumerate(bounding_boxes):
                box_area = box_coords[2] * box_coords[3]
                
                # Calculate whether this box is within the candidate threshold
                if dist(object_center, box_center) > self.mt or \
                   abs(box_area - object_area) > self.st:
                    continue
                    
                # If it is then compare with current candidate
                if not curr_cand:
                    curr_cand = (i, box_center, box_area)
                elif dist(object_center, box_center) < dist(object_center, curr_cand[1]):
                    curr_cand = (i, box_center, box_area)
                    
            # Update the current object with the best candidate
            if curr_cand:
                self.object_map[object_id][1].append((curr_cand[1], curr_cand[2]))
                del bounding_boxes[curr_cand[0]]
            else:
                self.object_map[object_id][1].append(None)    
            self.object_map[object_id][1] = self.object_map[object_id][1][1:]
            
        return bounding_boxes
    
    
    def parse_bounding_boxes(self, bounding_boxes):
        # Create new objects for each remaining bounding box
        for i, (box_center, box_coords) in enumerate(bounding_boxes):
            box_area = box_coords[2] * box_coords[3]
            
            new_id = self.create_object()
            self.object_map[new_id][1].append((box_center, box_area))
            self.object_map[new_id][1] = self.object_map[new_id][1][1:]
    
    
    def get_confirmed_objects(self):
        return {v[0]: self._get_latest_hist(k)[0] for k, v in self.object_map.items() if v[0] > -1}
    
    
    def add_frame(self, bounding_boxes):
        bounding_boxes = copy.deepcopy(bounding_boxes)
        
        # Check for matching bounding boxes for each object
        bounding_boxes = self.update_objects(bounding_boxes)
        # Create new objects for each remaining bounding box
        self.parse_bounding_boxes(bounding_boxes)
        # Check all current object for updates that need to be made
        self.check_current_objects()
        
        return self.get_confirmed_objects()