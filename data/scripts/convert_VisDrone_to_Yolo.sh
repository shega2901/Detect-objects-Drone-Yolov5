#Format of VisDrone text file (folder annotations)(http://aiskyeye.com/evaluate/results-format/): 

#<bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>
#   <bbox_left>	The x coordinate of the top-left corner of the predicted bounding box
#   <bbox_top>	The y coordinate of the top-left corner of the predicted object bounding box
#   <bbox_width>	The width in pixels of the predicted object bounding box
#   <bbox_height>	The height in pixels of the predicted object bounding box
#<score>		The score in the DETECTION result file indicates the confidence of the predicted bounding box enclosing an object instance.
#			The score in GROUNDTRUTH file is set to 1 or 0. 1 indicates the bounding box is considered in evaluation, 
#			while 0 indicates the bounding box will be ignored.
#<object_category>	The object category indicates the type of annotated object, 
#			(i.e., ignored regions (0), pedestrian (1), people (2), bicycle (3), car (4), van (5), truck (6), tricycle (7), 
#			awning-tricycle (8), bus (9), motor (10), others (11))
#<truncation>		The score in the DETECTION result file should be set to the constant -1. The score in the GROUNDTRUTH file indicates 
#			the degree of object parts appears outside a frame (i.e., no truncation = 0 (truncation ratio 0%), 
#			and partial truncation = 1(truncation ratio 1% ∼ 50%)).
#<occlusion>		The score in the DETECTION result file should be set to the constant -1. The score in the GROUNDTRUTH file indicates 
#			the fraction of objects being occluded (i.e., no occlusion = 0 (occlusion ratio 0%), partial occlusion = 1(occlusion ratio 1% ∼ 50%), 
#			and heavy occlusion = 2 (occlusion ratio 50% ~ 100%)).	


To convert YOLO
#Each text file will contain the center coordinates, height and width of the box, and the class of the object. 
#An image can have more than 1 object, so its text file will have multiple lines, one for each object.
#Each line of the text file will have below format(YOLO):
#<object-class> <x_center> <y_center> <width> <height>
