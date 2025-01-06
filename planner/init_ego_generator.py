import os
import sys
sys.path.append(os.path.join(os.environ['ALFRED_ROOT']))
sys.path.append(os.path.join(os.environ['ALFRED_ROOT'], 'gen'))

import json
import glob
import os
import constants
import cv2
import shutil
import numpy as np
import argparse
import threading
import time
import copy
import random
from utils.video_util import VideoSaver
from utils.py_util import walklevel
from env.thor_env import ThorEnv

ALFRED_ROOT = os.environ['ALFRED_ROOT']


TRAJ_DATA_JSON_FILENAME = "traj_data.json"
AUGMENTED_TRAJ_DATA_JSON_FILENAME = "augmented_traj_data.json"

ORIGINAL_IMAGES_FORLDER = "raw_images"
KEY_FRAME_FOLDER = "key_frames"
DEPTH_IMAGES_FOLDER = "depth_images"
INSTANCE_MASKS_FOLDER = "instance_masks"

IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300

render_settings = dict()
render_settings['renderImage'] = True
render_settings['renderDepthImage'] = True
render_settings['renderObjectImage'] = True
render_settings['renderClassImage'] = True

video_saver = VideoSaver()


def get_image_index(save_path):
    return len(glob.glob(save_path + '/*.png'))


def save_image_with_delays(env, action,
                           save_path, direction=constants.BEFORE):
    im_ind = get_image_index(save_path)
    counts = constants.SAVE_FRAME_BEFORE_AND_AFTER_COUNTS[action['action']][direction]
    for i in range(counts):
        save_image(env.last_event, save_path)
        env.noop()
    return im_ind


def save_image(event, save_path, env):
    # rgb
    rgb_save_path = os.path.join(save_path)
    rgb_image = event.frame[:, :, ::-1]

    # Panoramic views

    imgs = {
        'rgb': {'front': rgb_image},
    }

    horizon = np.round(env.last_event.metadata['agent']['cameraHorizon'])
    rotation = env.last_event.metadata['agent']['rotation']
    position = env.last_event.metadata['agent']['position']

    # Left
    event = env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotation": (rotation['y'] + 270.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
        'standing': True
    })
    imgs['rgb']['left'] = event.frame[:, :, ::-1]

    # Right
    event = env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotation": (rotation['y'] + 90.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
        'standing': True
    })
    imgs['rgb']['right'] = event.frame[:, :, ::-1]

    # Back
    event = env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotation": (rotation['y'] + 180.0) % 360,
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
        'standing': True
    })
    imgs['rgb']['back'] = event.frame[:, :, ::-1]



    # Back to original position
    event = env.step({
        "action": "TeleportFull",
        "horizon": horizon,
        "rotation": rotation['y'],
        "x": position['x'],
        "y": position['y'],
        "z": position['z'],
        "forceAction": True,
        'standing': True
    })


    rgb_image_panoramic = np.hstack((imgs['rgb']['left'], imgs['rgb']['front'], imgs['rgb']['right'], imgs['rgb']['back']))

    # dump images
    im_ind = get_image_index(rgb_save_path)
    cv2.imwrite(rgb_save_path + '/init_ego_panoramic.png', rgb_image_panoramic)

    return im_ind
    
def save_images_in_events(events, root_dir):
    for event in events:
        save_image(event, root_dir)


def clear_and_create_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def augment_traj(env, json_file):
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    # make directories
    root_dir = json_file.replace(TRAJ_DATA_JSON_FILENAME, "")

    orig_images_dir = os.path.join(root_dir, ORIGINAL_IMAGES_FORLDER)
    key_frame_dir = os.path.join(root_dir, KEY_FRAME_FOLDER)

    # fresh images list
    traj_data['images'] = list()

    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    object_toggles = traj_data['scene']['object_toggles']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']

    # reset
    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    env.step(dict(traj_data['scene']['init_action']))
    print("Task: %s" % (traj_data['task_id']))

    # setup task
    save_image(env.last_event, root_dir, env)



def run():
    '''
    replay loop
    '''
    # start THOR env
    env = ThorEnv(player_screen_width=IMAGE_WIDTH,
                  player_screen_height=IMAGE_HEIGHT)

    skipped_files = []

    while len(traj_list) > 0:
        lock.acquire()
        json_file = traj_list.pop()
        lock.release()

        print ("Augmenting: " + json_file)
        try:
            augment_traj(env, json_file)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print ("Error: " + repr(e))
            print ("Skipping " + json_file)
            skipped_files.append(json_file)

    env.stop()
    print("Finished.")

    # skipped files
    if len(skipped_files) > 0:
        print("Skipped Files:")
        print(skipped_files)


traj_list = []
lock = threading.Lock()

# parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default=f"{ALFRED_ROOT}/data/json_2.1.0")
parser.add_argument('--smooth_nav', dest='smooth_nav', action='store_true')
parser.add_argument('--time_delays', dest='time_delays', action='store_true')
parser.add_argument('--shuffle', dest='shuffle', action='store_true')
parser.add_argument('--num_threads', type=int, default=1)
args = parser.parse_args()

# make a list of all the traj_data json files
for dir_name, subdir_list, file_list in walklevel(args.data_path, level=3):
    if "trial_" in dir_name and 'tests' not in dir_name:
        json_file = os.path.join(dir_name, TRAJ_DATA_JSON_FILENAME)
        if not os.path.isfile(json_file): # or 'tests' in dir_name:
            continue
        traj_list.append(json_file)
        # print(json_file)

# random shuffle
if args.shuffle:
    random.shuffle(traj_list)

# start threads
threads = []
for n in range(args.num_threads):
    thread = threading.Thread(target=run)
    threads.append(thread)
    thread.start()
    time.sleep(1)