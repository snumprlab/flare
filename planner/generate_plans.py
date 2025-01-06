import os
import openai
import json
import random
import copy
from tqdm import tqdm
import time
from collections import defaultdict
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

random.seed(2)    
openai.api_key = YOUR_OPENAI_KEY
bias = json.load(open('bias_gpt4.json'))
failed = []
ALFRED_ROOT = os.environ['ALFRED_ROOT']

def load_task_json(task):
    '''
    load traj json from disk
    '''
    json_path = os.path.join(f'{ALFRED_ROOT}/data/json_2.1.0', task['task'], 'pp', 'ann_%d.json' % task['repeat_idx'])

    with open(json_path) as f:
        data = json.load(f)
        
    return data

def main(sp, destination ='few_examples_from_song'):
    cnt = 0
    retrived = json.load(open(f'few_examples_from_song/few-song-{sp}_retrieved_keys_clip_Img1_Txt1_panoramic.json'))
    few_examples = json.load(open('few_examples_from_song/few_examples.json'))
    splits = json.load(open(f'{ALFRED_ROOT}/data/splits/oct24.json'))
    result = defaultdict(dict)
    for  task in tqdm(splits[sp]):
        cnt +=1
        if cnt > 100:
            time.sleep(30)
            ## to avoud OpenAI limts.
            cnt=0
        inst = []
        data = load_task_json(task)
        r_idx = task['repeat_idx']
        task_id = task['task']
        instruction = data['turk_annotations']['anns'][r_idx]['task_desc']
        

        for desc in data['turk_annotations']['anns'][r_idx]['high_descs']:
            instruction+= desc

        goal = ''
        for i in range(len(data['ann']['goal'])-1):
            goal += data['ann']['goal'][i].strip() + ' '
        inst_list = [i for sub_list in data['ann']['instr'] for i in sub_list]
        high_descs = ''  
        for i in range(len(inst_list)-1):
            high_descs += inst_list[i].strip() + ' '

        result[instruction]['root'] = os.path.join('data/json_feat_2.1.0', task['task'])
        result[instruction]['triplet'] = []
        result[instruction]['low_actions'] = []
        result[instruction]['low_classes'] = []
        result[instruction]['high_idxs'] = []
        
        keys = retrived[task_id][str(r_idx)]
        for k in keys:
            inst.append(few_examples[k])
        
        text = f'''Create a high-level plan for completing a household task using the allowed actions and objects.
Allowed actions: ToggleObject, CleanObject, HeatObject, PickupObject, SliceObject, CoolObject, PutObject
Allowed objects: AlarmClock, Apple, AppleSliced, ArmChair, BaseballBat, BasketBall, Bathtub, Bed, Book, Bowl, Box, Bread, BreadSliced, ButterKnife, CD, Cabinet, Candle, Cart, CellPhone, Cloth, CoffeeMachine, CoffeeTable, CounterTop, CreditCard, Cup, Desk, DeskLamp, DiningTable, DishSponge, Drawer, Dresser, Egg, FloorLamp, Fork, Fridge, GarbageCan, Glassbottle, HandTowel, Kettle, KeyChain, Knife, Ladle, Laptop, Lettuce, LettuceSliced, Microwave, Mug, Newspaper, Ottoman, Pan, Pen, Pencil, PepperShaker, Pillow, Plate, Plunger, Pot, Potato, PotatoSliced, RemoteControl, Safe, SaltShaker, Shelf, SideTable, Sink, SoapBar, SoapBottle, Sofa, Spatula, Spoon, SprayBottle, Statue, StoveBurner, TennisRacket, TissueBox, Toilet, ToiletPaper, ToiletPaperHanger, Tomato, TomatoSliced, Vase, Watch, WateringCan, WineBottle'''
        
        # retrived in-context examples are here
        for k in range(9):
            text +=f'''Task description: {inst[0]['goal']}
Step-by-step instructions: {inst[0]['instruction'][:-1]}
Next plan: {inst[0]['pddl'][:-2]}'''

            text += '''Task description: {goal[:-1]}
Step-by-step instructions: {high_descs[:-1]}
Next plan: '''
        try:
            response = openai.ChatCompletion.create(
                model = "gpt-4-0125-preview",
                messages = [{"role": "user",
            "content": text}],
                temperature = 0,
                logit_bias = bias,
                max_tokens = 90,
                stop='\n'
                )
            result[instruction]['triplet'].append(response.choices[0]['message']['content'])
            
            with open(f'planner_results/{destination}/turbo-bias-{sp}_result.json', 'w') as f:
                json.dump(result, f, indent= 4)
            
                
        except Exception as e:
            print(e)
            print(instruction)
            failed.append(instruction)
            with open(f'{sp}failed.json', 'w') as f:
                json.dump(failed, f, indent= 4)

if __name__ == "__main__":
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)

    # settings
    parser.add_argument('--dn', help='desination', default='all_examples', type=str)
    args = parser.parse_args()

    for split in ['tests_seen', 'tests_unseen', 'valid_seen', 'valid_unseen']:
        main(split, args.dn)
