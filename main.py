import logging
import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import yaml
import os

from tools import plot_image_with_bboxes_and_points
from utils_model import get_text_from_img, get_mask, fuse_mask, DotDict
from utils_model import printd, mkdir


## configs
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config/COD10K_LLaVA1.5.yaml')
    # parser.add_argument('--visualization', default='False')
    parser.add_argument('--save_path', default='./res/prediction_RDVP_MSD/')

args = parser.parse_args()

## save dir
config_name = args.config.split("/")[-1][:-5]
save_path_pred_dir = f'{args.save_path}/{config_name}/'
mkdir(save_path_pred_dir)
visualize_save_path = f'{args.save_path}/visualize/{config_name}/'
mkdir(visualize_save_path)

# logging
logging.basicConfig(filename=args.save_path + 'log.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]',
                    level=logging.INFO, filemode='a', datefmt='%Y-%m-%d %I:%M:%S %p')

device = "cuda" if torch.cuda.is_available() else "cpu"

with open(args.config, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

filenames = sorted(os.listdir(config['test_dataset']['root_path_images']))
data_len = len(filenames)
printd(f"dataset size:\t {len(filenames)}")

## load pretrained model  
# CLIP surgery, SAM
# from segment_anything import sam_model_registry, SamPredictor
# sam = sam_model_registry[config['sam_model_type']](checkpoint=config['sam_checkpoint'])
# sam.to(device=device)
# sam_predictor = SamPredictor(sam)
from segment_anything_hq import sam_model_registry, SamPredictor
import clip
sam = sam_model_registry[config['sam_model_type']](checkpoint=config['sam_checkpoint'])
sam.to(device=device)
sam_predictor = SamPredictor(sam)
clip_params={ 'attn_qkv_strategy':config['clip_attn_qkv_strategy']}
clip_model, _ = clip.load(config['clip_model'], device=device, params=clip_params)
clip_model.eval()
llm_dict=None
if config['llm']=='blip':
    from lavis.models import load_model_and_preprocess
    # blip_model_type="pretrain_opt2.7b"
    blip_model_type="pretrain_opt6.7b" 
    printd(f'loading BLIP ({blip_model_type})...')
    BLIP_model, BLIP_vis_processors, _ = load_model_and_preprocess(name="blip2_opt", 
                                                                   model_type=blip_model_type, 
                                                                   is_eval=True, 
                                                                   device=device)
    BLIP_dict = {"demo_data/9.jpg": 'lizard in the middle',}
    llm_dict = {
        'model': BLIP_model,
        'vis_processors':  BLIP_vis_processors,
    }
elif config['llm']=='LLaVA' or config['llm']=='LLaVA1.5':
    from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import get_model_name_from_path
    disable_torch_init()
    if config['llm']=='LLaVA':
        model_path = 'liuhaotian/llava-llama-2-13b-chat-lightning-preview/'
    else:
        model_path = 'liuhaotian/llava-v1.5-13b'
    print(f'llava pretrained model: {model_path}')
    model_path = os.path.expanduser(model_path)
    config['model_name'] = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_path, config['model_base'], config['model_name'])
    if 'llama-2' in config['model_name'].lower(): # from clip.py
        conv_mode = "llava_llama_2"
    elif "v1" in config['model_name'].lower():
        conv_mode = "llava_v1"
    elif "mpt" in config['model_name'].lower():
        conv_mode = "mpt"
    else:
        conv_mode = "llava_v0"
    config['conv_mode'] = conv_mode
    llm_dict = {
        'model': model,
        'vis_processors':  image_processor,
        'tokenizer': tokenizer,
        'conv_mode': config['conv_mode'],
        'temperature': config['temperature'],
        'w_caption': config['LLaVA_w_caption'],
    }
else:
    exit(f'unknow LLM: {config["llm"]}')

## run model
printd('Start inference...')
for s_i, img_name in zip(range(data_len), filenames):
    img_path = config['test_dataset']['root_path_images'] + img_name
    printd(img_path)
    logging.info(img_path)

    # img_path = './test_my/images/camourflage_01194.jpg'
    pil_img = Image.open(img_path).convert("RGB")

    ## infer RDVP-MSD
    textfg_phrase_list, textbg_phrase_list,textfg_word_list, textbg_word_list,bbox_ori_list = get_text_from_img(pil_img, config['prompt_q'], llm_dict,
                                      config['use_gene_prompt'], config['clip_use_bg_text'], DotDict(config))
    print(f'iter 0 text:\t{textfg_phrase_list}, {textbg_phrase_list}')
    logging.info(f'iter 0 text:\t{textfg_phrase_list}, {textbg_phrase_list}')
    print(f'iter 0 text:\t{textfg_word_list}, {textbg_word_list}')
    logging.info(f'iter 0 text:\t{textfg_word_list}, {textbg_word_list}')
    print(f'iter 0 text:\t{bbox_ori_list}')
    logging.info(f'iter 0 text:\t{bbox_ori_list}')
    for i in range(len(bbox_ori_list)):\
        plot_image_with_bboxes_and_points(np.array(pil_img), visualize_save_path + img_path.split("/")[-1][:-4] + '_bbox' + str(i) + '.jpg', bbox=bbox_ori_list[i], points=None, labels=None, word_fg=[textfg_word_list[i],textfg_phrase_list[i]], word_bg=[textbg_word_list[i],textbg_phrase_list[i]], mask=None)

    mask_l, mask_logit_origin_l = get_mask(pil_img,
                                           textfg_phrase_list,
                                           textfg_word_list,
                                           sam_predictor,
                                           clip_model,
                                           DotDict(config),
                                           logging,
                                           visualize_save_path,
                                           img_path,
                                           llm_dict=llm_dict,
                                           textbg_phrase_list=textbg_phrase_list,
                                           textbg_word_list=textbg_word_list,
                                           bbox_ori_list=bbox_ori_list,)

    vis_mask_acc, vis_mask_logit_acc = fuse_mask(mask_logit_origin_l, 
                                                 sam_predictor.model.mask_threshold)


    # get metric of mask closest to fused mask 
    mask_delta_l = [np.sum((mask_i - vis_mask_acc)**2) for mask_i in mask_l]  # distance of each mask to fused one
    idxMaskSim = np.argmin(mask_delta_l)
    vis_tensor = mask_l[idxMaskSim].astype('uint8') * 255


    cv2.imwrite(save_path_pred_dir+ img_path.split("/")[-1][:-4] +'.png', vis_tensor)

