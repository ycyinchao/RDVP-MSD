import clip
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, InterpolationMode
import torch.nn.functional as F
import datetime
import os
import re

from tools import plot_image_with_bboxes_and_points

BICUBIC = InterpolationMode.BICUBIC
eps = 1e-7


def fuse_mask(mask_logit_origin_l, sam_thr, fuse='avg'):

    num_mask = len(mask_logit_origin_l)
    if fuse=='avg':
        mask_logit_origin = sum(mask_logit_origin_l)/num_mask  #
        mask_logit = F.sigmoid(torch.from_numpy(mask_logit_origin)).numpy()
        mask = mask_logit_origin > sam_thr

    mask = mask.astype('uint8')
    mask_logit *= 255
    mask_logit = mask_logit.astype('uint8')

    return mask, mask_logit


def get_mask(pil_img, textfg_phrase_list,textfg_word_list, sam_predictor, clip_model, args,logging,visualize_save_path,img_path, llm_dict=None, textbg_phrase_list=None,textbg_word_list=None,bbox_ori_list=None,):

    mask_l = []
    mask_logit_origin_l = []

    ori_image = np.array(pil_img)
    sam_predictor.set_image(ori_image)

    cur_image = ori_image
    with torch.no_grad():
        for i in range(args.recursive+1):
            sm_norm_fg_phrase, sm_norm_bg_phrase = clip_surgery(cur_image,
                                                  textfg_phrase_list,
                                                  clip_model,
                                                  args, device='cuda',
                                                  text_bg=textbg_phrase_list)

            sm_norm_fg_word, sm_norm_bg_word = clip_surgery(cur_image,
                                                  textfg_word_list,
                                                  clip_model,
                                                  args, device='cuda',
                                                  text_bg=textbg_word_list)

            for j in range(sm_norm_fg_phrase.shape[-1]):
                bboxes = np.array(bbox_ori_list[j])
                for n in range(2):
                    if n==0:
                        sm_norm_fg_n = sm_norm_fg_phrase[..., j]
                        sm_norm_bg_n = sm_norm_bg_phrase[..., j]
                    else:
                        sm_norm_fg_n = sm_norm_fg_word[..., j]
                        sm_norm_bg_n = sm_norm_bg_word[..., j]



                    p_fg_fg,l_fg_fg,p_fg_bg, l_fg_bg = clip.similarity_map_to_points(sm_norm_fg_n, cur_image.shape[:2],
                                                                          t=args.attn_thr,
                                                                          down_sample=args.down_sample,
                                                                          is_fg=True,
                                                                          bbox = bboxes)  # p: [pos (min->max), neg(max->min)]
                    p_bg_fg,l_bg_fg,p_bg_bg, l_bg_bg = clip.similarity_map_to_points(sm_norm_bg_n, cur_image.shape[:2],
                                                                          t=args.attn_thr,
                                                                          down_sample=args.down_sample,
                                                                          is_fg=False,
                                                                          bbox = bboxes)  # p: [pos (min->max), neg(max->min)]
                    points = p_fg_fg + p_bg_bg
                    labels = l_fg_fg.tolist() + l_bg_bg.tolist()
                    x_min, y_min, x_max, y_max = bboxes
                    if (x_max - x_min) * (y_max - y_min) < 100:
                        bboxes = np.array([0, 0, cur_image.shape[1], cur_image.shape[0]])

                    side = int(sm_norm_fg_n.shape[0] ** 0.5)
                    sm_norm_fg_n = sm_norm_fg_n.reshape(1, 1, side, side)
                    map_fg = torch.nn.functional.interpolate(sm_norm_fg_n, (ori_image.shape[0], ori_image.shape[1]), mode='bilinear').squeeze()
                    plot_image_with_bboxes_and_points(cur_image,bbox=bboxes, points=p_fg_fg, labels=l_fg_fg.tolist(), mask=map_fg.cpu().numpy(),save_path=visualize_save_path + img_path.split("/")[-1][:-4] + '_mapFG_' + str(j)+str(n) + '.jpg',show=True)
                    sm_norm_bg_n = sm_norm_bg_n.reshape(1, 1, side, side)
                    map_bg = torch.nn.functional.interpolate(sm_norm_bg_n, (ori_image.shape[0], ori_image.shape[1]), mode='bilinear').squeeze()
                    plot_image_with_bboxes_and_points(cur_image,bbox=bboxes, points=p_bg_bg, labels=l_bg_bg.tolist(), mask=map_bg.cpu().numpy(),save_path=visualize_save_path + img_path.split("/")[-1][:-4] + '_mapBG_' + str(j)+str(n) + '.jpg',show=True)

                    # Inference SAM with points from CLIP Surgery
                    if args.post_mode =='MaxIOUBoxSAMInput':

                        mask_logit_origin, scores, logits = sam_predictor.predict(point_labels=labels,
                                                                                  point_coords=np.array(points),
                                                                                  box=bboxes[None, :],
                                                                                  multimask_output=True,
                                                                                  return_logits=True)
                        # mask = mask_logit_origin[np.argmax(scores)] > sam_predictor.model.mask_threshold
                        mask_logit_origin = mask_logit_origin[np.argmax(scores)]
                        mask = mask_logit_origin > sam_predictor.model.mask_threshold



                        contours, _ = cv2.findContours(mask.copy().astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # cv2.RETR_EXTERNAL：只检测最外层的轮廓;

                        if len(contours)==0:
                            x_min = 0
                            x_max = mask_logit_origin.shape[1]
                            y_min = 0
                            y_max = mask_logit_origin.shape[0]
                            bboxes = np.array([x_min, y_min, x_max, y_max])
                        else:
                            overlaps = []
                            bboxes_list = []
                            for contour in contours:
                                x, y, w, h = cv2.boundingRect(contour)
                                bbox = np.array([x, y, x + w, y + h])
                                bboxes_list.append(bbox)
                                overlap = (w * h) / np.sum(mask)
                                overlaps.append(overlap)
                            bboxes_list = np.array(bboxes_list)
                            overlaps = np.array(overlaps)
                            max_overlap_idx = np.argmax(overlaps)
                            max_bbox = bboxes_list[max_overlap_idx]
                            scaled_bbox = max_bbox.copy()
                            scaled_bbox[:2] -= np.floor((scaled_bbox[2:] - scaled_bbox[:2]) * 0.1).astype(int) # 边界框扩展：通过缩小边界框的起点坐标 scaled_bbox[:2] 和增大终点坐标 scaled_bbox[2:] 来实现。
                            scaled_bbox[2:] += np.ceil((scaled_bbox[2:] - scaled_bbox[:2]) * 0.1).astype(int)
                            bboxes_list[max_overlap_idx] = scaled_bbox
                            bboxes = bboxes_list[max_overlap_idx]
                            # 确保边界框在图像内
                            bboxes[0] = max(0, bboxes[0])
                            bboxes[1] = max(0, bboxes[1])
                            bboxes[2] = min(mask_logit_origin.shape[1], bboxes[2])
                            bboxes[3] = min(mask_logit_origin.shape[0], bboxes[3])

                        if n==0:
                            plot_image_with_bboxes_and_points(cur_image, bboxes, points, labels
                                                          , textfg_phrase_list[j], textbg_phrase_list[j], mask=mask,save_path=visualize_save_path + img_path.split("/")[-1][:-4] + '_' + str(
                                                              j)+str(n) + '.jpg',show=True)
                        else:
                            plot_image_with_bboxes_and_points(cur_image, bboxes, points, labels
                                                          , textfg_word_list[j], textbg_word_list[j], mask=mask,save_path=visualize_save_path + img_path.split("/")[-1][:-4] + '_' + str(
                                                              j)+str(n) + '.jpg',show=True)


                    mask_l.append(mask)
                    mask_logit_origin_l.append(mask_logit_origin)


    return mask_l, mask_logit_origin_l


def clip_surgery(np_img, text, model, args, device='cuda', text_bg=None):

    pil_img = Image.fromarray(np_img.astype(np.uint8))
    h, w = np_img.shape[:2]
    preprocess =  Compose([Resize((336, 336), interpolation=BICUBIC), ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))])
    image = preprocess(pil_img).unsqueeze(0).to(device)

    # CLIP architecture surgery acts on the image encoder
    image_features = model.encode_image(image)
    image_features = image_features / image_features.norm(dim=1, keepdim=True)    # torch.Size([1, 197, 512])

    # Extract redundant features from an empty string
    redundant_features = clip.encode_text_with_prompt_ensemble(model, [args.rdd_str], device)  # torch.Size([1, 512])

    # Prompt ensemble for text features with normalization
    text_features = clip.encode_text_with_prompt_ensemble(model, text, device)  # torch.Size([x, 512])
    if args.clip_use_bg_text:
        text_bg_features = clip.encode_text_with_prompt_ensemble(model, text_bg, device)  # torch.Size([x, 512])


    def _norm_sm(_sm, h, w):
        side = int(_sm.shape[0] ** 0.5)
        _sm = _sm.reshape(1, 1, side, side)
        _sm = torch.nn.functional.interpolate(_sm, (h, w), mode='bilinear')[0, 0, :, :].unsqueeze(-1)
        _sm = (_sm - _sm.min()) / (_sm.max() - _sm.min())
        _sm = _sm.cpu().numpy()
        return _sm

    # Combine features after removing redundant features and min-max norm
    sm_fg = clip.clip_feature_surgery(image_features, text_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # sm指的是similarity
    sm_norm_fg = (sm_fg - sm_fg.min(0, keepdim=True)[0]) / (sm_fg.max(0, keepdim=True)[0] - sm_fg.min(0, keepdim=True)[0])
    # sm_mean_fg = sm_norm_fg.mean(-1, keepdim=True)

    sm_mean_bg, sm_mean_fg_bg=None, None
    if args.clip_use_bg_text:
        sm_bg = clip.clip_feature_surgery(image_features, text_bg_features, redundant_features)[0, 1:, :]  # 整个输出：torch.Size([1, 197, x])  # 最后的1，是text这个list 的长度。
        sm_norm_bg = (sm_bg - sm_bg.min(0, keepdim=True)[0]) / (sm_bg.max(0, keepdim=True)[0] - sm_bg.min(0, keepdim=True)[0])
        # sm_mean_bg = sm_norm_bg.mean(-1, keepdim=True)

        # if args.clip_bg_strategy=='FgBgHm':
        #     sm_mean_fg_bg = sm_mean - sm_mean_bg
        # else: # FgBgHmClamp
        #     sm_mean_fg_bg = torch.clamp(sm_mean - sm_mean_bg, 0, 1)

        # sm_mean_fg_bg = (sm_mean_fg_bg - sm_mean_fg_bg.min(0, keepdim=True)[0]) / (sm_mean_fg_bg.max(0, keepdim=True)[0] - sm_mean_fg_bg.min(0, keepdim=True)[0])
        # sm_mean_fg_bg_origin = sm_mean_fg_bg
        # sm_mean = sm_mean_fg_bg_origin

    # expand similarity map to original image size, normalize. to apply to image for next iter

    return sm_norm_fg, sm_norm_bg


template_fg_phrase_q='Provide a concise and comprehensive descriptive compound noun phrase for {}, without any environmental description, using only noun phrases and prepositional phrases, emphasizing its shape, color, and texture.'
template_bg_phrase_q='Provide a concise and comprehensive descriptive compound noun phrase for the environment that is highly similar to {}, using only noun phrases and prepositional phrases.'
template_fg_word_q='Name of the {} in one word.'
template_bg_word_q='Name of the environment of the {} in one word.'
prompt_qkeys_dict={

    'TheCamo':          ['camouflaged animal'],
    'TheShadow':        ['shadow'],
    'TheGlass':         ['glass'],
    'ThePolyp':         ['polyp'],

    '3attriTheBgSyn':   ['concealed animal', 'hidden animal', 'unseen animal'],
    '3attriTheBgSynCamo':   ['camouflaged object', 'camouflaged animal', 'camouflaged entity'],
    # '3attriTheBgSynCamo':   ['camouflaged animal'],
    '3attriTheBgSynCamoSpec':   ['camouflaged species', 'disguised species', 'hidden species'],

    '3TheGlassSyn':     ['glass', 'window', 'mirror'],
    '3TheGlassSyn1':     ['glass', 'window', 'transparent material'],

    '3TheShadowSyn':    ['shadow', 'silhouette', 'profile'],
    '3TheShadowSyn1':    ['shadow', 'silhouette', 'outline'],

    '3ThePolypSyn':     ['polyp', 'appendage', 'tentacle'],
    '3ThePolypSyn1':    ['polyp', 'appendage', 'tumor'],
    '3ThePolypSyn2':    ['polyp', 'tumor', 'growth'],

    '1attriTheCamouflageBg_test': ['camouflaged animal'],
    '3attriTheBgSynCamo_test':   ['camouflaged animal', 'disguised animal', 'hidden animal'],

}
prompt_q_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_q_dict.get(k) is None:
        prompt_q_dict[k] = [[template_fg_phrase_q.format(key), template_bg_phrase_q.format(key), template_fg_word_q.format(key),template_bg_word_q.format(key)] for key in prompt_qkeys_dict[k]]
prompt_gene_dict={}
for k, v in prompt_qkeys_dict.items():
    if prompt_gene_dict.get(k) is None:
        prompt_gene_dict[k] = [prompt_qkeys_dict[k], ['environment']]


def get_text_from_img(pil_img, prompt_q, llm_dict, use_gene_prompt, get_bg_text, args,
                        reset_prompt_qkeys=False, new_prompt_qkeys_l=None,
                        bg_cat_list=[],
                        post_process_per_cat_fg=False):
    if use_gene_prompt:
        return prompt_gene_dict[args.prompt_q]
    else:  # use LLM model: BLIP2; LLaVA
        model = llm_dict['model']
        vis_processors = llm_dict['vis_processors']
        use_gene_prompt_fg=args.use_gene_prompt_fg
        if args.llm=='blip':
            return get_text_from_img_blip(pil_img, prompt_q,
                        model, vis_processors,
                        get_bg_text=get_bg_text,)
        elif args.llm=='LLaVA' or args.llm=='LLaVA1.5':
            tokenizer = llm_dict['tokenizer']
            conv_mode = llm_dict['conv_mode']
            temperature = llm_dict['temperature']
            w_caption = llm_dict['w_caption']
            if args.check_exist_each_iter: # only for multiple classes
                if not cat_exist(
                    pil_img, new_prompt_qkeys_l[0],
                    model, vis_processors, tokenizer,
                    ):
                    return [], []

            return get_text_from_img_llava(pil_img, prompt_q,
                        model, vis_processors, tokenizer,
                        get_bg_text=get_bg_text,
                        conv_mode=conv_mode,
                        temperature=temperature,
                        w_caption=w_caption,
                        )


def get_text_from_img_blip(pil_img, prompt_q=None, model=None, vis_processors=None, get_bg_text=False, device='cuda', ):

    image = vis_processors["eval"](pil_img).unsqueeze(0).to(device)
    blip_output = model.generate({"image": image})
    blip_output = blip_output[0].split('-')[0]
    context = [
        ("Image caption",blip_output),
    ]
    template = "Question: {}. Answer: {}."

    question_l = ["Name of hidden animal in one word."] if prompt_q is None else prompt_q_dict[prompt_q]
    text_list = []
    textbg_list = []
    for question in question_l:
        out_list = []
        prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question[0] + " Answer:"
        blip_output_forsecond = model.generate({"image": image, "prompt": prompt})
        blip_output_forsecond = blip_output_forsecond[0].split('-')[0].replace('\'','')
        if len(blip_output_forsecond)==0:    continue
        out_list.append(blip_output_forsecond)
        out_list = " ".join(out_list)
        text_list.append(out_list)

        if get_bg_text:
            ## get background text
            outbg_list = []
            prompt = " ".join([template.format(context[i][0], context[i][1]) for i in range(len(context))]) + " Question: " + question[0] + " Answer:" + blip_output_forsecond + ". Question: " + question[1] + " Answer:"
            blip_output_forsecond = model.generate({"image": image, "prompt": prompt})
            blip_output_forsecond = blip_output_forsecond[0].split('-')[0].replace('\'','')
            print(prompt)
            print(blip_output_forsecond)
            if 'Question' in blip_output_forsecond:
                blip_output_forsecond = blip_output_forsecond.split('Question')[0]
            blip_output_forsecond = blip_output_forsecond.split('.')[0]
                # while blip_output_forsecond[-1]==' ':
                #     blip_output_forsecond = blip_output_forsecond[:-1]
            if len(blip_output_forsecond)==0:     continue
            outbg_list.append(blip_output_forsecond)
            outbg_list = " ".join(outbg_list)

            textbg_list.append(outbg_list)

    print(f'caption: {blip_output}')
    text = text_list
    text_bg = textbg_list

    # deal with empty text
    if len(text)==0:
        text = prompt_gene_dict[prompt_q][0]
    if get_bg_text:
        def _same(l1, l2):
            l1_ = [i1.replace(' ','') for i1 in l1]
            l2_ = [i2.replace(' ','') for i2 in l2]
            return set(l1_)==set(l2_)
        if _same(text, text_bg):    text_bg=[]
        if len(text_bg)==0:
            text_bg = prompt_gene_dict[prompt_q][1]

    print(text, text_bg)
    return text, text_bg


def get_text_from_img_llava(
    pil_img, prompt_q,
    model, image_processor, tokenizer,
    get_bg_text=False,
    conv_mode='llava_v0',
    temperature=0.2,
    w_caption=False):
    '''
    input
    '''
    from transformers import TextStreamer
    from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
    from llava.conversation import conv_templates, SeparatorStyle
    # from llava.model.builder import load_pretrained_model
    from llava.utils import disable_torch_init
    from llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria

    prompt_qkeys_l = prompt_qkeys_dict[prompt_q]
    question_l = prompt_q_dict[prompt_q]

    textfg_phrase_list = []
    textbg_phrase_list = []
    textfg_word_list = []
    textbg_word_list = []
    bbox_ori_list = []

    image = pil_img #load_image(img_path)
    image_width, image_height = image.size
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()

    disable_torch_init()
    for qi, qs in enumerate(question_l):

        if w_caption:
            q_keyword = prompt_qkeys_l[qi]
            bbox_naive = [0, 0, 1, 1]
            caption_q = f'This image is from {q_keyword} detection task, describe the {q_keyword} in one sentence'
            bbox_q = f' The naive bounding box of {q_keyword} is {bbox_naive}, adjust the bounding box to ensure that all {q_keyword} are fully displayed. Just output the adjusted bounding box.'
            qs=[caption_q] + qs + [bbox_q]

        image = pil_img #load_image(img_path)
        conv = conv_templates[conv_mode].copy() # 是否需要改一下system 提示词，换成caption？

        for i, inp in enumerate(qs):

            if image is not None:
                # first message
                if model.config.mm_use_im_start_end:
                    inp = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + inp
                else:
                    inp = DEFAULT_IMAGE_TOKEN + '\n' + inp
                conv.append_message(conv.roles[0], inp)
                image = None
            else:
                # later messages
                conv.append_message(conv.roles[0], inp)
            conv.append_message(conv.roles[1], None)
            prompt = conv.get_prompt()

            input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
            stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
            keywords = [stop_str]
            stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
            streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=1024,
                    streamer=streamer,
                    use_cache=True,
                    stopping_criteria=[stopping_criteria])

            outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip().lower()
            conv.messages[-1][-1] = outputs

            if w_caption and i == 0:
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
            if w_caption and i in [1,3]:
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs.find('.'):
                    outputs = outputs.split('.')[0]

                if outputs.find(' is '):
                    outputs = outputs.split(' is ')[0]
                if outputs.find(' are '):
                    outputs = outputs.split(' are ')[0]
                if outputs.find('that'):
                    outputs = outputs.split('that')[0]
                if outputs.find(' in which '):
                    outputs = outputs.split(' in which ')[0]
                if outputs.find(' which '):
                    outputs = outputs.split(' which ')[0]
                if outputs[-1] == ',':
                    outputs = outputs[:-1]


                while outputs[0] == ' ':  outputs = outputs[1:]
                while outputs[-1] == ' ':  outputs = outputs[:-1]



                if i == 1:
                    textfg_phrase_list.append(outputs)
                else:
                    textfg_word_list.append(outputs)

            if w_caption and i in [2,4]:
                if outputs.find('"') > 0:
                    outputs = outputs.split('"')[1]
                elif outputs.find(' is an ') > 0:
                    outputs = outputs.split(' is an ')[1]
                elif outputs.find(' is a ') > 0:
                    outputs = outputs.split(' is a ')[1]
                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                if outputs.find('.'):
                    outputs = outputs.split('.')[0]

                if outputs.find(' is '):
                    outputs = outputs.split(' is ')[0]
                if outputs.find(' are '):
                    outputs = outputs.split(' are ')[0]
                if outputs.find('that'):
                    outputs = outputs.split('that')[0]
                if outputs.find(' in which '):
                    outputs = outputs.split(' in which ')[0]
                if outputs.find(' which '):
                    outputs = outputs.split(' which ')[0]
                if outputs[-1] == ',':
                    outputs = outputs[:-1]

                while outputs[0] == ' ':  outputs = outputs[1:]
                while outputs[-1] == ' ':  outputs = outputs[:-1]

                if i == 2:
                    if outputs == textfg_phrase_list[-1]:
                        textbg_phrase_list.append('background')
                    else:
                        textbg_phrase_list.append(outputs)
                else:
                    if outputs == textfg_word_list[-1]:
                        textbg_word_list.append('background')
                    else:
                        textbg_word_list.append(outputs)
            if w_caption and i == 5:

                outputs = outputs.replace(DEFAULT_IM_END_TOKEN, '')  # "<im_end>"
                outputs = outputs.replace('<|im_end|>', '')
                outputs = outputs.replace('</s>', '')
                outputs = outputs.strip('[]</s> \n')
                string_numbers = re.findall(r'\d+\.\d+', outputs)
                outputs_bbox = [round(float(num), 2) for num in string_numbers]
                outputs_bbox = expand_bbox(outputs_bbox, 0.0)
                bbox_ori = [0, 0, 1, 1]
                if not outputs_bbox or len(outputs_bbox) != 4:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = int(bbox_ori[0] * image_width), int(bbox_ori[1] * image_height), int(bbox_ori[2] * image_width), int(bbox_ori[3] * image_height)
                    bbox_ori_list.append(bbox_ori)
                elif (outputs_bbox[2] - outputs_bbox[0]) * (outputs_bbox[3] - outputs_bbox[1]) == 0:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = int(bbox_ori[0] * image_width), int(bbox_ori[1] * image_height), int(bbox_ori[2] * image_width), int(bbox_ori[3] * image_height)
                    bbox_ori_list.append(bbox_ori)
                if len(outputs_bbox) >= 4:
                    bbox_ori[0], bbox_ori[1], bbox_ori[2], bbox_ori[3] = int(outputs_bbox[0] * image_width), int(outputs_bbox[
                        1] * image_height), int(outputs_bbox[2] * image_width), int(outputs_bbox[3] * image_height)
                    bbox_ori_list.append(bbox_ori)


    # if len(textbg_phrase_list)==0:
    #     textbg_phrase_list=['background']
    # if len(textfg_phrase_list)==0:
    #     textfg_phrase_list=['hidden animal']


    return textfg_phrase_list, textbg_phrase_list,textfg_word_list, textbg_word_list,bbox_ori_list


def expand_bbox(bbox, expansion_rate=0.15):
    if not bbox or len(bbox) != 4:
        return bbox

    x1, y1, x2, y2 = bbox

    original_width = x2 - x1
    original_height = y2 - y1

    expand_width = original_width * expansion_rate
    expand_height = original_height * expansion_rate

    new_x1 = max(0, x1 - expand_width / 2)
    new_y1 = max(0, y1 - expand_height / 2)
    new_x2 = min(1, x2 + expand_width / 2)
    new_y2 = min(1, y2 + expand_height / 2)

    return [new_x1, new_y1, new_x2, new_y2]

def get_dir_from_args(args, parent_dir='output_img/'):
    text_filename = f'{args.llm}Text'
    if args.update_text:
        text_filename += 'Update'
    parent_dir += f'{text_filename}/'

    exp_name = ''
    exp_name += f's{args.down_sample}_thr{args.attn_thr}'
    if args.recursive > 0:
        exp_name += f'_rcur{args.recursive}'
        if args.recursive_coef!=.3:
            exp_name += f'_{args.recursive_coef}'
    if args.rdd_str != '':
        exp_name += f'_rdd{args.rdd_str}'
    if args.clip_attn_qkv_strategy!='vv':
        exp_name += f'_qkv{args.clip_attn_qkv_strategy}'

    if args.clipInputEMA:  # darken
        exp_name += f'_clipInputEMA'

    if args.post_mode !='':
        exp_name += f'_post{args.post_mode}'
    if args.prompt_q!='Name of hidden animal in one word':
        exp_name += f'_prompt_q{args.prompt_q}'
        if args.use_gene_prompt:
            exp_name += 'Gene'
        if args.use_gene_prompt_fg:
            exp_name += 'GeneFg'
    if args.clip_use_bg_text:
        exp_name += f'_{args.clip_bg_strategy}'

    if args.llm=='LLaVA' and args.LLaVA_w_caption:
        exp_name += f'_shortCaption'


    save_path_dir = f'{parent_dir+exp_name}/'
    printd(f'{exp_name} ({args}')

    return save_path_dir


def one_dimensional_kmeans_with_min_max(data, k, max_iterations=100):
    np.random.seed(0)
    data = np.array(data)
    initial_centers = np.random.choice(data, size=k, replace=False)
    centers = initial_centers
    min_values = np.zeros(k)
    max_values = np.zeros(k)
    for _ in range(max_iterations):
        labels = np.argmin(np.abs(data[:, np.newaxis] - centers), axis=1)
        new_centers = np.array([data[labels == i].mean() for i in range(k)])
        for i in range(k):
            cluster_data = data[labels == i]
            min_values[i] = cluster_data.min()
            max_values[i] = cluster_data.max()
        if np.all(centers == new_centers):
            break
        centers = new_centers
    min_mean_cluster_index = np.argmin(min_values)
    max_mean_cluster_index = np.argmax(max_values)
    min_mean_cluster_count = np.sum(labels == min_mean_cluster_index)
    max_mean_cluster_count = np.sum(labels == max_mean_cluster_index)
    return min_mean_cluster_count, max_mean_cluster_count


#### utility ####
class DotDict:
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def mkdir(path):
    if not os.path.isdir(path) and not os.path.exists(path):
        os.makedirs(path)

def printd(str):
    dt = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(dt+'\t '+str)

def get_edge_img_path(mask_path, img_path):
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # edges = cv2.Canny(binary_mask, threshold1=30, threshold2=100)
    # kernel = np.ones((5, 5), np.uint8)
    # thicker_edges = cv2.dilate(edges, kernel, iterations=1)
    # coord=(thicker_edges==255)
    # img[binary_mask==255] = img[binary_mask==255]*0.8 + np.array([[[0,0,51]]])
    # img[...,2][coord]=255
    # return img
    return get_edge_img(binary_mask, img)

def get_edge_img(binary_mask, img):
    # img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # binary_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)


    kernel = np.ones((5, 5), np.uint8)

    binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

    edges = cv2.Canny(binary_mask, threshold1=30, threshold2=100)
    thicker_edges = cv2.dilate(edges, kernel, iterations=1)
    coord=(thicker_edges==255)
    img[...,:][coord]=np.array([255, 200,200])
    coord_fg = (binary_mask==255)
    coord_bg = (binary_mask==0)

    r = 0.2
    img[...,0][coord_fg] = img[...,0][coord_fg] * (1-r) + 255 * r
    img[...,2][coord_bg] = img[...,2][coord_bg] * (1-r) + 255 * r
    img = np.clip(img,0,255) #.astype(np.uint8)

    return img

