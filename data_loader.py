import nltk
import os
import torch
import torch.utils.data as data
from vocabulary import Vocabulary
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
from tqdm import tqdm
import random
import json

class CoCoDataset(data.Dataset):
    def __init__(self, transform, mode, batch_size, vocab_threshold, vocab_file, start_word, end_word, unk_word, annotations_file, vocab_from_file, img_folder, subset_size=None):
        self.transform = transform
        self.mode = mode
        self.batch_size = batch_size
        self.vocab = Vocabulary(vocab_threshold, vocab_file, start_word,
                                end_word, unk_word, annotations_file, vocab_from_file)
        self.img_folder = img_folder

        if self.mode == 'train':
            self.coco = COCO(annotations_file)
            self.ids = list(self.coco.anns.keys())

            # Randomly select a subset if subset_size is specified
            if subset_size is not None and subset_size < len(self.ids):
                self.ids = random.sample(self.ids, subset_size)

            print('Obtaining caption lengths for the subset...')
            all_tokens = [nltk.tokenize.word_tokenize(str(self.coco.anns[id]['caption']).lower()) for id in tqdm(self.ids)]
            self.caption_lengths = [len(token) for token in all_tokens]
        else:
            test_info = json.loads(open(annotations_file).read())
            self.paths = [item['file_name'] for item in test_info['images']]
        
    def __getitem__(self, index):
        # obtain image and caption if in training mode
        if self.mode == 'train':
            ann_id = self.ids[index]
            caption = self.coco.anns[ann_id]['caption']
            img_id = self.coco.anns[ann_id]['image_id']
            path = self.coco.loadImgs(img_id)[0]['file_name']

            # Convert image to tensor and pre-process using transform
            image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            image = self.transform(image)

            # Convert caption to tensor of word ids.
            tokens = nltk.tokenize.word_tokenize(str(caption).lower())
            caption = []
            caption.append(self.vocab(self.vocab.start_word))
            caption.extend([self.vocab(token) for token in tokens])
            caption.append(self.vocab(self.vocab.end_word))
            caption = torch.Tensor(caption).long()

            # return pre-processed image and caption tensors
            return image, caption

        # obtain image if in test mode
        else:
            path = self.paths[index]

            # Convert image to tensor and pre-process using transform
            PIL_image = Image.open(os.path.join(self.img_folder, path)).convert('RGB')
            orig_image = np.array(PIL_image)
            image = self.transform(PIL_image)

            # return original image and pre-processed image tensor
            return orig_image, image

    def get_train_indices(self):
        sel_length = np.random.choice(self.caption_lengths)
        all_indices = [i for i, c in enumerate(self.caption_lengths) if c == sel_length]
        indices = list(np.random.choice(all_indices, size=self.batch_size))
        return indices

    def __len__(self):
        if self.mode == 'train':
            return len(self.ids)
        else:
            return len(self.paths)
        
def get_loader(transform, mode='train', 
               batch_size=1, 
               vocab_threshold=None, 
               vocab_file='./vocab.pkl', 
               start_word="<start>", 
               end_word="<end>", 
               unk_word="<unk>", 
               vocab_from_file=True, 
               num_workers=0, 
               cocoapi_loc='/opt', 
               subset_size=None):
    
    assert mode in ['train', 'test'], "mode must be one of 'train' or 'test'."

    if mode == 'train':
        if vocab_from_file: 
            assert os.path.exists(vocab_file), "vocab_file does not exist. Change vocab_from_file to False to create vocab_file."
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/train2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/captions_train2014.json')
    elif mode == 'test':
        assert batch_size == 1, "Please change batch_size to 1 if testing your model."
        assert os.path.exists(vocab_file), "Must first generate vocab.pkl from training data."
        img_folder = os.path.join(cocoapi_loc, 'cocoapi/images/test2014/')
        annotations_file = os.path.join(cocoapi_loc, 'cocoapi/annotations/image_info_test2014.json')

    dataset = CoCoDataset(transform=transform, 
                          mode=mode, 
                          batch_size=batch_size, 
                          vocab_threshold=vocab_threshold, 
                          vocab_file=vocab_file, 
                          start_word=start_word, 
                          end_word=end_word, 
                          unk_word=unk_word, 
                          annotations_file=annotations_file, 
                          vocab_from_file=vocab_from_file, 
                          img_folder=img_folder, 
                          subset_size=subset_size)

    if mode == 'train':
        indices = dataset.get_train_indices()
        initial_sampler = data.sampler.SubsetRandomSampler(indices=indices)
        data_loader = data.DataLoader(dataset=dataset, 
                                      num_workers=num_workers,
                                      batch_sampler=data.sampler.BatchSampler(sampler=initial_sampler,
                                                                              batch_size=dataset.batch_size,
                                                                              drop_last=False))
    else:
        data_loader = data.DataLoader(dataset=dataset, 
                                      batch_size=dataset.batch_size,
                                      shuffle=True, 
                                      num_workers=num_workers)

    return data_loader