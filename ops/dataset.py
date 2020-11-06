# Code for "TSM: Temporal Shift Module for Efficient Video Understanding"
# arXiv:1811.08383
# Ji Lin*, Chuang Gan, Song Han
# {jilin, songhan}@mit.edu, ganchuang@csail.mit.edu

import torch.utils.data as data

from PIL import Image
import os
import numpy as np
from numpy.random import randint


class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class TSNDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=8, new_length=[1], modality=['RGB'],
                 image_tmpl=['{:06d}.jpg'], transform=None,
                 random_shift=True, test_mode=False,
                 remove_missing=False, random_sample=False, twice_sample=False,
                 dense_length=32, dense_number=1, dense_sample=False):

        if len(modality)==2 and modality[0]==modality[1]:
            self.mml = False
            self.root_path = root_path[:1]
            self.list_file = list_file[:1]
            self.new_length = new_length[:1]
            self.modality = modality[:1]
            self.image_tmpl = image_tmpl[:1]
        else:
            self.mml = True
            self.root_path = root_path
            self.list_file = list_file
            self.new_length = new_length
            self.modality = modality
            self.image_tmpl = image_tmpl
        self.num_segments = num_segments
        self.transform = transform
        self.random_shift = random_shift
        self.test_mode = test_mode
        self.remove_missing = remove_missing
        self.dense_sample = dense_sample  # using dense sample as I3D
        self.twice_sample = twice_sample  # twice sample for more validation
        self.random_sample = random_sample
        self.dense_length = dense_length
        self.dense_number = dense_number
        if self.dense_sample:
            print('=> Using dense sample for the dataset...')
            print('=> Number of frames for run:',dense_length)
        elif self.random_sample:
            print('=> Using random sample for the dataset...')
            print('=> Number of frames for run:',dense_length)
        if self.twice_sample:
            print('=> Using twice sample for the dataset...')
        if test_mode and (self.random_sample or self.dense_sample):
            print('=> Number of runs:',dense_number)

        self._parse_list()

    def _load_image(self, directory, idx, moda, root, tmpl):
        if moda == 'RGB' or moda == 'RGBDiff':
            try:
                return [Image.open(os.path.join(root, directory, tmpl.format(idx))).convert('RGB')]
            except Exception:
                print('error loading image:', os.path.join(root, directory, tmpl.format(idx)))
                return [Image.open(os.path.join(root, directory, tmpl.format(1))).convert('RGB')]
        elif moda == 'Flow':
            if tmpl == 'flow_{}_{:05d}.jpg':  # ucf
                x_img = Image.open(os.path.join(root, directory, tmpl.format('x', idx))).convert(
                    'L')
                y_img = Image.open(os.path.join(root, directory, tmpl.format('y', idx))).convert(
                    'L')
            elif tmpl == '{:06d}-{}_{:05d}.jpg':  # something v1 flow
                x_img = Image.open(os.path.join(root, '{:06d}'.format(int(directory)), tmpl.
                                                format(int(directory), 'x', idx))).convert('L')
                y_img = Image.open(os.path.join(root, '{:06d}'.format(int(directory)), tmpl.
                                                format(int(directory), 'y', idx))).convert('L')
            else:
                try:
                    # idx_skip = 1 + (idx-1)*5
                    flow = Image.open(os.path.join(root, directory, tmpl.format(idx))).convert(
                        'RGB')
                except Exception:
                    print('error loading flow file:',
                          os.path.join(root, directory, tmpl.format(idx)))
                    flow = Image.open(os.path.join(root, directory, tmpl.format(1))).convert('RGB')
                # the input flow file is RGB image with (flow_x, flow_y, blank) for each channel
                flow_x, flow_y, _ = flow.split()
                x_img = flow_x.convert('L')
                y_img = flow_y.convert('L')

            return [x_img, y_img]

    def _parse_list(self):
        self.video_list = []
        for i,lst in enumerate(self.list_file):
            tmp = [x.strip().split(' ') for x in open(lst)]
            if not self.test_mode or self.remove_missing:
                tmp = [item for item in tmp if int(item[1]) >= 3]
            self.video_list.append([VideoRecord(item) for item in tmp])

            if self.image_tmpl[i] == '{:06d}-{}_{:05d}.jpg':
                for v in self.video_list[i]:
                    v._data[1] = int(v._data[1]) / 2
            print('video number:%d' % (len(self.video_list[i])))

    def _sample_indices(self, record):
        """

        :param record: VideoRecord
        :return: list
        """
        frame_nums = np.array([x.num_frames for x in record])
        argmin_frames = np.argmin(frame_nums)
        differences = frame_nums-frame_nums[argmin_frames]

        if self.dense_sample or (self.random_sample and np.random.randint(2)==0):  # i3d dense sample
            sample_pos = max(1, 1 + record[argmin_frames].num_frames - self.dense_length)
            t_stride = self.dense_length // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record[argmin_frames].num_frames for idx in range(self.num_segments)]
            return [np.array(offsets) + 1 + np.random.randint(x+1) for x in differences]
        else:  # normal sample
            average_duration = (record[argmin_frames].num_frames - self.new_length[argmin_frames] + 1) // self.num_segments
            if average_duration > 0:
                offsets = np.multiply(list(range(self.num_segments)), average_duration) + randint(average_duration,
                                                                                                  size=self.num_segments)
            elif record[argmin_frames].num_frames > self.num_segments:
                offsets = np.sort(randint(record[argmin_frames].num_frames - self.new_length[argmin_frames] + 1, size=self.num_segments))
            else:
                offsets = np.zeros((self.num_segments,))
            return [np.array(offsets) + 1 + np.random.randint(x+1) for x in differences]

    def _get_val_indices(self, record):
        if self.dense_sample:  # i3d dense sample
            sample_pos = max(1, 1 + record[0].num_frames - self.dense_length)
            t_stride = self.dense_length // self.num_segments
            start_idx = 0 if sample_pos == 1 else np.random.randint(0, sample_pos - 1)
            offsets = [(idx * t_stride + start_idx) % record[0].num_frames for idx in range(self.num_segments)]
            return [np.array(offsets) + 1]
        else:
            if record[0].num_frames > self.num_segments + self.new_length[0] - 1:
                tick = (record[0].num_frames - self.new_length[0] + 1) / float(self.num_segments)
                offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            else:
                offsets = np.zeros((self.num_segments,))
            return [offsets + 1]

    def _get_test_indices(self, record):
        if self.random_sample:
            sample_pos = max(1, 1 + record[0].num_frames - self.dense_length)
            t_stride = self.dense_length // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=self.dense_number, dtype=int) if self.dense_number>1 else np.array([int((sample_pos-1)/2)])
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record[0].num_frames for idx in range(self.num_segments)]

            if self.twice_sample:
                tick = (record[0].num_frames - self.new_length[0] + 1) / float(self.num_segments)
                offsets += [int(tick / 2.0 + tick * x) for x in range(self.num_segments)] + [int(tick * x) for x in range(self.num_segments)]
            else:
                tick = (record[0].num_frames - self.new_length[0] + 1) / float(self.num_segments)
                offsets += [int(tick / 2.0 + tick * x) for x in range(self.num_segments)]

            return [np.array(offsets) + 1]
        elif self.dense_sample:
            sample_pos = max(1, 1 + record[0].num_frames - self.dense_length)
            t_stride = self.dense_length // self.num_segments
            start_list = np.linspace(0, sample_pos - 1, num=self.dense_number, dtype=int) if self.dense_number>1 else np.array([int((sample_pos-1)/2)])
            offsets = []
            for start_idx in start_list.tolist():
                offsets += [(idx * t_stride + start_idx) % record[0].num_frames for idx in range(self.num_segments)]
            return [np.array(offsets) + 1]
        elif self.twice_sample:
            tick = (record[0].num_frames - self.new_length[0] + 1) / float(self.num_segments)

            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)] +
                               [int(tick * x) for x in range(self.num_segments)])

            return [offsets + 1]
        else:
            tick = (record[0].num_frames - self.new_length[0] + 1) / float(self.num_segments)
            offsets = np.array([int(tick / 2.0 + tick * x) for x in range(self.num_segments)])
            return [offsets + 1]

    def __getitem__(self, index):
        record = [x[index] for x in self.video_list]
        # check this is a legit video folder

        for i in range(len(self.modality)):
            if self.image_tmpl[i] == 'flow_{}_{:05d}.jpg':
                file_name = self.image_tmpl[i].format('x', 1)
                full_path = os.path.join(self.root_path[i], record[i].path, file_name)
            elif self.image_tmpl[i] == '{:06d}-{}_{:05d}.jpg':
                file_name = self.image_tmpl[i].format(int(record[i].path), 'x', 1)
                full_path = os.path.join(self.root_path[i], '{:06d}'.format(int(record[i].path)), file_name)
            else:
                file_name = self.image_tmpl[i].format(1)
                full_path = os.path.join(self.root_path[i], record[i].path, file_name)

            while not os.path.exists(full_path):
                print('################## Not Found:', os.path.join(self.root_path[i], record[i].path, file_name))
                index = np.random.randint(len(self.video_list[i]))
                record[i] = self.video_list[i][index]
                if self.image_tmpl[i] == 'flow_{}_{:05d}.jpg':
                    file_name = self.image_tmpl[i].format('x', 1)
                    full_path = os.path.join(self.root_path[i], record[i].path, file_name)
                elif self.image_tmpl[i] == '{:06d}-{}_{:05d}.jpg':
                    file_name = self.image_tmpl[i].format(int(record[i].path), 'x', 1)
                    full_path = os.path.join(self.root_path[i], '{:06d}'.format(int(record[i].path)), file_name)
                else:
                    file_name = self.image_tmpl[i].format(1)
                    full_path = os.path.join(self.root_path[i], record[i].path, file_name)

        if not self.test_mode:
            segment_indices = self._sample_indices(record) if self.random_shift else self._get_val_indices(record)
        else:
            segment_indices = self._get_test_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in zip(*indices):
            p = [int(x) for x in seg_ind]
            for part in range(len(self.modality)):
                for i in range(self.new_length[part]):
                    seg_imgs = self._load_image(record[part].path, p[part], self.modality[part], self.root_path[part], self.image_tmpl[part])
                    images.extend(seg_imgs)
                    if not self.mml:
                        images.extend(seg_imgs)
                    if p[part] < record[part].num_frames:
                        p[part] += 1

        process_data = self.transform(images)
        return process_data, record[0].label

    def __len__(self):
        return len(self.video_list[0])
