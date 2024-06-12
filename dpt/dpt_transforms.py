import numpy as np
import cv2


class Resize(object):
    def __init__(self, width, height, resize_target=True, keep_aspect_ratio=False, ensure_multiple_of=1,
                 resize_method='lower_bound', interpolation_method=cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.resize_target = resize_target
        self.keep_aspect_ratio = keep_aspect_ratio
        self.multiple_of = ensure_multiple_of
        self.resize_method = resize_method
        self.interpolation_method = interpolation_method

    def constrain_to_multiple_of(self, image, min_val=0, max_val=None):
        result = (np.round(image / self.multiple_of) * self.multiple_of).astype(int)

        if max_val is not None and result > max_val:
            result = (np.floor(image / self.multiple_of) * self.multiple_of).astype(int)

        if result < min_val:
            result = (np.ceil(image / self.multiple_of) * self.multiple_of).astype(int)

        return result

    def get_size(self, width, height):
        scale_height = self.height / height
        scale_width = self.width / width

        if self.keep_aspect_ratio:
            if self.resize_method == 'lower_bound':
                if scale_width > scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.resize_method == 'upper_bound':
                if scale_width < scale_height:
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            elif self.resize_method == 'minimal':
                if abs(1 - scale_width) < abs(1 - scale_height):
                    scale_height = scale_width
                else:
                    scale_width = scale_height
            else:
                raise ValueError(f'resize_method {self.resize_method} not implemented')

        if self.resize_method == 'lower_bound':
            new_height = self.constrain_to_multiple_of(scale_height * height, min_val=self.height)
            new_width = self.constrain_to_multiple_of(scale_width * width, min_val=self.width)
        elif self.resize_method == 'upper_bound':
            new_height = self.constrain_to_multiple_of(scale_height * height, max_val=self.height)
            new_width = self.constrain_to_multiple_of(scale_width * width, max_val=self.width)
        elif self.resize_method == 'minimal':
            new_height = self.constrain_to_multiple_of(scale_height * height)
            new_width = self.constrain_to_multiple_of(scale_width * width)
        else:
            raise ValueError(f'resize_method {self.resize_method} not implemented')

        return new_width, new_height

    def __call__(self, sample):
        width, height = self.get_size(sample['image'].shape[1], sample['image'].shape[0])

        sample['image'] = cv2.resize(sample['image'], (width, height), interpolation=self.interpolation_method)

        if self.resize_target:
            if 'disparity' in sample:
                sample['disparity'] = cv2.resize(sample['disparity'], (width, height), interpolation=cv2.INTER_NEAREST)

            if 'depth' in sample:
                sample['depth'] = cv2.resize(sample['depth'], (width, height), interpolation=cv2.INTER_NEAREST)

            sample['mask'] = cv2.resize(sample['mask'].astype(np.float32), (width, height),
                                        interpolation=cv2.INTER_NEAREST)
            sample['mask'] = sample['mask'].astype(bool)

        return sample


class NormalizeImage(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, sample):
        sample['image'] = (sample['image'] - self.mean) / self.std

        return sample


class PrepareForNet(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        image = np.transpose(sample['image'], (2, 0, 1))
        sample['image'] = np.ascontiguousarray(image).astype(np.float32)

        if 'mask' in sample:
            sample['mask'] = sample['mask'].astype(np.float32)
            sample['mask'] = np.ascontiguousarray(sample['mask'])

        if 'disparity' in sample:
            disparity = sample['disparity'].astype(np.float32)
            sample['disparity'] = np.ascontiguousarray(disparity)

        if 'depth' in sample:
            depth = sample['depth'].astype(np.float32)
            sample['depth'] = np.ascontiguousarray(depth)

        return sample
