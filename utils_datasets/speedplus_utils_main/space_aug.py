"""
Space augmentation pipeline compatible with PyTorchSatellitePoseEstimationDataset.
Drop-in replacement for albumentations transform.
"""
import warnings
warnings.filterwarnings('ignore', message='Got processor for keypoints, but no transform to process it.')
import torch
import numpy as np
import albumentations as A

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
_stylaug = None

def _get_stylaug():
    global _stylaug
    if _stylaug is None:
        from styleaug import StyleAugmentor
        _stylaug = StyleAugmentor().to(DEVICE).eval()
    return _stylaug


class SpaceAugTransform:
    """
    Compatible with dataset transform interface:
        trans = SpaceAugTransform('aug4')
        result = trans(image=img, keypoints=kp, mask=mask, coors=coors)
    """
    def __init__(self, aug_type='aug1', styleaug_alpha=0.3, styleaug_p=0.5, to_gray=None):
        self.aug_type = aug_type
        self.styleaug_alpha = styleaug_alpha
        self.styleaug_p = styleaug_p
        self.IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
        self.norm = A.Normalize(mean=self.IMAGENET_DEFAULT_MEAN, std=self.IMAGENET_DEFAULT_STD)
        self._use_styleaug = aug_type in ('styleaug', 'augmix', 'aug4s')
        self._pipeline = self._build()

    def _build(self):
        """Build the albumentations part of the pipeline."""
        t = self.aug_type.lower()
        if t in ('none', 'styleaug'):
            return A.Compose([self.norm],
                             keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
                             additional_targets={'mask': 'mask', 'coors': 'mask'})

        if t == 'augbaseline':
            import cv2
            return A.Compose([
                A.RandomBrightnessContrast(p=1),
                # A.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=45, p=1,
                #                    border_mode=cv2.BORDER_CONSTANT, fill=0),
                A.OneOf([A.GaussNoise()], p=0.5),
                A.OneOf([A.MotionBlur(p=0.5), A.MedianBlur(blur_limit=3, p=0.5),
                         A.Blur(blur_limit=3, p=0.5)], p=1),
                # A.RandomSunFlare(flare_roi=(0, 0, 1, 1), src_radius=400,
                #                  num_flare_circles_range=(1, 2), p=1.0),
            ] + [self.norm], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
                additional_targets={'mask': 'mask', 'coors': 'mask'})

        # Incremental aug levels
        augs = []
        base_augs = [
            A.OneOf([A.GaussianBlur((3,7),p=1), A.MotionBlur((3,7),p=1)], p=0.5),
            A.Sharpen(p=0.5), A.Emboss(p=0.5),
            A.GaussNoise((0.01,0.05),p=0.5), A.CoarseDropout(num_holes_range=(1,8),p=0.5),
            A.RandomBrightnessContrast(contrast_limit=0.3,brightness_limit=0,p=0.5),
            A.RandomBrightnessContrast(contrast_limit=0,brightness_limit=0.3,p=0.5),
            A.InvertImg(p=0.3), A.MultiplicativeNoise((0.9,1.1),p=0.5),
        ]
        aug2_add = [A.Superpixels(p_replace=0.1,n_segments=100,p=0.5),
                     A.CLAHE(clip_limit=2.0,tile_grid_size=(8,8),p=0.5),
                     A.PixelDropout(0.02,p=0.5)]
        aug3_add = [A.ISONoise(p=0.5), A.RandomFog(0.2,p=0.5), A.RandomSnow(0.2,p=0.5),
                     A.RandomSunFlare((0,0,1,1),src_radius=400,p=0.5),
                     A.RandomBrightnessContrast(contrast_limit=0.5,brightness_limit=0,p=0.5)]
        aug4_add = [A.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,hue=0.1,p=0.5),
                     A.HueSaturationValue(20,30,20,p=0.5)]

        augs += base_augs
        if t in ('aug2', 'aug3', 'aug4', 'aug4s', 'aug5', 'augmix'):
            augs += aug2_add
        if t in ('aug3', 'aug4', 'aug4s', 'aug5', 'augmix'):
            augs += aug3_add
        if t in ('aug4', 'aug4s', 'aug5', 'augmix'):
            augs += aug4_add
        augs += [self.norm]

        if t in ('aug5', 'augmix'):
            return self._augmix()

        return A.Compose(augs,
            keypoint_params=A.KeypointParams(format='xy',remove_invisible=False),
            additional_targets={'mask':'mask','coors':'mask'})

    def _augmix(self):
        g_brightness = A.Compose([
            A.SomeOf([A.RandomBrightnessContrast(0.3,0.3,p=1),A.InvertImg(p=1),
                       A.MultiplicativeNoise((0.9,1.1),p=1),
                       A.ColorJitter(brightness=0.3,contrast=0.3,saturation=0.3,p=1),
                       A.HueSaturationValue(20,30,20,p=1)],n=2,p=1)],p=1)
        g_blur = A.Compose([
            A.SomeOf([A.GaussianBlur((3,7),p=1),A.MotionBlur((3,7),p=1),
                       A.Sharpen(p=1),A.Emboss(p=1),
                       A.CLAHE(clip_limit=2.0,tile_grid_size=(8,8),p=1)],n=2,p=1)],p=1)
        g_corrupt = A.Compose([
            A.SomeOf([A.GaussNoise((0.01,0.05),p=1),A.ISONoise(p=1),
                       A.RandomFog(0.2,p=1),A.RandomSnow(0.2,p=1),
                       A.RandomSunFlare((0,0,1,0.5),src_radius=200,p=1)],n=2,p=1)],p=1)
        g_general = A.Compose([
            A.SomeOf([A.CoarseDropout(num_holes_range=(1,8),p=1),A.PixelDropout(0.02,p=1),
                       A.Superpixels(p_replace=0.1,n_segments=100,p=1)],n=2,p=1)],p=1)
        return A.Compose([g_brightness,g_blur,g_corrupt,g_general, self.norm],
                         keypoint_params=A.KeypointParams(format='xy',remove_invisible=False),
                         additional_targets={'mask':'mask','coors':'mask'})

    def __call__(self, **kwargs):
        """Compatible with dataset transform interface"""
        image = kwargs.get('image')
        keypoints = kwargs.get('keypoints')
        mask = kwargs.get('mask')
        coors = kwargs.get('coors')

        # Build args for albumentations
        args = {'image': image}
        if keypoints is not None:
            args['keypoints'] = keypoints
        if mask is not None:
            args['mask'] = mask
        if coors is not None:
            args['coors'] = coors

        # Apply albumentations pipeline
        result = self._pipeline(**args)

        # Apply StyleAugmentor for styleaug and augmix
        if self._use_styleaug and np.random.rand() < self.styleaug_p:
            img_np = result['image']
            x = torch.from_numpy(img_np).permute(2,0,1).float().div(255).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                x = _get_stylaug()(x, alpha=self.styleaug_alpha)
            result['image'] = x.squeeze(0).permute(1,2,0).mul(255).clamp(0,255).byte().cpu().numpy()

        return result
