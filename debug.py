from PIL import Image
import torch
from torchvision.transforms import v2

from Create_Ushape_net import Create_Ushape_Net
ushapenet=Create_Ushape_Net(Encode_Type='dinov3',output_channel=11,)

def get_img():
    # url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open('visuals/000000039769.jpg').convert("RGB")
    return image

def make_transform(resize_size: int | list[int] = 768):
    to_tensor = v2.ToImage()
    resize = v2.Resize((resize_size, resize_size), antialias=True)
    to_float = v2.ToDtype(torch.float32, scale=True)
    normalize = v2.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    return v2.Compose([to_tensor, resize, to_float, normalize])
REPO_DIR='/opt/dl_workspace/algorithm/04-myself/domaingap/dinov3_main'
BACKBONE_NAME='dinov3_vits16plus'#dinov3_vits16plus dinov3_vits16

WEIGHT_DICT={'dinov3_vits16':'dinov3_vits16_pretrain_lvd1689m-08c60483.pth','dinov3_vits16plus':'dinov3_vits16plus_pretrain_lvd1689m-4057cbaa.pth'}
WEIGHT=REPO_DIR+'/checkpoints/'+WEIGHT_DICT[BACKBONE_NAME]
# encoder = torch.hub.load(REPO_DIR, 'dinov3_vit7b16_dd', source="local", weights='/opt/dl_workspace/algorithm/04-myself/domaingap/dinov3/checkpoints/dinov3_vit7b16_synthmix_dpt_head-02040be1.pth')
encoder = torch.hub.load(REPO_DIR, BACKBONE_NAME, source='local', weights=WEIGHT)
img_size = 512
H = W = img_size
patch_size = 16
grid_h = H // patch_size
grid_w = W // patch_size

img = get_img()
transform = make_transform(img_size)
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.bfloat16):
        batch_img = transform(img)[None]
        batch_img = batch_img.repeat(8, 1, 1, 1)
        # output = encoder.forward_features(batch_img)
        # patch_tokens = output['x_norm_patchtokens']  # [1, N, 384]
        # # 如果您知道 H_grid 和 W_grid
        # feature_map = patch_tokens.permute(0, 2, 1).reshape(1, 384, grid_h, grid_w)
# plt.figure(figsize=(12, 6))

# from dinov3_main.dinov3.eval.segmentation.models import build_segmentation_decoder
# from dinov3_main.dinov3.eval.segmentation.models import BackboneLayersSet
# segmentation_model = build_segmentation_decoder(
#     encoder,
#     BackboneLayersSet.FOUR_EVEN_INTERVALS,
#     "m2f",
#     hidden_dim=1024,
#     num_classes=11,
#     autocast_dtype=torch.float32, #torch.bfloat16, torch.float32
#     dropout=0.1,
# )

from dinov3_main.dinov3.eval.depth.models import build_depther
from dinov3_main.dinov3.eval.depth.models.encoder import BackboneLayersSet as bls

depther_model = build_depther(
    encoder,
    backbone_out_layers=(
        bls.FOUR_EVEN_INTERVALS  # One of BackboneLayersSet(Enum) or a list of indices e.g. [0, 1, 2, 3]
    ),
    n_output_channels=256,
    # use_backbone_norm=config.use_backbone_norm,
    use_batchnorm=True,
    use_cls_token=False,
    # adapt_to_patch_size="center_padding",
    head_type='dpt',
    autocast_dtype=torch.float32,
    # min_depth=config.min_depth,
    # max_depth=config.max_depth,
    # bins_strategy=config.bins_strategy,
    # norm_strategy=config.norm_strategy,
    output_channel=11,
    activate='sigmoid'


)
#'x_norm_clstoken' 'x_storage_tokens' 'x_norm_patchtokens' 'x_prenorm'
depther_model.float()
depther_model.cuda()
with torch.inference_mode():
    with torch.autocast('cuda', dtype=torch.float32):
        # output = encoder.forward_features(batch_img)
        # patch_tokens = output  # [1, N, 384]
        # pred_s = segmentation_model(batch_img)['pred_masks']
        pred_d = depther_model(batch_img.cuda())
print(pred_d.shape)