# # Modified from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
# from typing import Optional
#
# from torch import nn, Tensor
# from torch.nn import functional as F
#
# import torch.nn.functional as F
# # from detectron2.projects.point_rend.point_features import point_sample
# from scipy.optimize import linear_sum_assignment
# from torch import nn
# from torch.cuda.amp import autocast
# import scipy.optimize
# from typing import List, Optional
#
# import torch
# import torch.distributed as dist
# import torchvision
# from torch import Tensor
#
# class QueryProposal(nn.Module):
#
#     def __init__(self, num_features, num_queries, num_classes):
#         super().__init__()
#         self.topk = num_queries
#         self.num_classes = num_classes
#
#         self.conv_proposal_cls_logits = nn.Sequential(
#             nn.Conv2d(num_features, num_features, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(num_features, num_classes + 1, kernel_size=1, stride=1, padding=0),
#         )
#
#     @torch.no_grad()
#     def compute_coordinates(self, x):
#         h, w = x.size(2), x.size(3)
#         y_loc = torch.linspace(0, 1, h, device=x.device)
#         x_loc = torch.linspace(0, 1, w, device=x.device)
#         y_loc, x_loc = torch.meshgrid(y_loc, x_loc)
#         locations = torch.stack([x_loc, y_loc], 0).unsqueeze(0)
#         return locations
#
#     def seek_local_maximum(self, x, epsilon=1e-6):
#         """
#         inputs:
#             x: torch.tensor, shape [b, c, h, w]
#         return:
#             torch.tensor, shape [b, c, h, w]
#         """
#         x_pad = F.pad(x, (1, 1, 1, 1), "constant", 0)
#         # top, bottom, left, right, top-left, top-right, bottom-left, bottom-right
#         maximum = (x >= x_pad[:, :, :-2, 1:-1]) & \
#                   (x >= x_pad[:, :, 2:, 1:-1]) & \
#                   (x >= x_pad[:, :, 1:-1, :-2]) & \
#                   (x >= x_pad[:, :, 1:-1, 2:]) & \
#                   (x >= x_pad[:, :, :-2, :-2]) & \
#                   (x >= x_pad[:, :, :-2, 2:]) & \
#                   (x >= x_pad[:, :, 2:, :-2]) & \
#                   (x >= x_pad[:, :, 2:, 2:]) & \
#                   (x >= epsilon)
#         return maximum.to(x)
#
#     def forward(self, x, pos_embeddings):
#
#         proposal_cls_logits = self.conv_proposal_cls_logits(x)  # b, c, h, w
#         proposal_cls_probs = proposal_cls_logits.softmax(dim=1)  # b, c, h, w
#         proposal_cls_one_hot = F.one_hot(proposal_cls_probs[:, :-1, :, :].max(1)[1],
#                                          num_classes=self.num_classes + 1).permute(0, 3, 1, 2)  # b, c, h, w
#         proposal_cls_probs = proposal_cls_probs.mul(proposal_cls_one_hot)
#         proposal_local_maximum_map = self.seek_local_maximum(proposal_cls_probs)  # b, c, h, w
#         proposal_cls_probs = proposal_cls_probs + proposal_local_maximum_map  # b, c, h, w
#
#         # top-k indices
#         # x的分辨率是（30, 54），展平后相当于有 1620个索引代表（30，54）中的每个像素位置
#         # 而 topk_indices就是topk个取值范围在 0~1620 内的数，代表的是前topk个最大类别概率(logits)的像素位置
#         topk_indices = torch.topk(proposal_cls_probs[:, :-1, :, :].flatten(2).max(1)[0], self.topk, dim=1)[1]  # b, q
#         topk_indices = topk_indices.unsqueeze(1)  # b, 1, q
#
#         # topk queries
#         topk_proposals = torch.gather(x.flatten(2), dim=2, index=topk_indices.repeat(1, x.shape[1], 1))  # b, c, q
#         pos_embeddings = pos_embeddings.repeat(x.shape[0], 1, 1, 1).flatten(2)
#         topk_pos_embeddings = torch.gather(
#             pos_embeddings, dim=2, index=topk_indices.repeat(1, pos_embeddings.shape[1], 1)
#         )  # b, c, q
#         '''
#         原版
#         if self.training:
#             locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
#             topk_locations = torch.gather(
#                 locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
#             )
#             topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
#         else:
#             topk_locations = None
#         '''
#
#         ''' 删除 '''
#         locations = self.compute_coordinates(x).repeat(x.shape[0], 1, 1, 1)
#         topk_locations = torch.gather(
#             locations.flatten(2), dim=2, index=topk_indices.repeat(1, locations.shape[1], 1)
#         )
#         topk_locations = topk_locations.transpose(-1, -2)  # b, q, 2
#         return topk_proposals, topk_pos_embeddings, topk_locations, proposal_cls_logits
#
#
# def batch_dice_loss(inputs: torch.Tensor, targets: torch.Tensor):
#     """
#     Compute the DICE loss, similar to generalized IOU for masks
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     """
#     inputs = inputs.sigmoid()
#     inputs = inputs.flatten(1)
#     numerator = 2 * torch.einsum("nc,mc->nm", inputs, targets)
#     denominator = inputs.sum(-1)[:, None] + targets.sum(-1)[None, :]
#     loss = 1 - (numerator + 1) / (denominator + 1)
#     return loss
#
#
# batch_dice_loss_jit = torch.jit.script(
#     batch_dice_loss
# )  # type: torch.jit.ScriptModule
#
#
# def batch_sigmoid_ce_loss(inputs: torch.Tensor, targets: torch.Tensor):
#     """
#     Args:
#         inputs: A float tensor of arbitrary shape.
#                 The predictions for each example.
#         targets: A float tensor with the same shape as inputs. Stores the binary
#                  classification label for each element in inputs
#                 (0 for the negative class and 1 for the positive class).
#     Returns:
#         Loss tensor
#     """
#     hw = inputs.shape[1]
#
#     pos = F.binary_cross_entropy_with_logits(
#         inputs, torch.ones_like(inputs), reduction="none"
#     )
#     neg = F.binary_cross_entropy_with_logits(
#         inputs, torch.zeros_like(inputs), reduction="none"
#     )
#
#     loss = torch.einsum("nc,mc->nm", pos, targets) + torch.einsum(
#         "nc,mc->nm", neg, (1 - targets)
#     )
#
#     return loss / hw
#
#
# batch_sigmoid_ce_loss_jit = torch.jit.script(
#     batch_sigmoid_ce_loss
# )  # type: torch.jit.ScriptModule
#
#
# class HungarianMatcher(nn.Module):
#     """This class computes an assignment between the targets and the predictions of the network
#
#     For efficiency reasons, the targets don't include the no_object. Because of this, in general,
#     there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
#     while the others are un-matched (and thus treated as non-objects).
#     """
#
#     def __init__(self, cost_class: float = 1, cost_mask: float = 1, cost_dice: float = 1, cost_location: float = 1e3,
#                  num_points: int = 0):
#         """Creates the matcher
#
#         Params:
#             cost_class: This is the relative weight of the classification error in the matching cost
#             cost_mask: This is the relative weight of the focal loss of the binary mask in the matching cost
#             cost_dice: This is the relative weight of the dice loss of the binary mask in the matching cost
#             cost_location: This is the relative weight of the location loss of the query in the matching cost
#         """
#         super().__init__()
#         self.cost_class = cost_class
#         self.cost_mask = cost_mask
#         self.cost_dice = cost_dice
#         self.cost_location = cost_location
#
#         assert cost_class != 0 or cost_mask != 0 or cost_dice != 0, "all costs cant be 0"
#
#         self.num_points = num_points
#
#     @torch.no_grad()
#     def memory_efficient_forward(self, outputs, targets):
#         """memory-friendly matching"""
#         bs, num_queries = outputs["pred_logits"].shape[:2]
#
#         indices = []
#         # Iterate through batch size
#         for b in range(bs):
#             out_query_loc = outputs["query_locations"][b]  # [num_queries, 2(x, y)]
#             out_prob = outputs["pred_logits"][b].softmax(-1)  # [num_queries, num_classes]
#             out_mask = outputs["pred_masks"][b]  # [num_queries, H_pred, W_pred]
#             # gt masks are already padded when preparing target
#             tgt_mask = targets[b]["masks"].to(out_mask)  # [num_obj, h, w]
#             tgt_ids = targets[b]["labels"]
#
#             cost_location = point_sample(
#                 tgt_mask.unsqueeze(0),
#                 out_query_loc.unsqueeze(0),
#                 align_corners=False
#             ).squeeze(0)  # [num_obj, num_queries]
#             cost_location = (cost_location > 0).to(out_mask)
#             # add location cost when the proposal is not inside instance regions.
#             cost_location = -cost_location.transpose(0, 1)  # [num_queries, num_obj]
#
#             # Compute the classification cost. Contrary to the loss, we don't use the NLL,
#             # but approximate it in 1 - proba[target class].
#             # The 1 is a constant that doesn't change the matching, it can be ommitted.
#             cost_class = -out_prob[:, tgt_ids]  # [num_queries, num_obj]
#
#             # all masks share the same set of points for efficient matching!
#             point_coords = torch.rand(1, self.num_points, 2, device=out_mask.device)
#             # get gt labels
#             tgt_mask = point_sample(
#                 tgt_mask.unsqueeze(0),
#                 point_coords,
#                 align_corners=False,
#             ).squeeze(0)
#
#             out_mask = point_sample(
#                 out_mask.unsqueeze(0),
#                 point_coords,
#                 align_corners=False,
#             ).squeeze(0)
#
#             with autocast(enabled=False):
#                 out_mask = out_mask.float()
#                 tgt_mask = tgt_mask.float()
#                 # Compute the focal loss between masks
#                 cost_mask = batch_sigmoid_ce_loss_jit(out_mask, tgt_mask)
#
#                 # Compute the dice loss between masks
#                 if tgt_mask.shape[0] > 0:
#                     cost_dice = batch_dice_loss_jit(out_mask, tgt_mask)
#                 else:
#                     cost_dice = batch_dice_loss(out_mask, tgt_mask)
#
#             # Final cost matrix
#             C = (
#                     self.cost_mask * cost_mask
#                     + self.cost_class * cost_class
#                     + self.cost_dice * cost_dice
#                     + self.cost_location * cost_location
#             )
#             C = C.reshape(num_queries, -1).cpu()
#             if torch.isnan(C).any() or torch.isinf(C).any():
#                 print("Detected NaN or Inf in tensor:", C)
#             indices.append(linear_sum_assignment(C))
#
#         return [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices
#         ]
#
#     @torch.no_grad()
#     def memory_efficient_forward_for_proposal(self, outputs, targets):
#         """memory-friendly matching for proposals"""
#         bs = outputs["proposal_cls_logits"].shape[0]
#         proposal_size = outputs["proposal_cls_logits"].shape[-2:] # 用于生成query的特征图的宽高
#
#         indices = []
#         # Iterate through batch size
#         for b in range(bs):
#             proposal_cls_prob = outputs["proposal_cls_logits"][b].flatten(1).transpose(0, 1).softmax(
#                 -1)  # [proposal_hw, num_classes + 1]
#
#             # gt masks are already padded when preparing target
#             tgt_mask = targets[b]["masks"].to(proposal_cls_prob)  # [num_obj, h, w]
#             tgt_ids = targets[b]["labels"]
#
#             if tgt_mask.shape[0] > 0:
#                 scaled_tgt_mask = F.adaptive_avg_pool2d(tgt_mask.unsqueeze(0),
#                                                         output_size=proposal_size)
#                 scaled_tgt_mask = (scaled_tgt_mask.squeeze(0) > 0.).to(
#                     proposal_cls_prob)  # [num_obj, proposal_h ,proposal_w]
#             else:
#                 scaled_tgt_mask = torch.zeros([tgt_mask.shape[0], *proposal_size],
#                                               device=proposal_cls_prob.device)
#
#             # add location cost when the proposal is not inside the instance region.
#             cost_location = -scaled_tgt_mask.flatten(1).transpose(0, 1)  # [proposal_hw, num_obj]
#
#             # Compute the classification cost. Contrary to the loss, we don't use the NLL,
#             # but approximate it in 1 - proba[target class].
#             # The 1 is a constant that doesn't change the matching, it can be omitted.
#             cost_class = -proposal_cls_prob[:, tgt_ids]  # [proposal_hw, num_obj]
#
#             # Proposal cost matrix
#             C = self.cost_class * cost_class + self.cost_location * cost_location
#             C = C.reshape(proposal_size[0] * proposal_size[1], -1).cpu()
#             indices.append(linear_sum_assignment(C))
#
#         return [
#             (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
#             for i, j in indices
#         ]
#
#     @torch.no_grad()
#     def forward(self, outputs, targets):
#         """Performs the matching
#
#         Params:
#             outputs: This is a dict that contains at least these entries:
#                  "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
#                  "pred_masks": Tensor of dim [batch_size, num_queries, H_pred, W_pred] with the predicted masks
#
#             targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
#                  "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
#                            objects in the target) containing the class labels
#                  "masks": Tensor of dim [num_target_boxes, H_gt, W_gt] containing the target masks
#
#         Returns:
#             A list of size batch_size, containing tuples of (index_i, index_j) where:
#                 - index_i is the indices of the selected predictions (in order)
#                 - index_j is the indices of the corresponding selected targets (in order)
#             For each batch element, it holds:
#                 len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
#         """
#         if outputs.get("proposal_cls_logits", None) is not None:
#             return self.memory_efficient_forward_for_proposal(outputs, targets)
#         return self.memory_efficient_forward(outputs, targets)
#
#     def __repr__(self, _repr_indent=4):
#         head = "Matcher " + self.__class__.__name__
#         body = [
#             "cost_class: {}".format(self.cost_class),
#             "cost_mask: {}".format(self.cost_mask),
#             "cost_dice: {}".format(self.cost_dice),
#         ]
#         lines = [head] + [" " * _repr_indent + line for line in body]
#         return "\n".join(lines)
#
#
# def _max_by_axis(the_list):
#     # type: (List[List[int]]) -> List[int]
#     maxes = the_list[0]
#     for sublist in the_list[1:]:
#         for index, item in enumerate(sublist):
#             maxes[index] = max(maxes[index], item)
#     return maxes
#
#
# class NestedTensor(object):
#     def __init__(self, tensors, mask: Optional[Tensor]):
#         self.tensors = tensors
#         self.mask = mask
#
#     def to(self, device):
#         # type: (Device) -> NestedTensor # noqa
#         cast_tensor = self.tensors.to(device)
#         mask = self.mask
#         if mask is not None:
#             assert mask is not None
#             cast_mask = mask.to(device)
#         else:
#             cast_mask = None
#         return NestedTensor(cast_tensor, cast_mask)
#
#     def decompose(self):
#         return self.tensors, self.mask
#
#     def __repr__(self):
#         return str(self.tensors)
#
#
# def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
#     # TODO make this more general
#     if tensor_list[0].ndim == 3:
#         if torchvision._is_tracing():
#             # nested_tensor_from_tensor_list() does not export well to ONNX
#             # call _onnx_nested_tensor_from_tensor_list() instead
#             return _onnx_nested_tensor_from_tensor_list(tensor_list)
#
#         # TODO make it support different-sized images
#         max_size = _max_by_axis([list(img.shape) for img in tensor_list])
#         # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
#         batch_shape = [len(tensor_list)] + max_size
#         b, c, h, w = batch_shape
#         dtype = tensor_list[0].dtype
#         device = tensor_list[0].device
#         tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
#         mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
#         for img, pad_img, m in zip(tensor_list, tensor, mask):
#             pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
#             m[: img.shape[1], : img.shape[2]] = False
#     else:
#         raise ValueError("not supported")
#     return NestedTensor(tensor, mask)
#
#
# # _onnx_nested_tensor_from_tensor_list() is an implementation of
# # nested_tensor_from_tensor_list() that is supported by ONNX tracing.
# @torch.jit.unused
# def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
#     max_size = []
#     for i in range(tensor_list[0].dim()):
#         max_size_i = torch.max(
#             torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
#         ).to(torch.int64)
#         max_size.append(max_size_i)
#     max_size = tuple(max_size)
#
#     # work around for
#     # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
#     # m[: img.shape[1], :img.shape[2]] = False
#     # which is not yet supported in onnx
#     padded_imgs = []
#     padded_masks = []
#     for img in tensor_list:
#         padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
#         padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
#         padded_imgs.append(padded_img)
#
#         m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
#         padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
#         padded_masks.append(padded_mask.to(torch.bool))
#
#     tensor = torch.stack(padded_imgs)
#     mask = torch.stack(padded_masks)
#
#     return NestedTensor(tensor, mask=mask)
#
#
# def is_dist_avail_and_initialized():
#     if not dist.is_available():
#         return False
#     if not dist.is_initialized():
#         return False
#     return True