from typing import Callable, Dict, List, Tuple

import torch

from .cerberus_semantic_id import CerberusSemanticIDBranch


class CerberusAPSBranch(CerberusSemanticIDBranch):
    def empty_aps_losses(self, device: torch.device) -> Dict[str, torch.Tensor]:
        zero = torch.tensor(0.0, device=device)
        losses = {}
        for group_name in self.group_names:
            losses[f"loss_aps_cls_{group_name}"] = zero
            losses[f"loss_aps_align_{group_name}"] = zero
            losses[f"loss_aps_guidance_{group_name}"] = zero
            losses[f"loss_aps_relation_{group_name}"] = zero
        return losses

    def encode_attribute_text_groups(
        self,
        attributes_list: List[Dict[str, int]],
        tokenize_fn: Callable[[str], torch.Tensor],
        encode_text_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        prompts_by_group, valid_masks = self.build_group_prompts(attributes_list)
        text_feature_dict = {}
        for group_name in self.group_names:
            group_tokens = torch.stack(
                [tokenize_fn(prompt) for prompt in prompts_by_group[group_name]]
            ).to(device)
            text_feature_dict[f"{group_name}_embedding"] = encode_text_fn(group_tokens)
        text_embeddings = self.project_text_embeddings(text_feature_dict)
        valid_masks = {key: value.to(device) for key, value in valid_masks.items()}
        return text_embeddings, valid_masks

    def encode_query_from_attributes(
        self,
        attributes_list: List[Dict[str, int]],
        tokenize_fn: Callable[[str], torch.Tensor],
        encode_text_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        text_embeddings, valid_masks = self.encode_attribute_text_groups(
            attributes_list,
            tokenize_fn,
            encode_text_fn,
            device,
        )
        group_feats = self.stack_embeddings(text_embeddings)
        group_valid = self.stack_valid_masks(valid_masks)
        group_feats = group_feats * group_valid.unsqueeze(-1).to(group_feats.dtype)
        flat_feats = group_feats.flatten(1)
        return flat_feats, group_feats, group_valid

    def compute_aps_losses(
        self,
        image_embedding_dict: Dict[str, torch.Tensor],
        attributes_list: List[Dict[str, int]],
        tokenize_fn: Callable[[str], torch.Tensor],
        encode_text_fn: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
        image_valid_mask_dict: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if not attributes_list:
            return self.empty_aps_losses(device)

        text_embeddings, text_valid_masks = self.encode_attribute_text_groups(
            attributes_list,
            tokenize_fn,
            encode_text_fn,
            device,
        )
        raw_losses = self.compute_text_alignment_losses(
            image_embedding_dict,
            text_embeddings,
            attributes_list,
            image_valid_mask_dict=image_valid_mask_dict,
            text_valid_mask_dict=text_valid_masks,
        )
        return {key.replace("loss_cerberus_", "loss_aps_"): value for key, value in raw_losses.items()}
