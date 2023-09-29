import pickle
import random

from scipy.optimize import linear_sum_assignment
import numpy as np
import torch
from torch import nn
from einops import repeat, rearrange

from dataset.box import Box
from util.utility import DataTable
from model.util import dot_product_loss, tanh_nd, box_loss
from model.perceiver import Perceiver, FeedForward, DetrSinePositionEmbedding

class Track:
    def __init__(self, id, features, box, output_idx, target_idx=None, target_label=None):
        self.id = id
        self.age = -1
        self.update(features, box, output_idx, target_idx, target_label)

    def update(self, features, box, output_idx, target_idx=None, target_label=None):
        self.target_idx = target_idx
        self.target_label = target_label
        self.output_idx = output_idx
        self.features = features
        self.box = box
        self.age += 1

class TrackGroup:
    def __init__(self, max_tracks, token_length, device, clusters=None, augmentation_rate=1./3.):
        self.max_tracks = max_tracks
        self.token_length = token_length
        self.device = device
        self.clusters = clusters

        self.tracks = []

        self.sequence_counter = 0 # used in inference
        self.thresholds = {'add': 0.12, 'remove': 0.58}

        avg_active_tracks = 2
        self.fp_rate = augmentation_rate * avg_active_tracks / (self.max_tracks - avg_active_tracks)
        self.fn_rate = augmentation_rate

        self.unassigned_outputs = None # used for augmentation

    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        return self.tracks[idx]

    def clear(self):
        self.tracks = []

    def get_boxes(self):
        return TrackGroup.to_boxes(self.tracks)

    @staticmethod
    def to_boxes(tracks):
        boxes = []
        for track in tracks:
            np_box = track.box.copy().to_numpy()
            boxes.append(Box(
                location=np_box['center'][0] * Box.world_normalization,
                direction=np_box['direction'][0],
                dimensions=np.clip(np_box['dimensions'][0] * Box.mean_dimensions, a_min=0., a_max=None),
                label=np.argmax(np_box['cls']),
                confidence=np_box['conf'][0, 0],
                sequence_id=None,
                track_id= track.id
            ))
        return boxes

    def inference_assignment(self, features, outputs, nms_iou_thres=0.1):
        new_tracks = []
        removed_tracks, added_tracks = [], []
        for idx, conf in enumerate(outputs['conf']):
            if idx < len(self.tracks): # existing track
                track = self.tracks[idx]
                if conf < self.thresholds['remove']: # remove track
                    # pass
                    removed_tracks.append(self.tracks[idx].id)
                else: # continue track
                    track.update(features[idx], outputs[idx], idx)
                    new_tracks.append(track)
            elif idx >= self.max_tracks: # new track
                if conf > self.thresholds['add']:
                    new_tracks.append(Track(self.sequence_counter, features[idx], outputs[idx], idx))
                    added_tracks.append(new_tracks[-1].id)
                    self.sequence_counter += 1
                else: # detection is not added to tracks
                    pass

        # 1. check large overlap in new_tracks
        age_order = np.argsort([-t.age for t in new_tracks]) # order with descending age
        new_tracks = [new_tracks[i] for i in age_order]
        new_boxes = TrackGroup.to_boxes(new_tracks)
        remaining_tracks = np.arange(len(new_tracks))
        nms_new_tracks = []
        nms_new_boxes = []
        while len(remaining_tracks) > 0:
            reference_box = new_boxes[remaining_tracks[0]]
            ious = np.array([reference_box.iou_3d(new_boxes[i]) for i in remaining_tracks])
            large_overlap = ious > nms_iou_thres
            nms_new_tracks.append(new_tracks[remaining_tracks[0]])
            nms_new_boxes.append(new_boxes[remaining_tracks[0]])
            remaining_tracks = remaining_tracks[~large_overlap]

        # 2. check large overlap between removed old_tracks and added new_tracks
        removed_tracks = [t for t in self.tracks if t.id in removed_tracks]
        removed_boxes = TrackGroup.to_boxes(removed_tracks)
        remaining_tracks = np.arange(len(removed_tracks))
        for box, track in zip(nms_new_boxes, nms_new_tracks):
            if track.id not in added_tracks:
                continue
            if len(remaining_tracks) == 0:
                break

            ious = np.array([box.iou_3d(removed_boxes[i]) for i in remaining_tracks])
            largest_overlap = np.argmax(ious)
            if largest_overlap > nms_iou_thres:
                track.id = removed_tracks[largest_overlap].id
                track.age = removed_tracks[largest_overlap].age

                large_overlap = ious > nms_iou_thres
                remaining_tracks = remaining_tracks[~large_overlap]

        return nms_new_tracks

    def training_assignment(self, features, outputs, targets):
        def compute_cost_matrix(output, target):
            n_outputs, n_targets = len(output), len(target)
            output, target = output.copy(), target.copy()
            cdist_cost_matrix = torch.cdist(output['center'][:, :2], target['center'][:, :2])

            output.data = repeat(output.data, 'o c -> (o t) c', t=n_targets)
            target.data = repeat(target.data, 't c -> (o t) c', o=n_outputs)
            iou_cost_matrix = rearrange(box_loss(output, target, type='diou', with_direction=True), '(o t) -> o t', o=n_outputs)

            cost_matrix = 0.5 * cdist_cost_matrix + iou_cost_matrix
            return cost_matrix

        def match_tokens(output, target):
            kmeans_queries = DataTable(self.clusters, spec={k: output.spec[k] for k in ['center', 'direction', 'dimensions']})
            cost_matrix = compute_cost_matrix(kmeans_queries, target)
            output_ind, target_ind = linear_sum_assignment(cost_matrix.detach().cpu().numpy())

            return output_ind, target_ind

        new_tracks = []
        assigned_targets = np.zeros(len(targets), dtype=bool)
        for idx, track in enumerate(self.tracks):
            if track.id in targets['sequence_id']: # only currently tracked sequences that keep being tracked
                target_idx = torch.nonzero(targets['sequence_id'] == track.id)[0][0].item()
                target_label = int(targets['label'][target_idx, 0].item())
                track.update(features[idx], outputs[idx], idx, target_idx, target_label)
                new_tracks.append(track)
                assigned_targets[target_idx] = True
            else:  # currently tracked sequences that end
                pass

        n_detection_queries = len(outputs) - self.max_tracks
        free_target_idxs = np.nonzero(~assigned_targets)[0]
        free_output_idxs = np.ones(n_detection_queries, dtype=bool)
        if len(free_target_idxs) > 0:
            output_idxs, target_idxs = match_tokens(outputs[self.max_tracks:], targets[free_target_idxs])
            free_output_idxs[output_idxs] = False
            newly_assigned_target_idxs = free_target_idxs[target_idxs]
            track_ids = targets['sequence_id'][newly_assigned_target_idxs]

            for track_id, output_idx, target_idx in zip(track_ids, output_idxs, newly_assigned_target_idxs):
                idx = self.max_tracks + output_idx
                target_label = int(targets['label'][target_idx, 0].item())
                new_tracks.append(Track(track_id, features[idx], outputs[idx], idx, target_idx, target_label))

        self.unassigned_outputs = zip(
            free_output_idxs + self.max_tracks,
            features[self.max_tracks:][free_output_idxs],
            outputs[self.max_tracks:][free_output_idxs]
        ) # used for data augmentation
        return new_tracks

    def augment(self):
        if self.unassigned_outputs is None:
            return

        # augment false negatives
        new_tracks = [t for t in self.tracks if random.random() >= self.fn_rate]
        # augment false positives
        for idx, features, output in [o for o in self.unassigned_outputs if random.random() < self.fp_rate]:
            new_tracks.append(Track(-1, features, output, idx, None))
        self.tracks = new_tracks[:self.max_tracks] # don't exceed max allowed number

    def update(self, features, outputs, targets=None):
        if targets is not None:
            new_tracks = self.training_assignment(features, outputs, targets)
        else:
            new_tracks = self.inference_assignment(features, outputs)

        new_tracks = new_tracks[-self.max_tracks:] # trim old tracks in case there are too many
        self.tracks = new_tracks

    def state(self):
        queries = torch.zeros((self.max_tracks, self.token_length), device=self.device)
        mask = torch.zeros((self.max_tracks), dtype=torch.bool, device=self.device)

        if len(self.tracks) > 0:
            queries[:len(self.tracks)] = torch.stack([t.features for t in self.tracks], dim=0)
            mask[:len(self.tracks)] = True
        return queries, mask

class TrackModel(nn.Module):
    spec = {
        'center': [0, 1, 2],
        'direction': [3, 4],
        'dimensions': [5, 6, 7],
        'conf': [8],
        'cls': [9, 10, 11]
    }
    n_classes = 3

    def __init__(self, token_length, n_detection_queries=12):
        super().__init__()

        self.input_length = token_length
        self.token_length = 128

        self.loss_weights = {'center': 5., 'direction': 1., 'dimensions': 1., 'conf': 1., 'cls': 3., 'cls_regularization': 0.01, 'box': 1.}
        self.n_detection_queries = n_detection_queries
        self.n_tracking_queries = 12
        self.get_track_kwargs = lambda n_frames: {'max_tracks': self.n_tracking_queries, 'token_length': self.token_length,
                             'clusters': self.kmeans_queries,
                             'augmentation_rate': 1. / n_frames} # lambda so it can be evaluated lazily (kmeans_queries will change their device during runtime)

        self.pe = DetrSinePositionEmbedding(embedding_dim=self.token_length)
        self.perceiver = Perceiver(
            input_channels = self.token_length,     # number of channels for each token of the input
            input_axis = 1,                         # number of axis for input data (2 for images, 3 for video)
            num_freq_bands = 6,                     # number of freq bands, with original value (2 * K + 1)
            max_freq = 10.,                         # maximum frequency, hyperparameter depending on how fine the data is
            depth = 6,                              # depth of net. The shape of the final attention mechanism will be:
                                                    #   depth * (cross attention -> self_per_cross_attn * self attention)
            num_latents = 24,                       # number of latents, or induced set points, or centroids. different papers giving it different names
            latent_dim = self.token_length,         # latent dimension
            cross_heads = 1,                        # number of heads for cross attention. paper said 1
            latent_heads = 8,                       # number of heads for latent self attention, 8
            cross_dim_head = self.token_length // 1, # =cross_dim_heads # number of dimensions per cross attention head
            latent_dim_head = self.token_length // 8, # =latent_heads # number of dimensions per latent self attention head
            num_classes = 1000,                     # output number of classes
            attn_dropout = 0.3,
            ff_dropout = 0.3,
            weight_tie_layers = False,              # whether to weight tie layers (optional, as indicated in the diagram)
            self_per_cross_attn = 2                 # number of self attention blocks per cross attention
        )

        self.expansion = FeedForward(dim=self.input_length, out_dim=self.token_length)
        self.projection = FeedForward(dim=self.token_length, out_dim=9)
        self.classification = FeedForward(dim=self.token_length, out_dim=3)
        self.queries = FeedForward(dim=(2 + 1056), out_dim=self.token_length)
        self.state_normalization = nn.LayerNorm(self.token_length)
        self.load_anchor_boxes()

    def load_anchor_boxes(self):
        with open('clusters_fullbox_%d.p' % self.n_detection_queries, 'rb') as clusters:
            kmeans_queries = pickle.load(clusters)
        self.register_buffer('kmeans_queries', torch.tensor(kmeans_queries, dtype=torch.float32))
        self.kmeans_queries[:, 7] = 0 # center_z at zero so even lying boxes can have large overlap with tall boxes. difference in metrics is minor

    @staticmethod
    def build_targets(batch):
        box_tensor_spec = {'label': [0], 'center': [1, 2, 3], 'direction': [4, 5], 'dimensions': [6, 7, 8], 'sequence_id': [9], 'label_uncertainty': [10]}
        def generate_box_tensor(targets):
            targets = [target for target in targets]
            data = {
                'label': [target.label for target in targets],
                'label_uncertainty': [target.label_uncertainty for target in targets],
                'center': [target.center() / Box.world_normalization for target in targets],
                'direction': [target.direction for target in targets],
                'dimensions': [np.asarray(target.dimensions) / Box.mean_dimensions for target in targets],
                'sequence_id': [target.sequence_id for target in targets]
            }
            return DataTable.from_dict(data, spec=box_tensor_spec, dtype=torch.float32)

        targets = [[generate_box_tensor(t) for t in (sample.additional_targets + [sample.targets])] for sample in batch]
        # transpose (batch_dim, time_dim) to (time_dim, batch_dim)
        targets = [[t[i] for t in targets] for i in range(len(targets[0]))]
        return targets

    def compute_loss(self, outputs, target_dict, batch_tracks):
        def format_assignment(outputs, targets, batch_tracks):
            output_offsets = np.cumsum([0] + [len(o) for o in outputs])
            output_mask = np.concatenate([np.array([t.output_idx for t in tracks]) + o for tracks, o in zip(batch_tracks, output_offsets)], axis=0).astype(int)
            target_offsets = np.cumsum([0] + [len(t) for t in targets])
            target_mask = np.concatenate([np.array([t.target_idx for t in tracks]) + o for tracks, o in zip(batch_tracks, target_offsets)], axis=0).astype(int)

            device = outputs[0].data.device
            return torch.from_numpy(output_mask).long().to(device), torch.from_numpy(target_mask).long().to(device)

        def format_data(outputs, targets):
            output_tensor = DataTable(torch.cat([o.data for o in outputs], dim=0), outputs[0].spec)
            target_tensor = DataTable(torch.cat([t.data for t in targets], dim=0), targets[0].spec)
            return output_tensor, target_tensor

        targets = target_dict['targets']
        output_tensor, target_tensor = format_data(outputs, targets)
        output_mask, target_mask = format_assignment(outputs, targets, batch_tracks)
        target_conf = torch.zeros(len(output_tensor), device=outputs[0].data.device)
        target_conf[output_mask] = 1.

        occlusion_weights = 0.5 + 0.5 * (1 - target_dict['box_visibilities'][target_mask])
        sample_weights = occlusion_weights / torch.sum(occlusion_weights)
        cls_mask = (target_dict['label_uncertainty'] == 0)[target_mask]

        bce_loss = nn.BCELoss(reduction='mean')
        mse_loss = nn.MSELoss(reduction='none')
        cls_loss = nn.MultiLabelMarginLoss(reduction='mean') # normal MultiMarginLoss has erroneous warning spam bug

        loss_dict = {}
        loss_dict['center'] = torch.dot(mse_loss(output_tensor[output_mask]['center'], target_tensor[target_mask]['center']).mean(dim=1), sample_weights)
        loss_dict['dimensions'] = torch.dot(mse_loss(output_tensor[output_mask]['dimensions'], target_tensor[target_mask]['dimensions']).mean(dim=1), sample_weights)
        loss_dict['direction'] = torch.dot(dot_product_loss(output_tensor[output_mask]['direction'], target_tensor[target_mask]['direction'], with_acos=True, reduce=False), sample_weights)
        loss_dict['conf'] = bce_loss(output_tensor['conf'][:, 0], target_conf)

        cls_targets = -torch.ones_like(output_tensor[output_mask[cls_mask]]['cls']).long()
        cls_targets[:, 0] = target_tensor[target_mask[cls_mask]]['label'][:, 0].long()
        loss_dict['cls'] = cls_loss(output_tensor[output_mask[cls_mask]]['cls'], cls_targets)
        loss_dict['cls_regularization'] = sum(torch.norm(p, p=1) for p in self.classification.parameters())
        loss_dict['box'] = torch.dot(box_loss(output_tensor[output_mask], target_tensor[target_mask]), sample_weights)

        loss_dict['total'] = sum(self.loss_weights[key] * loss_dict[key] for key in loss_dict)
        return loss_dict

    def project_and_transform(self, _output):
        _output = torch.cat((
            self.projection(_output),
            self.classification(_output)
        ), dim=-1)

        output = torch.empty(_output.shape, device=_output.device)
        output[..., TrackModel.spec['center']] = _output[..., TrackModel.spec['center']]
        output[..., TrackModel.spec['dimensions']] = _output[..., TrackModel.spec['dimensions']]
        output[..., TrackModel.spec['direction']] = tanh_nd(_output[..., TrackModel.spec['direction']], dim=-1)
        output[..., TrackModel.spec['conf']] = torch.sigmoid(_output[..., TrackModel.spec['conf']])
        output[..., TrackModel.spec['cls']] = torch.softmax(_output[..., TrackModel.spec['cls']], dim=-1) # margin loss needs softmax?
        return output

    def state(self, track_states, context):
        batch_size, device = len(track_states), context.device

        tracking_queries, tracking_mask = [torch.stack([s[i] for s in track_states], dim=0) for i in range(2)]
        detection_queries = self.queries(torch.cat((
            repeat(self.kmeans_queries[:, :2], 'q c -> b q c', b=batch_size),
            repeat(context, 'b c -> b q c', q=self.n_detection_queries)
        ), dim=-1))

        detection_mask = torch.ones((batch_size, self.n_detection_queries), device=device, dtype=torch.bool)
        full_queries = torch.cat((tracking_queries, detection_queries), dim=1)
        full_mask = torch.cat((tracking_mask, detection_mask), dim=1)
        return full_queries, full_mask

    def step(self, features, track_states, context, mask=None):
        state, latent_mask = self.state(track_states, context)
        if self.training:
            state = state + torch.randn_like(state) * 0.02
        state = self.state_normalization(state)

        features = self.expansion(features) + self.pe(features[..., :2])
        output = self.perceiver(
            data=features,
            latent_mask=latent_mask,
            mask=mask,
            return_embeddings=True,
            latents=state,
        )
        boxes = self.project_and_transform(output)
        return output, boxes

    def forward(self, data_dict, targets=None):
        device = self.kmeans_queries.device
        data = data_dict['features']
        n_frames, batch_size = data.shape[0], data.shape[1]
        step_losses = []
        batch_tracks = [TrackGroup(device=device, **self.get_track_kwargs(n_frames)) for _ in range(batch_size)]

        for step in range(data.shape[0]):
            if self.training:
                [tracks.augment() for tracks in batch_tracks]

            features = data[step]
            track_states = [tracks.state() for tracks in batch_tracks]
            output, boxes = self.step(features, track_states, data_dict['context'][step], mask=data_dict['cell_mask'][step])
            boxes = [DataTable(b, spec=TrackModel.spec) for b in boxes]

            if self.training:
                assert targets is not None
                [tracks.update(f, o, t) for tracks, f, o, t in zip(batch_tracks, output, boxes, targets['transformer'][step]['targets'])]
                loss_dict = self.compute_loss(boxes, targets['transformer'][step], batch_tracks)
                step_losses.append(loss_dict)
            else:
                [tracks.update(f, o) for tracks, f, o in zip(batch_tracks, output, boxes)]

        if self.training:
            return step_losses
        else:
            return [tracks.get_boxes() for tracks in batch_tracks]
