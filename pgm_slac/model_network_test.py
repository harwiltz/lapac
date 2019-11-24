import unittest

from pgm_slac.model_network import FeatureExtractor
from pgm_slac.model_network import Decoder

import numpy as np
import torch

class ModelNetworkTest(unittest.TestCase):
    def test_encoding_dims(self):
        fe = FeatureExtractor()
        images = np.random.uniform(size=(7,8,81,81,3))
        images = torch.FloatTensor(images)
        features = fe(images)
        self.assertEqual(features.shape, torch.Size([7, 8, 256, 1, 1]))

    def test_decoding_dims(self):
        de = Decoder()
        features = np.random.uniform(size=(7, 8, 256, 1, 1))
        features = torch.FloatTensor(features)
        dists = de(features)
        self.assertEqual(dists.batch_shape, torch.Size([7, 8]))
        self.assertEqual(dists.event_shape, torch.Size([81, 81, 3]))

    def test_encode_decode_dims(self):
        img_shape = (7, 8, 81, 81, 3)
        fe = FeatureExtractor()
        de = Decoder()
        images = np.random.uniform(size=img_shape)
        images = torch.FloatTensor(images)
        features = fe(images)
        dists = de(features)
        self.assertEqual(dists.batch_shape, torch.Size(img_shape[:2]))
        self.assertEqual(dists.event_shape, torch.Size(img_shape[2:]))
        dists.log_prob(images)
