import numpy as np


class Resampler:
    def __init__(self, num: int):
        """Resamples a point cloud containing N points to one containing M

        Guaranteed to have no repeated points if M <= N.
        Otherwise, it is guaranteed that all points appear at least once.

        Args:
            num (int): Number of points to resample to, i.e. M

        """
        self.num = num

    def __call__(self, sample):

        if 'deterministic' in sample and sample['deterministic']:
            np.random.seed(sample['idx'])

        src_size = self.num
        ref_size = (sample['points_ref'].shape[0]//sample['points_src'].shape[0]) * self.num
        sample['points_src'] = self._resample(sample['points_src'], src_size)
        sample['points_ref'] = self._resample(sample['points_ref'], ref_size)

        return sample

    @staticmethod
    def _resample(points, k):
        """Resamples the points such that there is exactly k points.

        If the input point cloud has <= k points, it is guaranteed the
        resampled point cloud contains every point in the input.
        If the input point cloud has > k points, it is guaranteed the
        resampled point cloud does not contain repeated point.

        """

        if k <= points.shape[0]:
            rand_idxs = np.random.choice(points.shape[0], k, replace=False)
            return points[rand_idxs, :]
        else:
            rand_idxs = np.concatenate([np.random.choice(points.shape[0], points.shape[0], replace=False),
                                        np.random.choice(points.shape[0], k - points.shape[0], replace=True)])
            return points[rand_idxs, :]


class RandomJitter:
    """ generate perturbations """

    def __init__(self, scale=0.005, clip=0.005):
        self.scale = scale
        self.clip = clip

    def jitter(self, pts):

        noise = np.clip(np.random.normal(0.0, scale=self.scale, size=(pts.shape[0], 2)),
                        a_min=-self.clip, a_max=self.clip)
        pts[:, :2] += noise  # Add noise to xyz

        return pts

    def __call__(self, sample):

        sample['points_src'] = self.jitter(sample['points_src'])
        sample['points_ref'] = self.jitter(sample['points_ref'])

        return sample


class ShufflePoints:
    """Shuffles the order of the points"""

    def __call__(self, sample):

        sample['points_ref'] = np.random.permutation(sample['points_ref'])
        sample['points_src'] = np.random.permutation(sample['points_src'])

        return sample


class SetDeterministic:
    """Adds a deterministic flag to the sample such that subsequent transforms
    use a fixed random seed where applicable. Used for test"""

    def __call__(self, sample):
        sample['deterministic'] = True

        return sample
