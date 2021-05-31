import qtensor
import pyrofiler

class MergedSimulator(qtensor.QtreeSimulator):
    def _all_bucket_indices(self):
        """
        Get all indices in current buckets

        Uses: self.tn.buckets

        Returns:
            list[indices of bucket]
        """
        buckets = self.tn.buckets
        all_indices = set(sum([list(t.indices) for bucket in buckets for t in bucket], []))
        return all_indices

    def _slice_buckets(self, slice_dict):
        """
        Args: 
            slice_dict [any:any]:
                dictionary of mapping from:to
        Uses:
            self.tn
        Modifies:
            self.tn.buckets
            self.peo
        """
        sliced_buckets = self.tn.slice(slice_dict)
        self.tn.buckets = sliced_buckets
        all_indices = self._all_bucket_indices()
        identity_map = {v.identity: v for v in all_indices}

        #-- restore indices after slicing
        # because slicing operation creates new variable objects
        self.peo = [identity_map[x.identity] for x in self.peo]
        return identity_map

    def _merge_buckets(self, merged_ix):
        """
        Merges in-place the self.tn.buckets

        Args:
            merged_ix: list[int]

        Uses:
            self.peo
            self.tn

        Modifies:
            self.tn.buckets
            self.ibunch
        """
        buckets = self.tn.buckets
        merged_buckets = []
        var_bunch = []
        for var_ixs in merged_ix:
            bucket_bunch = [buckets[i] for i in var_ixs]
            var_bunch.append([self.peo[i] for i in var_ixs])
            # flatten the bucket bunch into a merged bucket
            merged_buckets.append(sum(bucket_bunch, []))
        self.tn.buckets = merged_buckets
        self.ibunch = var_bunch

    def _convert_peo_raw(self, peo):
        """
        Process input peo

        if ``peo'' is None than find the ordering
        then relabel it to use idintities from curent buckets
        """
        if peo is None:
            peo, self.tn = self.optimizer.optimize(self.tn)
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')

        all_indices = self._all_bucket_indices()
        identity_map = {v.name: v for v in all_indices}
        return [identity_map[i.name] for i in peo]


    def simulate_batch(self, qc, batch_vars=0, peo=None, dry_run=False):
        self._new_circuit(qc)
        self._create_buckets()
        free_final_qubits = list(range(batch_vars))
        self._set_free_qubits(free_final_qubits)
        self.peo = self._convert_peo_raw(peo)
        perm_dict = self._reorder_buckets()
        #-- Merged ix handling
        # Get tensor indices for merging subroutine
        bucket_ix = [[set(t.indices) for t in bucket]
                     for bucket in self.tn.buckets
                    ]
        with pyrofiler.timing('Finding mergeable'):
            merged_ix, width = qtensor.utils.find_mergeable_indices(self.peo, bucket_ix)

        self._merge_buckets(merged_ix)


        slice_dict = self._get_slice_dict()
        identity_map = self._slice_buckets(slice_dict)
        self.ibunch = [[identity_map[int(y)] for y in x] for x in self.ibunch]
        #-- 
        # A dirty workaround to pass the merged buckets to benchmark optimizaiton code
        # TODO: decompose the functions to be able to get buckets
        self.merged_buckets = self.tn.buckets
        self.ibunch = self.ibunch

        if dry_run:
            return peo, max(width)

        result = qtensor.merged_indices.bucket_elimination(
            self.tn.buckets,
            self.ibunch,
            self.backend.process_bucket_merged,
            n_var_nosum=len(self.tn.free_vars)
        )
        #--
        return self.backend.get_result_data(result).flatten()


class MergedQAOASimulator(qtensor.QAOASimulator.QAOASimulator, MergedSimulator):
    pass

class MergedQAOASimulatorSymmetryAccelerated(MergedSimulator, qtensor.QAOASimulator.QAOAQtreeSimulatorSymmetryAccelerated):
    pass
