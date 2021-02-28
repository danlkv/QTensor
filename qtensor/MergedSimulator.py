import qtensor
import pyrofiler

class MergedSimulator(qtensor.QtreeSimulator):
    def simulate_batch(self, qc, batch_vars=0, peo=None, dry_run=False):
        self._new_circuit(qc)
        self._create_buckets()
        free_final_qubits = list(range(batch_vars))
        self._set_free_qubits(free_final_qubits)

        if peo is None:
            peo, self.tn = self.optimizer.optimize(self.tn)
            if self.max_tw:
                if self.optimizer.treewidth > self.max_tw:
                    raise ValueError(f'Treewidth {self.optimizer.treewidth} is larger than max_tw={self.max_tw}.')

        all_indices = sum([list(t.indices) for bucket in self.tn.buckets for t in bucket], [])
        identity_map = {v.name: v for v in all_indices}
        self.peo = [identity_map[i.name] for i in peo]
        perm_dict = self._reorder_buckets()
        slice_dict = self._get_slice_dict()
        sliced_buckets = self.tn.slice(slice_dict)

        #-- restore indices after slicing
        all_indices = sum([list(t.indices) for bucket in sliced_buckets for t in bucket], [])
        identity_map = {v.identity: v for v in all_indices}
        new_permute = {k: identity_map[v.identity] for
                       k,v in perm_dict.items()}



        #-- Merged ix handling
        # Get tensor indices for merging subroutine
        bucket_ix = [[set(t.indices) for t in bucket]
                     for bucket in sliced_buckets
                    ]
        # Indices are already permuted, adjust peo to this
        perm_peo = [new_permute[x] for x in self.peo]
        #print('Permuted peo:', perm_peo)
        #print('mat tw:', self.optimizer.treewidth)
        with pyrofiler.timing('Finding mergeable'):
            merged_ix, width = qtensor.utils.find_mergeable_indices(perm_peo, bucket_ix)
        if dry_run:
            return peo, max(width)
        merged_buckets = []
        ibunch = []
        for ixs in merged_ix:
            bbunch_ = [sliced_buckets[i] for i in ixs]
            ibunch.append([perm_peo[i] for i in ixs])
            merged_buckets.append(sum(bbunch_, []))
        #print('ibunch', ibunch)
        #print('merged largest', max(width))
#       with pyrofiler.timing('only contract'):
#           result = qtensor.merged_indices.bucket_elimination(
#               merged_buckets,
#               ibunch,
#               self.backend.process_bucket_merged,
#               n_var_nosum=len(self.tn.free_vars)
#           )

        result = qtensor.merged_indices.bucket_elimination(
            merged_buckets,
            ibunch,
            self.backend.process_bucket_merged,
            n_var_nosum=len(self.tn.free_vars)
        )
        #--
        return self.backend.get_result_data(result).flatten()


class MergedQAOASimulator(qtensor.QAOASimulator.QAOASimulator, MergedSimulator):
    pass
