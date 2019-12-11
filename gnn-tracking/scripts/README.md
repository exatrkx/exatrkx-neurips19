## Scripts to make doublets and graphs for training

### Make Doublets
Initial doublets are made from any combination of the two hits from adjacent layers.

```
make_pairs_for_training_segments configs/make_doublets.yaml 21000 --workers 8
```

to make true doublets

```
srun -n 4 -c 32 make_true_pairs_for_training_segments_mpi configs/make_true_doublets.yaml 10
```
